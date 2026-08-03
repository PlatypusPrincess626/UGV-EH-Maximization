"""
Cost-MDP actor-critic. The critic IS the Lyapunov function.

Trunk, action distribution, log_std parameterization and initialization
are identical to transformer.py, so a cost-vs-baseline comparison
isolates the reward formulation and the critic head. The differences:

  * no auxiliary heads at all -- no Lyapunov, barrier, latent-dynamics
    or energy encoder. The Lyapunov property lives in the reward and
    the critic instead.

  * the critic ends in -Softplus, so the value is <= 0 and
    V_cost = -value >= 0. Under a cost MDP (r = -c, c >= 0) the value
    function is exactly the negative cost-to-go, so this is the natural
    range rather than a constraint fighting the estimator.

  * the critic layers are spectral-normalized, bounding the Lipschitz
    constant. This is what lets a pointwise decrease check extend to a
    region rather than only to the sampled states.

WHY -SOFTPLUS RATHER THAN SOFTPLUS

A reward critic must span both signs -- measured on the reward MDP,
8.06% of return-to-go targets were negative, reaching -10.66. Softplus
cannot represent those. Under a cost MDP the sign flips: every return
is <= 0, so -Softplus covers the whole range with nothing clipped, and
V_cost = -value is non-negative with zero at the goal.

THE LYAPUNOV CONDITIONS, FOR REFERENCE

    V_cost >= 0                     by construction (-Softplus)
    V_cost = 0 at the goal          when all costs are zero forever:
                                    parked, in sun, at target charge
    dV_cost < 0                     from the Bellman equation:
                                      V(s) = c(s) + gamma*V(s')
                                    => dV = -c(s) - (1-gamma)*V(s')

Note the last line: (1-gamma) plays the role of the decay rate alpha.
With gamma = 0.99 that is 0.01, which is why LYAPUNOV_ALPHA is set to
1 - GAMMA for this variant rather than the 0.001 the lyapunov arm used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

try:                                        # torch >= 2.0
    from torch.nn.utils.parametrizations import spectral_norm
except ImportError:                         # older torch
    from torch.nn.utils import spectral_norm

# Mirrors transformer.py and lyupnov_transformer.py.
LOG_STD_MIN = -4.0
LOG_STD_MAX = 0.5
RAW_LOG_STD_TARGET = 0.3


class CostTransformerActorCritic(nn.Module):
    """Input: [batch, sequence, flattened_patch + scalar_features]."""

    _TANH_EPS = 1e-6

    def __init__(self, view_dist, scalar_dim=8, action_dim=2, d_model=128,
                 nhead=4, num_layers=2, dim_feedforward=256,
                 sequence_length=32, dropout=0.0,
                 spectral_critic=True):
        super().__init__()

        self.patch_dim = (2 * int(view_dist) + 1) ** 2
        self.input_dim = self.patch_dim + scalar_dim

        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.attention_pool = nn.Linear(d_model, 1)

        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

        # Critic depth matches the other two variants.
        L1 = int(2 * d_model)
        L3 = int(d_model / 2)

        def lin(a, b):
            layer = nn.Linear(a, b)
            # Spectral normalization bounds each layer's largest
            # singular value at 1, so the composed network is
            # Lipschitz with a computable constant. Without it the
            # decrease condition could only ever be claimed at the
            # sampled states.
            return spectral_norm(layer) if spectral_critic else layer

        self.critic_body = nn.Sequential(
            lin(d_model, L1), nn.GELU(),
            lin(L1, d_model), nn.GELU(),
            lin(d_model, L3), nn.GELU(),
            lin(L3, 1),
        )

        self.log_std_param = nn.Parameter(torch.zeros(action_dim))

        self.apply(self._initialize_weights)

        POLICY_OUTPUT_GAIN = 0.01
        policy_out = [m for m in self.actor if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            policy_out.weight.mul_(POLICY_OUTPUT_GAIN)
            if policy_out.bias is not None:
                policy_out.bias.zero_()

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            # Spectral-normalized layers do not own a plain `weight`
            # Parameter -- they own a reparameterized tensor, and the
            # NAME depends on which of the two APIs was imported at
            # the top of this file:
            #
            #   torch.nn.utils.spectral_norm (legacy)
            #       -> module.weight_orig, a Parameter
            #   torch.nn.utils.parametrizations.spectral_norm (>=2.0)
            #       -> module.parametrizations.weight.original
            #
            # Under the >= 2.0 API `module.weight` is a PROPERTY that
            # recomputes original / sigma_max on every access, so
            # nn.init.xavier_uniform_(module.weight) writes into a
            # temporary that is discarded the moment it returns --
            # silently, since the init runs under no_grad and raises
            # nothing. Checking only for `weight_orig` (the legacy
            # name) meant that on torch >= 2.0 -- the branch this file
            # prefers -- the critic was never Xavier-initialized at
            # all and kept nn.Linear's default Kaiming-uniform init.
            if hasattr(module, "weight_orig"):
                nn.init.xavier_uniform_(module.weight_orig)
            elif (hasattr(module, "parametrizations")
                  and "weight" in module.parametrizations):
                nn.init.xavier_uniform_(
                    module.parametrizations.weight.original)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, sequence):
        x = self.input_projection(sequence.float())
        x = x + self.position_embedding[:, :x.size(1)]
        x = self.encoder(x)
        w = torch.softmax(self.attention_pool(x), dim=1)
        return (x * w).sum(dim=1)

    def critic(self, latent):
        """
        Value <= 0. V_cost = -value >= 0 is the Lyapunov function.

        Softplus approaches zero asymptotically rather than reaching
        it, which is why LYAPUNOV_BALL exists -- the certificate is
        stated over a ball around the goal, not at a single point.

        RANGE, AND WHY THE REWARD IS SCALED

        Spectral norm makes critic_body 1-Lipschitz in `latent`, and
        `latent` is a convex combination of LayerNorm'd tokens, so
        ||latent|| <= sqrt(d_model) ~ 11.3. The pre-Softplus output
        therefore spans only ~+-11 around whatever offset the biases
        supply, and biases are the one unconstrained path -- AdamW
        moves them at ~lr per step. Targets must land inside that
        range and within reach of the run's optimizer-step budget,
        which is what main.py's COST_REWARD_SCALE = 1 - GAMMA is for:
        it puts TD(lambda) returns at ~-0.23 instead of ~-23.
        """
        return -F.softplus(self.critic_body(latent)).squeeze(-1)

    def forward(self, sequence):
        latent = self.encode(sequence)
        raw_mean = self.actor(latent)
        raw_log_std = self.log_std_param.unsqueeze(0).expand(raw_mean.shape[0], -1)
        return raw_mean, raw_log_std, self.critic(latent), latent

    def _squash(self, z):
        action = torch.tanh(z)
        correction = torch.log(1.0 - action.pow(2) + self._TANH_EPS)
        return action, correction

    def distribution(self, sequence):
        raw_mean, raw_log_std, value, latent = self(sequence)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(raw_log_std) + 1.0
        )
        return Normal(raw_mean, log_std.exp()), value, latent, raw_log_std

    def act(self, sequence, deterministic=False):
        dist, value, _latent, _rls = self.distribution(sequence)
        z = dist.mean if deterministic else dist.rsample()
        action, correction = self._squash(z)
        log_prob = (dist.log_prob(z) - correction).sum(dim=-1)
        return action, z, log_prob, value

    def evaluate_actions(self, sequence, actions):
        """`actions` are pre-squash z values, as stored during rollout."""
        dist, value, _latent, raw_log_std = self.distribution(sequence)

        raw_mean = dist.mean
        z = actions
        _, correction = self._squash(z)

        log_probs = (dist.log_prob(z) - correction).sum(dim=-1)
        entropy = (dist.entropy() + correction).sum(dim=-1)

        mean_std = dist.stddev.mean()
        mean_raw_log_std = raw_log_std.mean()
        raw_log_std_reg = F.relu(raw_log_std.abs() - RAW_LOG_STD_TARGET).mean()

        return (log_probs, entropy, value, mean_std,
                mean_raw_log_std, raw_log_std_reg, raw_mean)

    def value_only(self, sequence):
        """Value for the NEXT state, without the policy head."""
        return self.critic(self.encode(sequence))
