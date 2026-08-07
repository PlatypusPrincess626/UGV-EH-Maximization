"""
Baseline actor-critic for the Lyapunov comparison.

This is deliberately the Lyapunov model MINUS its auxiliary machinery.
Everything that is not the Lyapunov contribution is mirrored exactly:
the trunk, the action distribution, the log_std parameterization and
the initialization.

WHY IT IS BUILT THIS WAY
------------------------
The original baseline differed from the Lyapunov model in several ways
that had nothing to do with Lyapunov functions, and each one would have
shown up in the results as if it did:

  * no learnable positional embedding
  * no closing LayerNorm on the encoder, so the residual stream's scale
    at initialization was whatever fell out of accumulated residual
    additions -- and that unnormalized latent fed straight into the
    actor
  * mean-of-last-timestep instead of attention pooling
  * a 2-layer critic instead of 4
  * Normal(tanh(mean), sigma) with the sample CLAMPED to +-0.999, and
    log_prob scored under the unclamped Normal -- so probability mass
    at the boundary was misattributed, biasing the gradient exactly
    where a parking policy operates
  * an unbounded free log_std, where the Lyapunov model squashes it
    into [LOG_STD_MIN, LOG_STD_MAX]

Mirroring these leaves exactly one difference between the two arms: the
Lyapunov, barrier and latent-dynamics heads and their losses. That is
what the comparison is supposed to measure.

Interface matches the Lyapunov model where main.py needs it to:
`act()` returns (action, raw_action, log_prob, value) so the rollout
stores the PRE-SQUASH z, and `evaluate_actions()` re-applies the tanh
correction to that z.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Mirrors lyupnov_transformer.py.
LOG_STD_MIN = -4.0
# Tightened from 0.5.
#
# Exploration noise is applied in ACTION space and then scaled by
# MAX_MOVE_PER_STEP = 20, so a std of 0.215 is ~4.3 cells of random
# walk per step on top of the intended move. Measured on cost seed 1,
# the SAME policy sampled vs taken at the mean:
#
#                 stochastic   deterministic   ratio
#   path m             2427            550      4.4x
#   motion mAh         1645            398      4.1x
#   min battery %      7.54          21.04      0.36
#
# The deployed policy had 21% battery margin; the sampled one had 7.5%
# and died in 38% of episodes. The agent was being killed by its own
# exploration and then learning from a state distribution its own
# deterministic policy never visits.
#
# At MIN=-4 the squash is std = exp(-4 + (MAX+4)/2 * (tanh(raw)+1)), so
# MAX=-0.5 moves the raw=0 std from 0.174 to 0.105 and the ceiling
# under the RAW_LOG_STD_TARGET regulariser from ~0.335 to ~0.176 --
# roughly halving the positional noise. Mirrored across all three model
# files so the arms stay comparable.
LOG_STD_MAX = -0.5
RAW_LOG_STD_TARGET = 0.3

class TransformerActorCritic(nn.Module):
    """Temporal actor-critic. Input: [batch, sequence, flattened_patch + scalar_features]."""

    _TANH_EPS = 1e-6

    def __init__(self, view_dist, scalar_dim=7, action_dim=2, d_model=128,
                 nhead=4, num_layers=2, dim_feedforward=256,
                 sequence_length=32,
                 # 0.0, not 0.1. main.py records rollout log-probs under
                 dropout=0.0):
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

        L1 = int(2 * d_model)
        L3 = int(d_model / 2)
        self.critic = nn.Sequential(
            nn.Linear(d_model, L1),
            nn.GELU(),
            nn.Linear(L1, d_model),
            nn.GELU(),
            nn.Linear(d_model, L3),
            nn.GELU(),
            nn.Linear(L3, 1),
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
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, sequence):
        sequence = sequence.float()
        x = self.input_projection(sequence)
        x = x + self.position_embedding[:, :x.size(1)]
        x = self.encoder(x)
        attention_scores = self.attention_pool(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return (x * attention_weights).sum(dim=1)

    def forward(self, sequence):
        latent = self.encode(sequence)
        raw_mean = self.actor(latent)
        raw_log_std = self.log_std_param.unsqueeze(0).expand(raw_mean.shape[0], -1)
        critic = self.critic(latent).squeeze(-1)
        return raw_mean, raw_log_std, critic, latent

    def _squash(self, z):
        action = torch.tanh(z)
        correction = torch.log(1.0 - action.pow(2) + self._TANH_EPS)
        return action, correction

    def distribution(self, sequence):
        """
        Pre-squash Normal. Actions come from sampling this and applying
        tanh -- never from clamping, which is what the original baseline
        did and which misattributes probability mass at the boundary.
        """
        raw_mean, raw_log_std, critic, latent = self(sequence)

        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(raw_log_std) + 1.0
        )
        dist = Normal(raw_mean, log_std.exp())
        return dist, critic, latent, raw_log_std

    def act(self, sequence, deterministic=False):
        dist, critic, _latent, _raw_log_std = self.distribution(sequence)
        z = dist.mean if deterministic else dist.rsample()
        action, correction = self._squash(z)
        log_prob = (dist.log_prob(z) - correction).sum(dim=-1)
        return action, z, log_prob, critic

    def evaluate_actions(self, sequence, actions):
        """`actions` are pre-squash z values, as stored during rollout."""
        dist, critic, _latent, raw_log_std = self.distribution(sequence)

        raw_mean = dist.mean
        z = actions
        _, correction = self._squash(z)

        log_probs = (dist.log_prob(z) - correction).sum(dim=-1)
        entropy = (dist.entropy() + correction).sum(dim=-1)

        mean_std = dist.stddev.mean()
        mean_raw_log_std = raw_log_std.mean()
        raw_log_std_reg = F.relu(raw_log_std.abs() - RAW_LOG_STD_TARGET).mean()

        return (log_probs, entropy, critic, mean_std,
                mean_raw_log_std, raw_log_std_reg, raw_mean)
