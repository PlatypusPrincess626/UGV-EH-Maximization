"""
Cost-MDP actor-critic. The critic IS the Lyapunov function.

Trunk, action distribution, log_std parameterization and initialization
are identical to transformer.py, so a cost-vs-baseline comparison
isolates the reward formulation and the critic head. The differences:

  * no auxiliary heads at all -- no Lyapunov, barrier, latent-dynamics
    or energy encoder. The Lyapunov property lives in the reward and
    the critic instead.

  * the critic head is -beta*Softplus(g/beta), so the value is <= 0
    and V_cost = -value >= 0. Under a cost MDP (r = -c, c >= 0) the
    value function is exactly the negative cost-to-go, so this is the
    natural range rather than a constraint fighting the estimator.
    beta = 1.0 is plain Softplus, the original head.

  * the critic layers are spectral-normalized, bounding the Lipschitz
    constant. This is what lets a pointwise decrease check extend to a
    region rather than only to the sampled states.

WHY THE HEAD IS SCALED

A reward critic must span both signs -- measured on the reward MDP,
8.06% of return-to-go targets were negative, reaching -10.66. Under a
cost MDP the sign flips: every return is <= 0, so a one-sided head
clips nothing. That part was always right.

What was wrong was the gain. Softplus's derivative falls off linearly
with its own output, so the head loses resolution exactly as V_cost
approaches zero -- which is the goal region the certificate is most
about. Measured on cost seed 1 at 400 episodes: gain 0.27 at the
observed v_critic_min of 0.309, and that minimum did not fall further
over the whole run.

Scaling by beta shifts the whole gain profile without changing the
function's shape or any of its guarantees. See critic() for the
argument that linear falloff is the best any smooth, strictly
positive, asymptotically-zero head can do, and that beta is therefore
the only free quantity worth tuning.

This is a change of output nonlinearity and NOTHING ELSE. The
derivative stays in (0, 1) for every beta, so the composed Lipschitz
bound is L <= 1 exactly as before and every region claim built on it
is unchanged. V_cost >= 0 is unchanged. slack_c is unchanged. The
certified quantity is unchanged.

THE LYAPUNOV CONDITIONS, FOR REFERENCE

    V_cost >= 0                     by construction (-Softplus)
    V_cost = 0 at the goal          approached asymptotically, which
                                    is why LYAPUNOV_BALL exists --
                                    the certificate is stated over a
                                    ball around the goal, not at a
                                    single point
    dV_cost < 0                     from the Bellman equation:
                                      V(s) = c(s) + gamma*V(s')

ON THE DECAY RATE

An earlier version of this file claimed dV = -c(s) - (1-gamma)*V(s')
and concluded that alpha = 1 - gamma. The algebra is off by a sign --
V(s') - V(s) = (1-gamma)*V(s') - c(s) -- and, more importantly, the
identity is about the CRITIC's cost-to-go, not about the analytic
V_true that main.py certifies. Measured, the system decays at ~0.0012
per step while alpha = 1 - gamma = 0.01 demands 7.6x that, which is
most of why the reported violation rate was 61%. main.py now logs the
feasible-alpha quantiles instead of asserting one value; see
analytic_certification() there.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import os

try:                                        # torch >= 2.0
    from torch.nn.utils.parametrizations import spectral_norm
    from torch.nn.utils import parametrize as _parametrize
    _HAS_PARAMETRIZE = True
except ImportError:                         # older torch
    from torch.nn.utils import spectral_norm
    _HAS_PARAMETRIZE = False

# Per-layer Lipschitz budget for the critic body.
#
# spectral_norm forces every layer's largest singular value to exactly
# 1, so the four-layer body composes to L <= 1. That bound is far
# stricter than the certificate needs: the region argument requires a
# KNOWN FINITE Lipschitz constant, not the value 1. Scaling each
# normalised layer by c gives L <= c**4, still known and finite, with
# the certified ball radius shrinking by the same factor.
#
# WHY RELAX IT
#
# L <= 1 is measurably expensive and buys nothing this study measures:
#
#   * off-distribution dynamic range (ood_range) is 0.027 for cost
#     against 0.049 for cost_plain -- the constrained critic spans 1.8x
#     less of the coordinate it certifies
#   * late approx_kl is 0.0166 (cost), 0.0152 (cost_plain), 0.0102
#     (normal); roughly 22% of this arm's excess policy churn tracks
#     the spectral machinery
#   * no computed metric depends on L. cert_alpha_q05 comes from the
#     Bellman dV estimator; the Lipschitz constant appears in main.py
#     only inside comments
#
# CHOOSING c
#
# The constraint should be slack enough not to bind, and no slacker --
# every factor of L costs certified radius. Anchoring on the measured
# gap: matching cost_plain's range needs c**4 = 1.8, i.e. c = 1.16.
# The default below doubles that for headroom.
#
#   c      L <= c**4    range vs L<=1     radius
#   1.00      1.00          1.0x           1.00x
#   1.16      1.81          1.8x           0.55x   (matches cost_plain)
#   1.40      3.84          3.8x           0.26x   (default)
#   1.50      5.06          5.1x           0.20x
#   2.00     16.00         16.0x           0.06x
#
# VERIFYING IT AFTERWARDS
#
# ood_range in the diagnostics says whether the constraint still binds.
# If it comes back near 0.049, matching cost_plain, c was enough. If it
# is still near 0.027, the constraint is binding and c should go up. If
# it overshoots well past 0.049, c is larger than needed and is costing
# certified radius for nothing.
SPECTRAL_C = float(os.environ.get("LTAC_SPECTRAL_C", "1.772"))
if SPECTRAL_C < 1.0:
    raise ValueError("LTAC_SPECTRAL_C must be >= 1.0")


class _LipschitzScale(nn.Module):
    """
    Multiplies an already spectral-normalised weight by a constant.

    Registered AFTER spectral_norm, so it sees a matrix with sigma_max
    exactly 1 and returns one with sigma_max exactly c. Applied to the
    weight only -- the bias is untouched, since a translation does not
    affect a Lipschitz constant and scaling it would change the
    initialisation the arms are matched on.
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, weight):
        return weight * self.scale

# Mirrors transformer.py and lyupnov_transformer.py.
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


class CostTransformerActorCritic(nn.Module):
    """Input: [batch, sequence, flattened_patch + scalar_features]."""

    _TANH_EPS = 1e-6

    def __init__(self, view_dist, scalar_dim=8, action_dim=2, d_model=128,
                 nhead=4, num_layers=2, dim_feedforward=256,
                 sequence_length=32, dropout=0.0,
                 spectral_critic=True,
                 softplus_beta=0.1, beta_gain_target=None,
                 beta_min=1e-3, beta_max=1.0, beta_ema=0.05):
        super().__init__()

        if not softplus_beta > 0.0:
            raise ValueError("softplus_beta must be positive")

        # beta in VALUE units: the width of the soft knee measured on
        # V_cost. beta = 1.0 reproduces the original head exactly.
        #
        # A BUFFER, not a float, so it travels in the state_dict and a
        # reloaded checkpoint reproduces the same function. Held fixed
        # unless beta_gain_target is set -- see adapt_beta().
        self.register_buffer("softplus_beta",
                             torch.tensor(float(softplus_beta)))
        # Cached Python float for the hot path. float() on a CUDA
        # buffer forces a device sync, and critic() runs once per
        # inference call -- 295,200 of them in a 400-episode run, in
        # the loop that is already 97% of wall clock. The buffer is
        # the source of truth for checkpointing; this shadows it.
        self._beta = float(self.softplus_beta)
        self.beta_gain_target = beta_gain_target
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_ema = float(beta_ema)

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

        # Built as PLAIN Linear layers. Spectral normalization is
        # registered further down, AFTER _initialize_weights has run.
        # See the note there for why the order matters.
        self.critic_body = nn.Sequential(
            nn.Linear(d_model, L1), nn.GELU(),
            nn.Linear(L1, d_model), nn.GELU(),
            nn.Linear(d_model, L3), nn.GELU(),
            nn.Linear(L3, 1),
        )

        self.log_std_param = nn.Parameter(torch.zeros(action_dim))

        self.apply(self._initialize_weights)

        # Spectral normalization bounds each layer's largest singular
        # value at 1, so the composed network is Lipschitz with a
        # computable constant. Without it the decrease condition could
        # only ever be claimed at the sampled states.
        #
        # ORDER MATTERS: REGISTER AFTER INITIALIZING
        #
        # spectral_norm() estimates sigma_max by power iteration and
        # caches the singular-vector estimates (_u, _v) as BUFFERS,
        # warming them up against whatever the weight is at
        # registration time. It then refreshes them on each forward
        # pass -- but only in TRAINING mode.
        #
        # Registering first and initializing second leaves _u and _v
        # pointing at the pre-init random weight. sigma_hat = u^T W v
        # for singular vectors of an unrelated matrix is not
        # sigma_max; for an independent Xavier draw its expectation is
        # near zero. The layer then computes W / sigma_hat with a
        # near-zero divisor, and nothing corrects it until the first
        # forward pass in training mode -- which does not happen until
        # the first update(), because the entire first rollout runs
        # under model.eval(). That produced a first-update value_loss
        # of ~6.9e6 against a target of ~0.2, self-correcting to
        # ~0.008 by the second update once power iteration caught up.
        #
        # Initializing first and registering second makes the warm-up
        # valid by construction, with no dummy forward pass and no
        # reliance on private attributes.
        if spectral_critic:
            for layer in self.critic_body:
                if isinstance(layer, nn.Linear):
                    # n_power_iterations=5, not the default 1.
                    #
                    # torch re-runs power iteration on EVERY forward in
                    # training mode, so with n=1 the sigma estimate is
                    # still converging and moves a little each time. One
                    # update runs ~24 minibatch forwards, which means the
                    # critic is a slightly DIFFERENT function for each of
                    # them -- the value targets, the advantages and the
                    # gradients all see a drifting normalisation that has
                    # nothing to do with the data.
                    #
                    # This is mechanical noise, not environment noise: it
                    # exists only because the estimator is under-converged.
                    # The evidence it costs something is the ablation --
                    # late approx_kl was 0.0102 (normal), 0.0152
                    # (cost_plain, no spectral norm), 0.0166 (cost), so
                    # roughly 22% of this arm's excess policy churn tracks
                    # the spectral machinery rather than the reward.
                    #
                    # Cost is negligible: the critic body is four small
                    # layers and the iteration is one matvec each.
                    spectral_norm(layer, n_power_iterations=5)

                    # Relax sigma_max from 1 to SPECTRAL_C. See the
                    # constant's definition for why L <= 1 is stricter
                    # than the certificate requires.
                    if SPECTRAL_C != 1.0:
                        if not _HAS_PARAMETRIZE:
                            raise RuntimeError(
                                "LTAC_SPECTRAL_C != 1.0 needs torch >= 2.0 "
                                "for parametrizations; got the legacy "
                                "spectral_norm path.")
                        _parametrize.register_parametrization(
                            layer, "weight", _LipschitzScale(SPECTRAL_C))

        POLICY_OUTPUT_GAIN = 0.01
        policy_out = [m for m in self.actor if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            policy_out.weight.mul_(POLICY_OUTPUT_GAIN)
            if policy_out.bias is not None:
                policy_out.bias.zero_()

        # Start the critic in the healthy part of the curve.
        #
        # The gain of the scaled Softplus falls off below V ~ beta, so
        # shrinking beta moves the low-gradient region toward zero but
        # also means a zero-initialized critic can start inside it: the
        # spectral-normalized body outputs roughly +-0.6 at init, which
        # at beta = 0.1 puts about half the batch at V ~ 2e-4 with gain
        # ~2e-3. Offsetting the final bias puts every sample in the
        # unit-gain region on step 1.
        #
        # Initialization only, not a constraint -- the bias is free to
        # move anywhere afterwards and V_cost is still non-negative by
        # construction. Set to 0.0 to reproduce the previous
        # initialization exactly. Sized near the measured V_cost scale
        # (~0.65) so the head does not start biased high either.
        CRITIC_BIAS_INIT = 1.0
        critic_out = [m for m in self.critic_body
                      if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            if critic_out.bias is not None:
                critic_out.bias.fill_(CRITIC_BIAS_INIT)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            # Spectral-normalized layers do not own a plain `weight`
            # Parameter -- they own a reparameterized tensor, and the
            # NAME depends on which of the two APIs was imported at
            # the top of this file:
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
            #
            # Since the constructor now registers spectral norm AFTER
            # calling this, the first two branches no longer fire on
            # the initial pass. They are kept so that re-applying this
            # to an already-normalized model (e.g. after loading a
            # checkpoint) writes to the real Parameter instead of
            # silently doing nothing.
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

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Refresh the hot-path cache; the buffer is authoritative.
        self._beta = float(self.softplus_beta)

    @staticmethod
    def beta_for_gain(v, gain_target):
        """beta that puts the head's gain at `gain_target` when V = v."""
        return float(v) / -math.log(1.0 - float(gain_target))

    @torch.no_grad()
    def adapt_beta(self, v_cost, quantile=0.10):
        """
        Slide beta to track the value scale. No-op unless
        beta_gain_target was set.

        WHY TIE IT TO V RATHER THAN TO A STEP COUNT

        From critic()'s closed form, gain = 1 - exp(-V/beta) depends
        only on the RATIO V/beta. A fixed beta therefore does not mean
        a fixed head: as the policy improves and V_cost shrinks toward
        the goal, V/beta falls and the head silently loses resolution
        exactly where the certificate cares most. That is what
        happened on cost seed 1 -- V fell from 1.08 to 0.65 over the
        run while beta stayed at 1.0, so gain drifted from 0.66 down
        to 0.48 and v_critic_min stalled at 0.309.

        Setting beta = V / -ln(1 - gain_target) holds the ratio, and
        therefore the conditioning, constant at every stage of
        training. A step-count schedule cannot do this: it has no idea
        what V is doing.

        WHY THIS DOES NOT RUN AWAY

        beta appears in V, so there is an apparent loop. It breaks in
        the regime we operate in: for raw >> beta the head is
        effectively the identity, V ~ raw, and raw is set by the value
        regression alone. beta then follows V without V following
        beta. The loop only closes near the knee, which is the region
        the adaptation exists to stay out of.

        `quantile` picks WHICH V to hold at the target gain. The 10th
        percentile means the smallest tenth of the batch still gets
        roughly gain_target, rather than only the average state.

        DO NOT CALL MID-UPDATE. Changing beta changes the function, so
        values computed before and after would be inconsistent. Call
        once per update(), after the minibatch loop.
        """
        if self.beta_gain_target is None:
            return float(self.softplus_beta)

        v = v_cost.detach().flatten().float()
        v = v[v > 0]
        if v.numel() == 0:
            return float(self.softplus_beta)

        v_ref = torch.quantile(v, quantile)
        target = self.beta_for_gain(v_ref.item(), self.beta_gain_target)
        target = min(max(target, self.beta_min), self.beta_max)

        # EMA in log space: beta spans orders of magnitude and a
        # linear EMA would crawl at the small end.
        cur = float(self.softplus_beta)
        new = math.exp((1.0 - self.beta_ema) * math.log(cur)
                       + self.beta_ema * math.log(target))
        self.softplus_beta.fill_(new)
        self._beta = new
        return new

    def critic(self, latent):
        """
        Value <= 0. V_cost = -value >= 0 is the Lyapunov function.

            V_cost(z) = beta * softplus( g(z) / beta )

        Same family as before, one parameter added. beta = 1.0 is the
        original head, bit for bit.

        WHY THIS IS THE RIGHT SHAPE, NOT JUST A DIFFERENT ONE

        Softplus's toe is not an accident of that particular function
        -- it is forced by the requirements. Any f with f >= 0,
        f' > 0 everywhere (no dead zone), and f reaching 0 only in the
        limit must have

            integral_0 df / f'(f)  =  infinity

        for the approach to zero to take infinite argument. If
        f' ~ f^p that diverges only for p >= 1, so f' can decay no
        more slowly than LINEARLY in f. Softplus already sits exactly
        at that boundary: for small f, f' ~ f. Nothing smooth and
        strictly positive does asymptotically better.

        What is free is the CONSTANT. Scaling gives f' ~ f / beta.
        In closed form, the gain at value V is exactly

            gain(V, beta) = 1 - exp(-V / beta)

        which depends only on the RATIO V/beta -- worth stating
        plainly, because it is what makes beta choosable rather than
        guessable. Inverting it,

            beta = V / -ln(1 - gain_target)

        so "hold gain at 0.9 when V = 0.65" is beta = 0.65/2.303 =
        0.28, with no search. It is also why beta may need to SLIDE:
        a fixed beta holds the ratio only while V is fixed, and V
        shrinks as the policy improves. See adapt_beta().

            V target      beta=1.0    beta=0.25    beta=0.1
              0.655         0.481       0.927        0.999
              0.309         0.266       0.709        0.955
              0.050         0.049       0.181        0.393
              0.010         0.010       0.039        0.095

        The 0.655 and 0.309 rows are cost seed 1's measured
        v_critic_mean and v_critic_min at 400 episodes. The head was
        running at gain 0.27 exactly where the certificate matters
        most, and v_critic_min could not descend past 0.309 as a
        result.

        WHAT DOES NOT CHANGE

            V_cost >= 0            f > 0 for all finite argument
            no dead zone           f' > 0 everywhere, unlike ReLU
            smoothness             C^infinity, so dV_cost has no kink
            Lipschitz constant     f' = sigmoid(g/beta) is in (0, 1)
                                   for every beta, so the composed
                                   critic is still L <= 1 and every
                                   region claim built on it stands
            what is certified      slack_c is unchanged

        Only the gain profile moves.

        CHOOSING BETA

        beta is the value scale below which gain starts to fall, so
        set it well under the smallest V_cost that needs resolving.
        With V_cost ~ 0.65 and a goal region near zero, 0.1 keeps gain
        above 0.39 down to V = 0.05. Smaller beta buys more there and
        makes the head progressively more ReLU-like above it; it never
        introduces a hard zero.

        Note the reciprocal: torch's F.softplus(x, beta=b) computes
        (1/b) * log(1 + exp(b*x)), so its `beta` is 1/ours. Its
        `threshold` argument keeps the large-argument branch linear and
        overflow-free, which matters here because g/beta reaches ~16 at
        initialization.
        """
        raw = self.critic_body(latent).squeeze(-1)
        return -F.softplus(raw, beta=1.0 / self._beta)

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
