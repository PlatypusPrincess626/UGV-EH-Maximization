"""
Certificate head after Manek & Kolter (2019), on the bounded observer.

WHAT THIS CHANGES AND WHY

The softplus head of cost_transformer gives V > 0 but reaches 0 only
as its pre-activation goes to -infinity, so condition cond_extrema
(V = 0 on Z*) is approached and never attained, and Assumption 4's
lower envelope kappa_1 has to be FITTED from rollouts. Fitting it is
the weak link the certification section rests on: a 5th-percentile
envelope over visited states is violated by 5% of that data by
construction and says nothing about states the policy never reaches,
so kappa_1^{-1} converts value into distance by regression rather
than by bound.

Manek & Kolter close both gaps with one head:

    V(x) = sigma( g(x) - g(x*) ) + eps * ||x - x*||^2

The SHIFT by g(x*) makes V vanish exactly at the target rather than
asymptotically. The QUADRATIC TERM makes V >= eps * ||x - x*||^2
identically, so a class-K lower bound is KNOWN rather than measured:
kappa_1(d) = eps d^2, with eps chosen.

Adapted here, the target set is defined by state of charge alone
(Z* = {b >= b_tgt}), so the quadratic term is taken on the SoC
deficit rather than on a latent reference point -- there is no
canonical h* to subtract, and the belief is explicitly not required
to converge to anything physical.

    V_cost(z) = beta * softplus( (g(h) - g_ref) / beta )  +  eps * d(z)^2
    d(z)      = relu(b_tgt - b)

g_ref is an EMA of g(h) over states already inside Z*, so the shift
tracks what the network currently assigns to the target set instead
of assuming a fixed reference. Before any such state is seen the
shift is zero and the head is the original one.

CONSEQUENCES FOR THE BOUND

  Assumption 3   V(z) = 0 for z in Z*, exactly, since d = 0 there and
                 softplus(0) contributes beta*log 2 -- see the note on
                 SOFTPLUS_AT_ZERO below.
  Assumption 4   kappa_1(d) = eps d^2 holds by construction. No fit.
  Lipschitz      the quadratic term adds 2 eps d_max to the constant
                 on a bounded domain, which is finite and computable.

EMPIRICAL ANCHOR FOR EPS

The fitted quadratic envelope on the existing bounded arm was
kappa_1 = 1.451 +- 0.150 across eight seeds. Setting eps there asks
the head for no more than the current critic already delivers, which
is why it is the default.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from cost_transformer import CostTransformerActorCritic

try:                                        # torch >= 2.0
    from torch.nn.utils.parametrizations import spectral_norm
except ImportError:                         # older torch
    from torch.nn.utils import spectral_norm


# Per-layer spectral budget for the encoder.
#
# Sized from the measured checkpoint, not guessed: the trained
# unconstrained encoder reached 1.78-5.33 depending on layer, so a
# uniform cap has to sit at the top of that range or it will bind on
# the projections that grew most. 6.0 gives the largest measured layer
# (5.334) about 12% headroom.
#
# Lower it to tighten the bound at the cost of capacity. The
# relationship is brutal -- the bound is a product over ~10 layers, so
# halving this divides the bound by ~1000 while halving every layer's
# expressiveness.
ENCODER_C = float(os.environ.get("LTAC_ENCODER_C", "2.0"))

# SEPARATE budget for the tied query/key projection, and a divisor on
# the L2 logits.
#
# WHY THIS IS NOT THE SAME KNOB
#
# spectral_norm FIXES sigma_max at exactly c -- it divides by the
# measured sigma, so the result is c, not "at most c". At c = 6.0
# every encoder layer therefore STARTS about 3x above its own Xavier
# scale (1.74 for the input projection, 1.93 for qk) rather than being
# capped near it. Sizing c from the trained softmax encoder's measured
# sigma_max was the wrong reference: those were values training grew
# INTO, not values to begin at.
#
# And the qk projection is worse than the rest, because the L2 logit
# is QUADRATIC in it:
#
#     A_ij = softmax( -||W x_i - W x_j||^2 / sqrt(d_head) )
#
# so doubling sigma_max quadruples the logit magnitude. With
# ||x|| ~ sqrt(d_model) ~ 11.3 after LayerNorm, the worst-case logit
# is (2*c*11.3)^2 / sqrt(32):
#
#     c = 1.0 ->    90      c = 3.0 ->   815
#     c = 2.0 ->   362      c = 6.0 ->  3258
#
# Every one of those saturates a softmax. The distance is exactly zero
# at i = j, so a saturated L2 attention collapses to each token
# attending only to itself: no temporal mixing, and no gradient
# through the attention path. That is consistent with the observed
# failure -- a run sitting at its episode-1 death rate 380 episodes
# in, rather than converging slowly.
#
# The fix is a temperature, not a smaller c alone. Dividing the logits
# by QK_TEMPERATURE * sqrt(d_head) rescales them into a usable range
# without shrinking the projection's capacity, and it enters the
# Lipschitz bound as a simple 1/T factor.
ENCODER_QK_C = float(os.environ.get("LTAC_ENCODER_QK_C", "1.0"))
QK_TEMPERATURE = float(os.environ.get("LTAC_QK_TEMP", "4.0"))
#
# 16.0 was the first working value -- it cleared the saturation that
# froze the run at its episode-1 death rate. But it went too far the
# other way. At T = 16 with qk_c = 1.0 the typical logit spread is only
# ~2.8, an attention weight ratio of ~17x across a 32-step window,
# which is close to uniform. A near-uniform attention hands the policy
# a smoothed average of the window: enough to predict value, which
# depends on aggregate state, and too blunt to act on. That matches
# what the c = 5.0 run showed -- EV climbing to 0.892 while |a| stayed
# flat at 0.017-0.022 against cost's 0.046, with clip fraction and KL
# both BELOW the other arms rather than above.
#
#     T      typical logit    weight ratio
#     16.0        2.8              17x        near-uniform
#      8.0        5.7             286x
#      4.0       11.3          8.2e4x         selective
#      2.0       22.6          6.7e9x
#      1.0       45.3            saturated
#
# 4.0 is sharp enough to select and an order of magnitude short of the
# saturation point. Watch attn_entropy to confirm rather than assume.

# LayerNorm epsilon.
#
# ||J_LayerNorm|| <= ||gamma||_inf / sqrt(var + eps) <= ||gamma||_inf /
# sqrt(eps), so eps is what makes the normalisation Lipschitz at all.
# PyTorch's 1e-5 gives 316 per norm and there are five of them, which
# alone would contribute 3e12. 1e-2 gives 10 per norm.
#
# This changes the forward pass slightly versus the other arms: at
# typical residual-stream variance the eps term is negligible either
# way, but it is a real difference and belongs in the writeup.
LAYERNORM_EPS = float(os.environ.get("LTAC_LN_EPS", "1e-2"))

# Clamp on the LayerNorm gain, so ||gamma||_inf is known rather than
# whatever training produced.
LAYERNORM_GAMMA_MAX = float(os.environ.get("LTAC_LN_GAMMA_MAX", "2.0"))


def _sn(layer, scale=None):
    """Spectral-normalise so sigma_max == scale (ENCODER_C by default)."""
    scale = ENCODER_C if scale is None else scale
    spectral_norm(layer, n_power_iterations=5)
    if scale != 1.0:
        try:
            from torch.nn.utils import parametrize
        except ImportError as exc:                     # pragma: no cover
            raise RuntimeError(
                "LTAC_ENCODER_C != 1.0 needs torch >= 2.0") from exc
        parametrize.register_parametrization(
            layer, "weight", _Scale(scale))
    return layer


class _Scale(nn.Module):
    """Rescales an already-normalised weight so sigma_max == scale."""

    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, weight):
        return weight * self.scale

    def right_inverse(self, weight):
        # Without this the registration-time consistency check sees
        # forward(W) = c*W != W. Cheap to provide, and it also makes
        # checkpoint restores that assign to .weight behave.
        return weight / self.scale


class L2SelfAttention(nn.Module):
    """
    Multi-head self-attention with L2 logits and TIED query/key
    projections.

        A_ij = softmax_j( -||W_h x_i - W_h x_j||^2 / sqrt(d_head) )
        out  = W_o concat_h( A_h V_h )

    The tying is what buys Lipschitzness -- with independent W_q and
    W_k the map is not Lipschitz even with both spectrally bounded,
    because the logit is bilinear in two independently varying terms.
    Tying makes it a function of a single projected difference.
    """

    def __init__(self, d_model, nhead):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        # One matrix serving as both query and key. NOT two matrices
        # initialised identically -- they would diverge on the first
        # gradient step and the guarantee with them.
        self.last_entropy = float("nan")
        # OFF by default. The entropy itself is cheap -- an elementwise
        # pass over the [batch, heads, N, N] attention matrix, about
        # 1/32 the cost of the matmul that produced it -- but the
        # float() conversion forces a GPU->CPU sync. Left unguarded
        # that fires on every attention forward: every step of every
        # rollout (720 per episode, since the rollout runs in eval
        # mode) and every minibatch of every PPO epoch. Serialising
        # the pipeline that often is a real slowdown for a number
        # nobody reads. Enabled for exactly one forward per update by
        # diagnostics.py.
        self.record_entropy = False
        self.qk_proj = _sn(nn.Linear(d_model, d_model, bias=False),
                           scale=ENCODER_QK_C)
        self.v_proj = _sn(nn.Linear(d_model, d_model, bias=False))
        self.out_proj = _sn(nn.Linear(d_model, d_model, bias=False))

    def forward(self, x):
        b, n, _ = x.shape
        h, dh = self.nhead, self.d_head

        qk = self.qk_proj(x).view(b, n, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, n, h, dh).transpose(1, 2)

        # -||q_i - q_j||^2 via the expansion, which avoids
        # materialising an [b, h, n, n, d_head] difference tensor.
        sq = (qk * qk).sum(-1)
        logits = -(sq.unsqueeze(-1) + sq.unsqueeze(-2)
                   - 2.0 * qk @ qk.transpose(-2, -1))
        logits = logits / (QK_TEMPERATURE * math.sqrt(dh))

        attn = torch.softmax(logits, dim=-1)

        # Normalised attention entropy, for diagnosis rather than
        # training. 1.0 means uniform over the window -- the encoder is
        # averaging and the policy gets a smoothed state. 0.0 means
        # collapsed to a single key; since the L2 distance is exactly
        # zero at i = j, collapse here means each step attends only to
        # itself and there is no temporal mixing at all. Both failures
        # look identical in |a| and the death rate, which is why this
        # is worth measuring directly.
        if self.record_entropy:
            with torch.no_grad():
                p = attn.clamp_min(1e-9)
                ent = -(p * p.log()).sum(-1).mean()
                self.last_entropy = (
                    float(ent / math.log(n)) if n > 1 else 1.0)

        out = (attn @ v).transpose(1, 2).reshape(b, n, self.d_model)
        return self.out_proj(out)


class LipschitzEncoderLayer(nn.Module):
    """Pre-norm block: x + attn(norm(x)), then x + ff(norm(x))."""

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=LAYERNORM_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LAYERNORM_EPS)
        self.attn = L2SelfAttention(d_model, nhead)
        self.linear1 = _sn(nn.Linear(d_model, dim_feedforward))
        self.linear2 = _sn(nn.Linear(dim_feedforward, d_model))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x

    @torch.no_grad()
    def clamp_gamma(self):
        for norm in (self.norm1, self.norm2):
            norm.weight.clamp_(-LAYERNORM_GAMMA_MAX, LAYERNORM_GAMMA_MAX)


class LipschitzEncoder(nn.Module):
    """
    Stack of Lipschitz encoder layers plus the closing norm.

    A real nn.Module assigned to `self.encoder`, NOT a bare list with
    the parent's encoder set to None. main.py builds its optimizer
    parameter groups from `model.encoder.parameters()` in three
    places, so anything that is not a module there fails with
    "NoneType has no attribute 'parameters'" before the first episode.
    Keeping the attribute name and the callable interface means the
    trunk/head learning-rate split works unchanged.
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.layers = nn.ModuleList([
            LipschitzEncoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=LAYERNORM_EPS)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    @torch.no_grad()
    def clamp_gamma(self):
        for layer in self.layers:
            layer.clamp_gamma()
        self.norm.weight.clamp_(-LAYERNORM_GAMMA_MAX, LAYERNORM_GAMMA_MAX)


class LipschitzCostTransformerActorCritic(CostTransformerActorCritic):
    """
    The cost arm with its softmax encoder replaced by a Lipschitz one.

    Subclassed rather than copied so the actor, the critic head, the
    log_std parameterisation, the layer sizes and every loss stay
    exactly as the reference arm -- the encoder is the single
    difference, which is what makes the comparison mean anything.
    """

    def __init__(self, *args, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, **kwargs):
        super().__init__(*args, d_model=d_model, nhead=nhead,
                         num_layers=num_layers,
                         dim_feedforward=dim_feedforward, **kwargs)

        # Replaces the parent's nn.TransformerEncoder in-place, under
        # the same attribute name, so its parameters neither train nor
        # travel in the state_dict and every caller keeps working.
        self.encoder = LipschitzEncoder(
            d_model, nhead, num_layers, dim_feedforward)

        # The input projection and the pooling head are on the path
        # from state to latent, so they need bounding too.
        self.input_projection[0] = _sn(self.input_projection[0])
        self.attention_pool = _sn(self.attention_pool)

    def encode(self, sequence):
        sequence = sequence.float()
        x = self.input_projection(sequence)
        x = x + self.position_embedding[:, :x.size(1)]
        x = self.encoder(x)
        scores = self.attention_pool(x)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)

    @torch.no_grad()
    def clamp_layernorm(self):
        """
        Call once per update. The gamma bound is part of the
        certificate, so it has to be enforced rather than assumed.
        """
        self.encoder.clamp_gamma()

    def set_entropy_recording(self, enabled):
        """Toggle entropy capture. See L2SelfAttention.record_entropy."""
        for layer in self.encoder.layers:
            layer.attn.record_entropy = bool(enabled)

    @torch.no_grad()
    def attention_entropy(self):
        """
        Mean normalised attention entropy across encoder layers, from
        the most recent forward pass.

        Read it off a forward on HELD-OUT states (diagnostics.py runs
        one before every update), not off a training minibatch, so it
        describes the encoder rather than whatever the last gradient
        step happened to touch.
        """
        vals = [layer.attn.last_entropy for layer in self.encoder.layers]
        vals = [v for v in vals if v == v]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    @torch.no_grad()
    def encoder_lipschitz(self):
        """
        Analytic upper bound on the state -> latent map.

        Composed as a product of per-component bounds:

            LayerNorm       ||gamma||_inf / sqrt(eps)
            residual block  1 + Lip(sublayer)
            L2 attention    per Kim et al. (2021) Thm 3.2, the bound
                            has the form
                            k(N, d_head) * ||W_qk|| * ||W_v|| * ||W_o||
                            with k growing sub-linearly in the
                            sequence length. K_ATTN below is that
                            factor and is the one number here taken
                            from the paper rather than measured --
                            CHECK IT against the published constant
                            before quoting the result.

        Loose by construction: every step is a worst case and they
        multiply. Report it as finite and computable, not as tight.
        """
        ln = LAYERNORM_GAMMA_MAX / math.sqrt(LAYERNORM_EPS)
        c = ENCODER_C

        # Placeholder for the sequence-length factor of Thm 3.2.
        k_attn = math.sqrt(self.encoder.layers[0].attn.d_head) / QK_TEMPERATURE

        total = c                                  # input projection
        for _ in self.encoder.layers:
            attn_block = 1.0 + ln * k_attn * ENCODER_QK_C * c * c
            ff_block = 1.0 + ln * c * c
            total *= attn_block * ff_block
        total *= ln                                # closing norm
        total *= c                                 # attention pool
        return float(total)


# Deficit at which the certificate is required to vanish. Matches
# SOC_TARGET in main.py; kept local so this module does not import it.
B_TGT = float(os.environ.get("LTAC_SOC_TARGET", "0.90"))

# kappa_1(d) = EPS_Q * d^2, by construction rather than by fit. The
# default is the quadratic envelope measured on the existing bounded
# arm (1.451 +- 0.150 over eight seeds), so the head is asked for no
# more than the current critic already achieves.
EPS_Q = float(os.environ.get("LTAC_EPS_Q", "1.0"))

# EMA horizon for the reference shift g_ref.
REF_MOMENTUM = float(os.environ.get("LTAC_REF_MOM", "0.99"))


class ICNNCostTransformerActorCritic(LipschitzCostTransformerActorCritic):
    """
    Bounded observer with a certificate that vanishes on the target set
    and carries a known quadratic lower bound.
    """

    def __init__(self, *args, eps_q=None, scalar_dim=8, **kwargs):
        super().__init__(*args, scalar_dim=scalar_dim, **kwargs)
        self.eps_q = float(EPS_Q if eps_q is None else eps_q)
        self.scalar_dim = int(scalar_dim)
        # Reference shift, tracked rather than fixed: what the network
        # assigns to states already in Z* moves during training, and a
        # constant would make V vanish at a point the critic has left.
        self.register_buffer("g_ref", torch.zeros(()))
        self.register_buffer("g_ref_init", torch.zeros((), dtype=torch.bool))

    # -- the deficit the quadratic term is taken on -------------------
    #
    # State of charge is the fifth scalar of the observation's scalar
    # block, and the LAST timestep of the window is the current state.
    # Reading it here rather than threading it through every call site
    # keeps the head's signature identical to the parent's, which is
    # what lets main.py dispatch on the variant alone.
    def _deficit(self, sequence):
        # obs() packs the patch first, then the scalar block, whose
        # fifth entry is battery/100. Index from the END so the patch
        # size never enters.
        b = sequence[:, -1, -self.scalar_dim + 4]
        return torch.clamp(B_TGT - b, min=0.0)

    @torch.no_grad()
    def update_reference(self, latent, deficit):
        """EMA of g(h) over states already inside Z*."""
        inside = deficit <= 0
        if not bool(inside.any()):
            return
        g_in = self.critic_body(latent[inside]).squeeze(-1).mean()
        if not bool(self.g_ref_init):
            self.g_ref.fill_(float(g_in))
            self.g_ref_init.fill_(True)
        else:
            self.g_ref.mul_(REF_MOMENTUM).add_((1.0 - REF_MOMENTUM) * g_in)

    def critic_with_deficit(self, latent, deficit):
        """
        V^pi = -V_cost, with

            V_cost(z) = beta*softplus((g(h) - g_ref)/beta) + eps*d^2.

        SOFTPLUS_AT_ZERO. softplus(0) = log 2 != 0, so the shifted
        softplus alone does not vanish on Z*; subtracting
        beta*softplus(0) makes the first term exactly zero when
        g(h) = g_ref, at the cost of admitting small negative values
        when g(h) < g_ref. The quadratic term cannot repair that on
        Z* itself, where d = 0, so the first term is clamped at zero.
        Clamping is safe here precisely because the quadratic term
        supplies the lower bound: V >= eps d^2 regardless of what the
        softplus branch does.
        """
        raw = self.critic_body(latent).squeeze(-1)
        shifted = raw - self.g_ref
        soft = self._beta * F.softplus(shifted / self._beta, beta=1.0)
        soft = soft - self._beta * math.log(2.0)
        v_cost = torch.clamp(soft, min=0.0) + self.eps_q * deficit ** 2
        return -v_cost

    # -- parent interface, with the deficit read from the window ------
    def critic(self, latent):
        raise RuntimeError(
            "ICNNCostTransformerActorCritic needs the state-of-charge "
            "deficit, which is not recoverable from the latent alone. "
            "Call value_only(sequence) or forward(sequence); a bare "
            "critic(latent) would silently drop the quadratic term and "
            "with it the guaranteed lower bound.")

    def value_only(self, sequence):
        latent = self.encode(sequence)
        return self.critic_with_deficit(latent, self._deficit(sequence))

    def forward(self, sequence):
        # Same tuple as the parent, but the value is computed with the
        # deficit. The parent's forward calls critic(latent), which
        # this class refuses, so overriding is required rather than
        # optional -- without it every training step would raise.
        latent = self.encode(sequence)
        deficit = self._deficit(sequence)
        if self.training:
            self.update_reference(latent, deficit)
        raw_mean = self.actor(latent)
        raw_log_std = self.log_std_param.unsqueeze(0).expand(
            raw_mean.shape[0], -1)
        return (raw_mean, raw_log_std,
                self.critic_with_deficit(latent, deficit), latent)
