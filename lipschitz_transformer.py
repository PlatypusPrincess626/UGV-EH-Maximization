"""
Cost arm with a LIPSCHITZ-CERTIFIABLE encoder.

WHAT PROBLEM THIS SOLVES

The checkpoint at episode 400 says the spectral norm is registered on
critic_body.0/2/4/6 and nowhere else. So `L <= 1` bounds the map from
LATENT to V_cost -- and the latent is produced by an entirely
unconstrained transformer whose measured spectral norms are:

    input_projection            4.678
    layer0 self_attn.in_proj    4.928     layer1  5.334
    layer0 self_attn.out_proj   2.891     layer1  2.772
    layer0 linear1              3.133     layer1  2.836
    layer0 linear2              2.621     layer1  2.376
    attention_pool              1.780

Every one of these grew 1.16-2.76x above its Xavier initialisation, so
the encoder is where the learning went -- and it carries no bound at
all. Any region claim stated in STATE space rests on a Lipschitz
constant this architecture does not have.

Worse, it is not a matter of choosing the constraint well. Softmax
dot-product attention is not Lipschitz on an unbounded domain: its
sensitivity depends on the VALUES of the queries and keys, not only on
the norms of the projections, so spectral-normalising W_q, W_k, W_v
does not bound it. Note which layers grew most -- both self_attn
in_proj, at 2.55x and 2.76x. The unbounded component is also the one
doing the most work.

WHAT THIS ARM CHANGES

L2 self-attention, following Kim, Papamakarios & Mnih (2021), "The
Lipschitz Constant of Self-Attention". The dot product is replaced by
a negative squared distance and the query and key projections are
TIED:

    A_ij = softmax_j( -||W x_i - W x_j||^2 / sqrt(d_head) )

With W_q = W_k this form is provably Lipschitz on the whole space --
no bounded-domain assumption needed. The observation range being
[0,1] or [-1,1] is therefore a nice property but not something this
construction depends on, which is worth knowing: the guarantee does
not quietly break if a future observation channel leaves the box.

Everything else that could be unbounded is also closed:

    projections     spectral-normalised to sigma <= ENCODER_C
    feedforward     spectral-normalised to sigma <= ENCODER_C
    residuals       Lip(x + f(x)) <= 1 + Lip(f)
    LayerNorm       Lipschitz with constant ||gamma||_inf / sqrt(eps).
                    PyTorch's default eps of 1e-5 gives a factor of
                    ~316 per norm, which is finite but ruins the
                    product, so eps is raised (see LAYERNORM_EPS) and
                    gamma is clamped.
    attention_pool  spectral-normalised

WHAT IT COSTS, STATED HONESTLY

The bound is a product over layers, so it is large even when every
factor is modest -- `encoder_lipschitz()` reports it and it should be
read as "finite and computable", not "tight". A loose finite bound is
still categorically different from no bound: it makes the region claim
a theorem with a bad constant rather than an assertion.

COMPARABILITY WITH THE OTHER ARMS

This arm does NOT share the bit-identical trunk the others do. The
encoder is a different module with a different parameter count and it
consumes the RNG differently, so a given seed no longer produces the
same initial weights as `cost` or `cost_plain`. Task and certification
comparisons remain valid; the seed-paired "only one thing differs"
property does not. Say so when reporting it.
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
QK_TEMPERATURE = float(os.environ.get("LTAC_QK_TEMP", "16.0"))

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
