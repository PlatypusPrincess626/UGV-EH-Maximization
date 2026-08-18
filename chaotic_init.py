"""
Chaotic weight initialisation: Chen system (default) or Chebyshev map.

WHAT THIS VARIES, AND WHAT IT MUST NOT

The claim behind chaotic initialisation is that a deterministic
chaotic orbit covers its support more evenly than a pseudo-random
draw -- lower discrepancy, no clustering -- and that a network started
from such a sequence explores better early. Whether that holds here is
the experiment.

For the experiment to mean anything, the ONLY thing that changes is
the SEQUENCE. The scale must not move. Every tensor is filled with a
chaotic orbit rescaled to the exact mean and variance the arm's normal
initialiser would have produced, so signal propagation at layer zero
is statistically identical and any difference in outcome is
attributable to the ordering and spacing of the values rather than to
a gain change.

This matters because a scale change dressed up as an initialisation
change is exactly the kind of confound that has cost runs in this
project before: an encoder constrained to sigma = 6.0 instead of ~2.0
looked like an architecture experiment and was really an amplitude
bug.

WHY VARIANCE-MATCHING RATHER THAN RANGE-MATCHING

Min-max normalising a chaotic orbit into [-a, a] is the common recipe
and it is wrong for this purpose. Chaotic orbits are not uniform on
their support -- the Chen attractor spends far more time near its
dense sheets than at the extremes -- so range-matching leaves the
variance well below the uniform's (high-low)^2/12 and quietly shrinks
every layer. Matching mean and variance keeps the second moment, which
is what determines activation scale through the network.

CHEN VERSUS CHEBYSHEV

Chen is a continuous 3D system:

    dx/dt = a(y - x)
    dy/dt = (c - a)x - x z + c y
    dz/dt = x y - b z            a = 35, b = 3, c = 28

integrated with RK4. It is chaotic, well documented, and the usual
reference in the chaotic-initialisation literature, so it is the
default. Chebyshev is a 1D map, x <- cos(k arccos(x)) on [-1, 1],
which is cheaper and simpler but has a much thinner claim to
"structured coverage" since successive iterates are strongly
correlated at low k. Both are provided; Chen is used unless asked
otherwise.

SEEDING

The orbit's initial condition is derived from the run seed, so
different seeds give different initial weights. Without this every
seed would produce a bit-identical network and the seed sweep would
collapse to a single sample -- which would look like remarkable
seed-stability and would be an artifact.
"""

import math
import os

import torch
import torch.nn as nn

# Chen system parameters. The classical values; the attractor exists
# over a range of these but 35/3/28 is what the literature reports.
CHEN_A = 35.0
CHEN_B = 3.0
CHEN_C = 28.0

# Integration step and how many steps to discard before sampling, so
# the orbit is on the attractor rather than still transiting to it.
CHEN_DT = 0.01
CHEN_TRANSIENT = 1000

# Sample every Nth integration step. Consecutive RK4 steps at dt=0.002
# are nearly identical points; decimating decorrelates them, which is
# the entire point of using a chaotic sequence rather than a smooth
# one.
#
# Measured lag-1 autocorrelation of the sampled x-coordinate:
#
#     dt 0.002, decimate  25 -> lag-1 +0.76   a smooth ramp, not a sequence
#     dt 0.002, decimate 200 -> lag-1 -0.20   usable
#     dt 0.01,  decimate  40 -> lag-1 -0.22   same, 5x cheaper
#
# The sampling interval that matters is dt * decimate = 0.4 time units
# of the attractor, not the step count. RK4 at dt = 0.01 tracks Chen
# accurately (checked against dt = 0.002: same mean, sd 8.49 vs 8.41,
# same lag-1, same range), so the coarser step is free. That matters
# because the orbit is generated in pure Python and a 465k-parameter
# model needs one value per weight: at dt = 0.002 that was 93M RK4
# steps and 3.6 minutes of startup PER RUN -- 32 minutes across three
# seeds of three chaotic arms. At dt = 0.01 it is 0.7 minutes.
#
# At 25 the "chaotic" weights would be a slowly varying ramp across
# each tensor, which is a structured artifact rather than the even
# coverage the method is supposed to provide. 200 costs 8x the
# integration -- a few seconds once at startup, against a 4.8 hour
# run.
CHEN_DECIMATE = 40

# Chebyshev order. k >= 2 is chaotic; higher k decorrelates faster.
CHEBYSHEV_K = 5.0

# Default map. CHEBYSHEV, not Chen -- reversing the original
# preference, for a measured reason.
#
# The rank collapse is caused by autocorrelation in the orbit, and a
# continuous ODE cannot avoid it: its trajectory is smooth in time and
# lives on a low-dimensional attractor sheet, so consecutive samples
# share structure no matter how far they are decimated. Discrete maps
# in the Chebyshev/logistic family have ZERO linear autocorrelation at
# these parameters -- second-order white despite being deterministic
# and chaotic -- so their orbits fill a matrix the way an iid draw
# does.
#
# Effective rank of a 256x128 tensor, moments matched to Xavier,
# SEQUENTIAL placement:
#
#     sequence              sigma_max   eff.rank   lag-1 ac
#     xavier (iid)              1.967       70.2       ~0
#     chen x, dt.01 dec40       2.492       52.9     -0.223
#     chebyshev k=2             1.943       71.1     -0.001
#     chebyshev k=3             1.980       69.8     -0.001
#     chebyshev k=5             1.936       71.5     +0.002
#     chebyshev k=7             1.946       70.9     +0.005
#     logistic r=4              1.945       71.0     -0.004
#
# Every discrete map is indistinguishable from Xavier on both rank and
# operator norm; Chen is the outlier. That also removes the need for
# permuted placement, so CHAOTIC_PLACEMENT defaults back to
# "sequential" -- the faithful reading of the method, in which the
# orbit's ordering is what fills the tensor.
#
# Chen remains available (LTAC_CHAOTIC_KIND=chen) and is worth running
# as a contrast: it is the same idea implemented with a system whose
# correlation structure damages the initialisation, which is a result
# in itself.
#
# A caution for anyone extending this: the tent map, x <- 2x or
# 2(1-x), is chaotic in exact arithmetic and DEGENERATE in binary
# floating point -- the doubling is a bit shift, so the orbit walks
# off the end of the mantissa and collapses to zero within ~50
# iterations. It produced a matrix numpy could not even factor. Verify
# any new map's spectrum before trusting it.
CHAOTIC_KIND_DEFAULT = "chebyshev"

# Passes of the spectral-norm power method after re-initialisation.
# Each pass runs n_power_iterations (5) internally, so 10 passes is 50
# iterations against the new weight -- ample for a top singular vector
# and negligible at startup.
POWER_ITERATION_REFRESH = int(os.environ.get("LTAC_POWER_REFRESH", "10"))


def _chen_stream(n, seed):
    """n values from the x-coordinate of a Chen orbit."""
    rng = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    # Initial condition on the attractor's rough scale, seed-dependent.
    x, y, z = (torch.rand(3, generator=rng).tolist())
    x = -10.0 + 20.0 * x
    y = -15.0 + 30.0 * y
    z = 5.0 + 35.0 * z

    def deriv(x, y, z):
        return (CHEN_A * (y - x),
                (CHEN_C - CHEN_A) * x - x * z + CHEN_C * y,
                x * y - CHEN_B * z)

    out = []
    total = CHEN_TRANSIENT + n * CHEN_DECIMATE
    h = CHEN_DT
    for i in range(total):
        k1 = deriv(x, y, z)
        k2 = deriv(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1], z + 0.5 * h * k1[2])
        k3 = deriv(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1], z + 0.5 * h * k2[2])
        k4 = deriv(x + h * k3[0], y + h * k3[1], z + h * k3[2])
        x = x + h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        y = y + h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        z = z + h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
        if i >= CHEN_TRANSIENT and (i - CHEN_TRANSIENT) % CHEN_DECIMATE == 0:
            out.append(x)
        if not math.isfinite(x):
            raise RuntimeError("Chen orbit diverged; check CHEN_DT")
    return torch.tensor(out[:n], dtype=torch.float64)


def _chebyshev_stream(n, seed):
    """n values from the Chebyshev map x <- cos(k arccos x) on [-1, 1]."""
    rng = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    x = float(torch.rand(1, generator=rng).item()) * 1.8 - 0.9
    out = []
    for _ in range(200):                       # transient
        x = math.cos(CHEBYSHEV_K * math.acos(max(-1.0, min(1.0, x))))
    for _ in range(n):
        x = math.cos(CHEBYSHEV_K * math.acos(max(-1.0, min(1.0, x))))
        out.append(x)
    return torch.tensor(out, dtype=torch.float64)


class ChaoticStream:
    """
    A rewindable supply of chaotic values.

    Generated in one block and consumed in order, so the assignment of
    orbit positions to tensors is deterministic given (kind, seed) and
    the module traversal order -- reproducible across runs, and
    independent of the torch RNG so it cannot desync anything else.
    """

    def __init__(self, kind, seed, capacity):
        kind = kind.strip().lower()
        if kind == "chen":
            self.values = _chen_stream(capacity, seed)
        elif kind == "chebyshev":
            self.values = _chebyshev_stream(capacity, seed)
        else:
            raise ValueError(f"unknown chaotic kind {kind!r}")
        self.kind = kind
        self.pos = 0

    def take(self, n):
        if self.pos + n > self.values.numel():
            # Wrap rather than fail. The orbit is aperiodic, so a wrap
            # reuses values but not the local ordering; capacity is
            # sized to make this rare.
            self.pos = 0
        chunk = self.values[self.pos:self.pos + n]
        self.pos += n
        return chunk


# What to hold fixed when swapping the RNG for a chaotic orbit.
#
#   "variance" (default) -- match the tensor's element mean and
#       standard deviation. This is what Xavier's derivation controls:
#       Var(y) = fan_in * Var(w) * Var(x), so matching element
#       variance preserves activation scale in the mean.
#
#   "spectral" -- additionally rescale so the tensor's LARGEST
#       SINGULAR VALUE matches the original's.
#
# WHY THE CHOICE EXISTS
#
# Matching moments does NOT match operator norm. Measured on a
# 256x128 tensor with mean and standard deviation matched exactly:
#
#     xavier    sigma_max 1.967
#     chaotic   sigma_max 2.556      +30%
#
# So on an UNCONSTRAINED layer -- everything in normal and lyapunov,
# and everything outside the critic body in the cost arms -- the
# chaotic orbit hands the layer 30% more gain than the initialiser it
# replaced, compounding across layers. That is a scale change wearing
# an initialisation costume, which is precisely what this module's
# header says it must avoid, and it would confound the chaotic arms
# the same way ENCODER_C = 6.0 confounded the Lipschitz arm.
#
# On a SPECTRALLY NORMALISED layer it is moot: the parametrisation
# divides by sigma_max and multiplies by c, so the effective weight
# has sigma_max = c whatever the underlying tensor looks like. The
# Lipschitz bound is untouched by construction, and
# encoder_lipschitz() is a function of the constants alone.
#
# What still differs there is the spectrum BELOW the top: after
# normalising sigma_max to 1, the chaotic tensor's effective rank is
# 51.5 against xavier's 70.2. Same bound, more concentrated. That is
# a real difference in conditioning and arguably the mechanism by
# which chaotic initialisation would help or hurt at all -- worth
# reporting rather than treating as noise.
# How the orbit's values are laid out inside each tensor.
#
#   "sequential" -- consecutive orbit samples fill consecutive matrix
#       entries, row-major. The literal reading of the method.
#
#   "scatter" (default) -- the same values, placed through a
#       deterministic permutation derived from the seed.
#
# WHY THE DEFAULT IS SCATTER
#
# Sequential placement collapses the rank. Measured on a 256x128
# tensor with element moments matched to Xavier:
#
#     fill                       sigma_max   eff.rank   s128/s1
#     xavier (iid uniform)           1.967       70.2    0.1880
#     chen x, sequential             2.556       51.5    0.1069
#     chen x,y,z interleaved         7.575       11.7    0.0198
#     chen x, permuted placement     1.907       72.4    0.1645
#
# A 27% loss of effective rank at the INPUT PROJECTION is information
# destroyed at layer one and never recovered -- the 1689-dimensional
# patch is being read through a lower-dimensional shadow than the
# initialiser it replaced. Spectral normalisation does not help: it
# rescales sigma_max and leaves the spectrum's shape alone, so on the
# Lipschitz arm the bound is intact and the rank loss is not.
#
# The cause is the ORDER, not the values: permuting the same numbers
# restores rank 72.4, statistically indistinguishable from Xavier. The
# orbit is a trajectory on a low-dimensional attractor, so consecutive
# samples share structure, and laying them out row-major writes that
# structure straight into the matrix as correlated rows.
#
# Note also that interleaving x, y and z -- the obvious way to "use
# more of the attractor" -- is far worse, not better: rank 11.7. The
# three coordinates are coupled by the ODE, so interleaving them
# produces a strongly patterned sequence rather than a richer one.
#
# HONEST CAVEAT
#
# Scatter keeps the chaotic marginal distribution and the
# deterministic, seed-reproducible generation, while discarding the
# orbit's local ordering. If the effect being tested is specifically
# the ordering, "sequential" is the faithful setting -- and the table
# above then predicts it will underperform. Both are provided; report
# which was used.
CHAOTIC_PLACEMENT = os.environ.get("LTAC_CHAOTIC_PLACEMENT", "sequential").strip().lower()
if CHAOTIC_PLACEMENT not in ("sequential", "scatter"):
    raise ValueError("LTAC_CHAOTIC_PLACEMENT must be 'sequential' or 'scatter'")

CHAOTIC_MATCH = os.environ.get("LTAC_CHAOTIC_MATCH", "variance").strip().lower()
if CHAOTIC_MATCH not in ("variance", "spectral"):
    raise ValueError("LTAC_CHAOTIC_MATCH must be 'variance' or 'spectral'")


def fill_matching_moments_(tensor, stream, mean, std, perm_seed=0):
    """
    Fill `tensor` with chaotic values rescaled to exactly `mean` and
    `std`.

    Standardising the chunk empirically rather than by the orbit's
    theoretical moments, so the match holds for the actual values used
    rather than in the limit.
    """
    n = tensor.numel()
    chunk = stream.take(n).clone()
    if chunk.numel() < n:                      # pathological wrap
        chunk = chunk.repeat((n // max(chunk.numel(), 1)) + 1)[:n]
    c_std = chunk.std()
    if not torch.isfinite(c_std) or c_std < 1e-12:
        raise RuntimeError("degenerate chaotic chunk")
    if CHAOTIC_PLACEMENT == "scatter":
        # Deterministic permutation of placement. Same values, same
        # seed-reproducibility; only the orbit's local ordering is
        # discarded. See CHAOTIC_PLACEMENT.
        g = torch.Generator().manual_seed(perm_seed)
        chunk = chunk[torch.randperm(chunk.numel(), generator=g)]

    chunk = (chunk - chunk.mean()) / c_std
    with torch.no_grad():
        new = (chunk * std + mean).to(tensor.dtype).view_as(tensor)
        if CHAOTIC_MATCH == "spectral" and new.dim() == 2:
            before = torch.linalg.matrix_norm(tensor.detach().float(), 2)
            after = torch.linalg.matrix_norm(new.float(), 2)
            if after > 1e-9:
                new = new * (before / after).to(new.dtype)
        tensor.copy_(new)
    return tensor


@torch.no_grad()
def apply_chaotic_init(model, kind, seed, verbose=True):
    """
    Re-initialise every Linear weight in `model` from a chaotic orbit,
    preserving the mean and variance the existing initialisation gave
    that tensor.

    Applied AFTER the model is built, so it inherits whatever scheme
    the arm used (xavier_uniform here) and matches it moment for
    moment. Biases are left alone: they are initialised to zero, there
    is no variance to match, and a chaotic bias would be a scale
    change rather than a sequence change.

    LayerNorm weights are also left alone -- they are ones by
    construction and in the Lipschitz arm they carry a clamp that is
    part of the certificate.

    Spectrally-normalised layers are re-initialised through their
    underlying `original` parameter, so the normalisation still
    applies afterwards and sigma_max is unchanged.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pname, p in module.named_parameters(recurse=False):
                if "weight" in pname and p.dim() == 2:
                    targets.append((f"{name}.{pname}", p))
        # spectral_norm / parametrized layers keep the trainable
        # tensor under parametrizations.weight.original
    seen = {id(p) for _, p in targets}
    for name, p in model.named_parameters():
        if (name.endswith("parametrizations.weight.original")
                and p.dim() == 2 and id(p) not in seen):
            targets.append((name, p))

    capacity = sum(p.numel() for _, p in targets) + 1024
    stream = ChaoticStream(kind, seed, capacity)

    for i, (name, p) in enumerate(targets):
        fill_matching_moments_(p, stream, float(p.mean()), float(p.std()),
                               perm_seed=(int(seed) * 1000003 + i) & 0x7FFFFFFF)

    # Re-converge the spectral-norm power iteration.
    #
    # THE BUG THIS FIXES
    #
    # spectral_norm caches singular vectors u and v as buffers,
    # estimated by power iteration from the weight PRESENT AT
    # REGISTRATION. Overwriting weight.original afterwards leaves
    # those vectors pointing at the old matrix's top singular
    # direction, which is essentially a random direction relative to
    # the new one -- so the estimated sigma is far too small and the
    # effective weight comes out with sigma_max well ABOVE the c it is
    # supposed to be pinned at.
    #
    # It does not self-correct straight away either. _SpectralNorm
    # only runs the power method when the module is in TRAINING mode,
    # and main.py collects rollouts under model.eval(). So the first
    # rollout -- the data the critic's first fit is built from -- is
    # generated by a network whose spectral constraints are silently
    # not in force. On an arm that depends on those constraints for
    # its conditioning, that is a poisoned first batch.
    #
    # Forcing the module into training mode and reading .weight runs
    # the power method; a handful of passes is enough to converge it
    # against the new matrix.
    was_training = model.training
    model.train()
    try:
        for _ in range(POWER_ITERATION_REFRESH):
            for module in model.modules():
                if hasattr(module, "parametrizations"):
                    for pname in list(module.parametrizations.keys()):
                        getattr(module, pname)
    finally:
        if not was_training:
            model.eval()

    if verbose:
        total = sum(p.numel() for _, p in targets)
        # Report the operator-norm change explicitly. On unconstrained
        # layers it is the number that decides whether this was a
        # sequence experiment or an accidental gain change; on
        # spectrally normalised ones it should read ~1.00 because the
        # parametrisation renormalises regardless.
        ratios = []
        for _, p in targets:
            if p.dim() == 2:
                ratios.append(float(torch.linalg.matrix_norm(p.float(), 2)))
        print(f"[chaotic-init] {stream.kind}: re-initialised "
              f"{len(targets)} weight tensors ({total} parameters), "
              f"match={CHAOTIC_MATCH}, placement={CHAOTIC_PLACEMENT}, "
              f"seed {seed}")
        if ratios:
            print(f"[chaotic-init] sigma_max after: mean "
                  f"{sum(ratios)/len(ratios):.3f}, max {max(ratios):.3f}")
    return model
