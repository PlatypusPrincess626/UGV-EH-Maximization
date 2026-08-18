"""
Arm-agnostic critic diagnostics.

Everything here runs under `torch.no_grad()`, before the first
gradient step of an update, on the batch the critic has NOT yet been
trained on. Nothing touches the loss, the RNG consumed by training, or
the optimizer state, so a run with these enabled is bit-identical in
its policy to a run without them.

WHY THESE THREE

The paired sweep could not answer its own question. Three gaps:

1. `value_loss` is MSE(values, returns) / running Var(returns), taken
   on the minibatch the critic is currently fitting. It is a training
   loss, not a fit measure: it is computed on seen data and its
   denominator is a lagged running estimate rather than the variance
   of the batch it is scoring. `critic_ev` below is the standard
   held-out explained variance, computed once per update on fresh
   rollout states, and is comparable across arms.

2. Every `v_*` column was gated on the cost variants, so the reward
   arms logged NaN for exactly the quantities the cost arm is claimed
   to win on. There was nothing to compare against. The agreement
   probe here runs in all arms: it correlates each arm's critic
   against the analytic Lyapunov function V_true(soc), which is
   defined from measured state of charge and is therefore independent
   of which reward the arm was trained on.

3. `v_critic_min` samples only visited states, so the structural
   guarantee V_cost >= 0 was being tested exactly where the data
   already enforces it. `soc_sweep_probe` moves off that manifold.

THE SIGN CONVENTION

For a cost arm the critic's output IS the negated Lyapunov candidate:
V_cost = -value. For a reward arm the value is a reward-to-go and
higher means healthier, so -value is again the quantity that should
grow with battery deficit. Defining

    V_lyap = -value

in every arm makes the agreement correlation directly comparable.
This is NOT a claim that a reward arm's critic is a Lyapunov function
-- it is not, and no affine map makes it one, because its descent
condition is a statement about reward accumulation rather than
stability. The correlation measures something weaker and still
useful: whether the critic encodes the safety-relevant coordinate at
all. Report it as agreement, never as certification.
"""

import numpy as np
import torch

# Dedicated RNG for the subsampling below.
#
# NOT the global torch RNG. `ablation_transformer.py` is subclassed
# rather than copied specifically so that a given seed consumes the
# RNG in the same order in every arm and produces a bit-identical
# encoder. A `torch.randperm` drawn from the global stream inside a
# diagnostic would desync that -- an arm with diagnostics enabled
# would sample different exploration noise than one without, and the
# paired sweep's whole design would quietly stop holding. Seeded from
# a constant so the probe subsample is itself reproducible.
_DIAG_GEN = torch.Generator().manual_seed(0x17AC)


def _subsample(n, k, device):
    """Indices for a k-of-n subsample, drawn off the diagnostic RNG."""
    return torch.randperm(n, generator=_DIAG_GEN)[:k].to(device)


def analytic_v_true(soc, soc_target):
    """
    The analytic Lyapunov function, from measured state of charge.

    Mirrors the definition already used in main.py's certification
    block so the two cannot drift apart. `soc` is the normalized
    observation channel, in [0, 1].
    """
    return (torch.relu(soc_target - soc) / soc_target) ** 2


def _pearson(a, b):
    if a.numel() < 2 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def _spearman(a, b):
    """
    Rank correlation, computed by ranking then Pearson.

    Worth logging alongside Pearson because V_true is a squared hinge:
    it is exactly zero for every state at or above SOC_TARGET, which
    puts a large point mass at zero and drags a linear correlation
    toward whatever the critic does in that flat region. Spearman is
    insensitive to that monotone distortion, so a large gap between
    the two says the critic has the right ORDER but the wrong shape.
    """
    if a.numel() < 2:
        return float("nan")
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return _pearson(ra, rb)


@torch.no_grad()
def critic_explained_variance(values, returns):
    """
    Held-out explained variance: 1 - Var(returns - values) / Var(returns).

    1.0 is a perfect critic, 0.0 is exactly as good as predicting the
    mean return, and negative is worse than the mean. Unlike
    `value_loss`, the denominator is the variance of the SAME batch
    being scored, so the number is self-contained and does not depend
    on the state of a running tracker.

    Scale-free, so it is comparable between a cost MDP whose returns
    are O(1) and a reward MDP whose returns are O(100).
    """
    blank = {"critic_ev": float("nan"), "critic_rmse_norm": float("nan"),
             "returns_std": float("nan"), "returns_mean": float("nan")}
    if returns.numel() < 2:
        return blank
    var_y = returns.var(unbiased=False)
    if var_y < 1e-12:
        return blank
    resid = returns - values
    ev = 1.0 - (resid.var(unbiased=False) / var_y)
    # RMSE in units of the return standard deviation. Equal to
    # sqrt(1 - EV) when the residual is unbiased; the two diverge when
    # the critic has a constant offset, which is worth being able to
    # see separately.
    rmse_norm = resid.pow(2).mean().sqrt() / var_y.sqrt()

    # Log the DENOMINATOR alongside the ratio.
    #
    # EV is 1 - Var(resid)/Var(returns), so it says nothing on its own
    # when Var(returns) is small: an unchanged critic reads 0.99 at
    # std(returns) = 3, 0.30 at 0.3, and -5.0 at 0.1. Batch-to-batch
    # resampling at std(returns) = 0.3 alone moves EV over a 0.27-wide
    # range. So an EV swinging between -0.5 and +0.5 update to update
    # is the signature of a COLLAPSED RETURN SPREAD, not necessarily
    # of a critic that cannot learn -- and the two demand opposite
    # responses.
    #
    # Returns collapse when every episode ends the same way: uniform
    # early deaths give near-identical discounted returns, the
    # denominator goes to nothing, and both EV and value_loss (which
    # divides by the running return variance) become unreadable.
    #
    # returns_std is the number that separates the cases. Compare it
    # against a healthy run of the same arm before concluding anything
    # from EV.
    return {"critic_ev": float(ev.item()),
            "critic_rmse_norm": float(rmse_norm.item()),
            "returns_std": float(var_y.sqrt().item()),
            "returns_mean": float(returns.mean().item())}


@torch.no_grad()
def critic_agreement(values, soc, soc_target):
    """
    Agreement between the arm's critic and the analytic Lyapunov
    function, computed identically in every arm.

    Returns Pearson and Spearman correlation of V_lyap = -value
    against V_true(soc), plus the fraction of the batch sitting in the
    flat region (soc >= soc_target) where V_true carries no gradient
    -- without which a low correlation cannot be distinguished from a
    batch that simply had nothing to correlate.
    """
    v_lyap = -values
    v_true = analytic_v_true(soc, soc_target)
    return {
        "xv_true_corr": _pearson(v_lyap, v_true),
        "xv_true_rank": _spearman(v_lyap, v_true),
        "xv_true_flat_frac": float((soc >= soc_target).float().mean().item()),
    }


@torch.no_grad()
def soc_sweep_probe(model, states, soc_index, soc_target,
                    soc_grid=(0.05, 0.15, 0.25, 0.35, 0.45,
                              0.55, 0.65, 0.75, 0.85, 0.95),
                    max_states=256):
    """
    Off-distribution probe of the critic's shape in the one coordinate
    the analytic Lyapunov function depends on.

    Takes a subsample of real observation windows and overwrites the
    SOC channel -- at EVERY timestep of the window, so the sequence
    stays internally consistent rather than presenting a battery that
    teleports on the last frame -- with each value on `soc_grid`. The
    rest of the observation (the obfuscation patch, position, sun
    geometry) is left exactly as recorded.

    This is a legitimate probe rather than an adversarial one because
    V_true is a function of SOC alone: sweeping SOC while holding
    everything else fixed traces the curve the critic is claimed to
    have learned. Most of the resulting states are off-distribution --
    a 5% battery at midday in full sun is not a state the policy
    visits -- and that is the point. The structural guarantees are
    claimed everywhere, not only where the rollout went.

    Metrics
    -------
    ood_v_min, ood_v_mean
        min and mean of V_lyap = -value over the whole probe grid.
        For a cost arm with the Softplus head, ood_v_min < 0 is
        impossible by construction; for `cost_plain` it is the
        measurement that `v_critic_min` could never make, because
        `v_critic_min` only ever saw visited states.

    ood_nonneg_frac
        fraction of probe points with V_lyap >= 0.

    ood_monotone_frac
        fraction of adjacent grid pairs (lower SOC, higher SOC) where
        V_lyap decreases as SOC rises, per probe state. A Lyapunov
        candidate in this task must be non-increasing in battery: this
        is the shape claim, tested independently of any correlation.

    ood_slope_corr
        Spearman correlation between V_lyap and V_true across the full
        probe set. The agreement number, recomputed off-distribution.

    ood_range
        mean per-state spread of V_lyap across the grid. A critic that
        has collapsed to a constant in SOC scores a perfect
        `ood_monotone_frac` of nothing; this catches that.
    """
    n = states.shape[0]
    if n == 0:
        return {}
    k = min(max_states, n)
    base = states[_subsample(n, k, states.device)]

    grid = torch.tensor(soc_grid, dtype=base.dtype, device=base.device)
    vals = []
    for g in grid:
        probe = base.clone()
        probe[:, :, soc_index] = g
        vals.append(model.value_only(probe))
    # [n_grid, k]
    v_lyap = -torch.stack(vals, dim=0)

    v_true = analytic_v_true(grid, soc_target).unsqueeze(1).expand_as(v_lyap)

    diffs = v_lyap[1:] - v_lyap[:-1]
    return {
        "ood_v_min": float(v_lyap.min().item()),
        "ood_v_mean": float(v_lyap.mean().item()),
        "ood_nonneg_frac": float((v_lyap >= 0).float().mean().item()),
        "ood_monotone_frac": float((diffs <= 0).float().mean().item()),
        "ood_slope_corr": _spearman(v_lyap.reshape(-1), v_true.reshape(-1)),
        "ood_range": float((v_lyap.max(dim=0).values
                            - v_lyap.min(dim=0).values).mean().item()),
    }


DIAGNOSTIC_FIELDS = [
    "attn_entropy",
    "critic_ev", "critic_rmse_norm", "returns_std", "returns_mean",
    "xv_true_corr", "xv_true_rank", "xv_true_flat_frac",
    "ood_v_min", "ood_v_mean", "ood_nonneg_frac",
    "ood_monotone_frac", "ood_slope_corr", "ood_range",
]


@torch.no_grad()
def pre_update_diagnostics(model, states, returns, soc_all,
                           soc_target, soc_index,
                           probe_states=1024, probe_grid_states=256):
    """
    One call, run before the first gradient step of an update.

    `states` and `returns` come straight out of compute_batch, so the
    critic is being scored on a rollout it has never been fit to --
    this is genuine held-out evaluation, obtained without withholding
    any data from training.

    Subsampled to `probe_states` for the forward pass because a full
    720 x UPDATE_EVERY_EPISODES batch is larger than needed to
    estimate a correlation and this runs every update.
    """
    was_training = model.training
    model.eval()
    try:
        n = states.shape[0]
        if n == 0:
            return {f: float("nan") for f in DIAGNOSTIC_FIELDS}
        k = min(probe_states, n)
        idx = _subsample(n, k, states.device)

        # Entropy capture is off during rollout and training; switch it
        # on for this one forward only.
        records_entropy = hasattr(model, "set_entropy_recording")
        if records_entropy:
            model.set_entropy_recording(True)
        try:
            values = model.value_only(states[idx])
        finally:
            if records_entropy:
                model.set_entropy_recording(False)

        # Attention entropy, if the arm's encoder exposes it. Recorded
        # here rather than inside training so it is measured on the
        # same held-out forward as everything else -- and so the column
        # exists (as NaN) in every arm, keeping the CSV schema uniform.
        out = {}
        if hasattr(model, "attention_entropy"):
            out["attn_entropy"] = model.attention_entropy()
        out.update(critic_explained_variance(values, returns[idx]))
        out.update(critic_agreement(values, soc_all[idx], soc_target))
        out.update(soc_sweep_probe(model, states, soc_index, soc_target,
                                   max_states=probe_grid_states))
        for f in DIAGNOSTIC_FIELDS:
            out.setdefault(f, float("nan"))
        return out
    finally:
        if was_training:
            model.train()
