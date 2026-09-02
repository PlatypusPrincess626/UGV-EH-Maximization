#!/usr/bin/env python
"""
Trace kappa_1 against the number of episodes used to estimate it.

    python kappa_trace.py Certify_Probe/*_steps.csv

WHY THIS IS NEEDED

certify_probe.py reports one kappa_1 per run, fitted over all episodes,
so nothing in its output says whether that value has settled. It
matters because the estimator is biased in a known direction:
kappa_1 is fitted through per-bin MINIMA of V against distance, and the
minimum of a finite sample can only overestimate the true infimum. So
the fitted slope starts high and falls as episodes accumulate, and
R = (c + L_V D)/kappa_1 correspondingly starts small and grows.

An unsettled kappa_1 therefore understates the radius -- an error in
the unsafe direction. Reporting a value that is still falling claims a
tighter guarantee than the data supports.

WHAT TO LOOK FOR

Run this on the existing _steps.csv; no re-run is required, since the
per-step (V, d) pairs are already recorded. Then:

  * kappa_1 flat over the last third of the trace -> settled, quote it
  * still falling at the final episode count -> collect more episodes,
    or quote kappa_1 as an upper estimate and R as a lower one, saying
    so explicitly

The bin count is held fixed across the trace. Varying it would change
the estimator between points and confound the thing being measured.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

QUANTILE_BINS = 24


def envelope_slope(d, V, n_bins=QUANTILE_BINS, min_per_bin=5):
    """
    Least-squares slope through the origin of the per-bin minima.

    Identical to certify_probe.py's fit, so the final point of the
    trace reproduces the value in the summary JSON rather than merely
    resembling it.
    """
    keep = d > 0
    d, V = d[keep], V[keep]
    if len(d) < 10:
        return np.nan, 0
    edges = np.linspace(0.0, d.max(), n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < min_per_bin:
            continue
        xs.append(0.5 * (lo + hi))
        ys.append(V[m].min())
    if len(xs) < 2:
        return np.nan, len(xs)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    denom = float((xs * xs).sum())
    return (float((xs * ys).sum() / denom) if denom > 0 else np.nan), len(xs)


def trace(path, step=5):
    d = pd.read_csv(path)
    eps = sorted(d.episode.unique())
    rows = []
    for k in list(range(step, len(eps), step)) + [len(eps)]:
        sub = d[d.episode.isin(eps[:k])]
        k1, nb = envelope_slope(sub.d.values, sub.V.values)
        rows.append({"episodes": k, "kappa1": k1, "bins_used": nb,
                     "n_states": len(sub)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("steps_csv", nargs="+")
    ap.add_argument("--step", type=int, default=5,
                    help="episode increment between trace points")
    args = ap.parse_args()

    paths = []
    for p in args.steps_csv:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    for p in paths:
        if not os.path.isfile(p):
            print("skip %s" % p)
            continue
        t = trace(p, args.step)
        name = os.path.basename(p).replace("_steps.csv", "")
        print("\n=== %s ===" % name)
        print(t.to_string(index=False,
                          formatters={"kappa1": lambda v: "%8.4f" % v}))

        k = t.kappa1.dropna().values
        if len(k) >= 4:
            # Compare the last third against the preceding third. A
            # drop of more than a few percent means the estimate is
            # still moving and R would be quoted too small.
            third = max(1, len(k) // 3)
            early = k[-2 * third:-third].mean()
            late = k[-third:].mean()
            drift = (late - early) / early if early else np.nan
            verdict = ("SETTLED" if abs(drift) < 0.03 else
                       "STILL FALLING" if drift < 0 else "RISING")
            print("  last-third mean %.4f vs preceding %.4f "
                  "-> %+.1f%%  %s" % (late, early, 100 * drift, verdict))
            if verdict == "STILL FALLING":
                print("  kappa_1 has not converged; R computed from it is a "
                      "LOWER bound on the true radius.")


if __name__ == "__main__":
    main()
