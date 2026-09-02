#!/usr/bin/env python
"""
Measure the certification constants from a saved checkpoint.

    python certify_probe.py CHECKPOINT.pt [--episodes 30] [--out DIR]

WHAT THIS PRODUCES AND WHY IT NEEDS A FORWARD PASS

Proposition 1 reports a radius

    R = kappa_1^{-1}( c + L_V D ),    c = varrho_eff / (1 - alpha)

and three of those quantities cannot be recovered from the CSVs a
training run writes:

  kappa_1   a LOWER envelope of V_eta over distance to the target set.
            training_metrics.csv logs per-update aggregates -- means,
            quantiles, a batch minimum -- and an envelope is a
            per-state minimum over a distance bin. `ood_range` is the
            closest column and it is a MEAN spread across probe
            states, so using it as a lower bound would understate R in
            the unsafe direction.

  nu        the defect in the factorisation V_eta(z) ~ Vtilde(SoC).
            Needs the joint (V, SoC) scatter; only the marginals are
            logged.

  D         sup ||F(z,t) - z|| in the certified coordinates. Available
            in SoC-only coordinates from the evaluation CSV, but not
            in the latent, which is never written to disk.

All three are properties of a FORWARD PASS, not of optimisation, so
this script re-rolls episodes from a trained checkpoint rather than
retraining. Thirty episodes is a few minutes.

WHAT IT DOES NOT DO

It does not certify anything. It measures the constants that
Proposition 1 consumes. The decrease condition itself is already
measured during training as `cert_alpha_q05`; this fills in the terms
that turn a rate into a radius.
"""

import argparse
import csv
import json
import os
import sys
from collections import deque

import numpy as np
import torch
from pvlib import solarposition

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# main.py guards its training entry point with __main__, so importing
# it gives the constants, the observation builder and the model
# dispatch without starting a run.
import main as M
from environment import sim_env


# torch.load defaults to weights_only=True from PyTorch 2.6. A
# full checkpoint carries the numpy RNG state, whose tuple contains a
# numpy array, and numpy objects are not on the allowlist -- so the
# load fails. These files are written by our own runs, so unpickling
# them is no more dangerous than running the script that wrote them;
# the flag is passed explicitly rather than left to the default so the
# behaviour does not depend on the installed torch version. Older
# versions have no such kwarg, hence the fallback.
def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model(ckpt_path, device):
    """
    Reconstruct the architecture the checkpoint was saved from.

    The variant is taken from the environment, exactly as a training
    run would, so the same LTAC_VARIANT / LTAC_ENCODER_C / ... that
    produced the checkpoint must be set when probing it. A mismatch
    surfaces as a state_dict key or shape error rather than as a
    silently wrong model, which is the failure mode worth having.
    """
    variant = M.TRANSFORMER_VARIANT
    if variant == "cost_lipschitz":
        from lipschitz_transformer import LipschitzCostTransformerActorCritic as C
        model = C(M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
                  sequence_length=M.SEQUENCE_LENGTH,
                  softplus_beta=M.COST_BETA_INIT,
                  beta_gain_target=M.COST_BETA_GAIN_TARGET,
                  spectral_critic=True)
    elif variant in M.COST_VARIANTS:
        from cost_transformer import CostTransformerActorCritic as C
        model = C(M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
                  sequence_length=M.SEQUENCE_LENGTH,
                  softplus_beta=M.COST_BETA_INIT,
                  beta_gain_target=M.COST_BETA_GAIN_TARGET,
                  spectral_critic=(variant != "cost_plain"))
    else:
        from transformer import TransformerActorCritic as C
        model = C(M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
                  sequence_length=M.SEQUENCE_LENGTH)

    state = _torch_load(ckpt_path)
    # Format-2 checkpoints are a dict of run state with the weights
    # under "model"; older ones are a bare state_dict. Accept both, so
    # the probe works on checkpoints written before full checkpointing
    # existed.
    if isinstance(state, dict) and "model" in state and "format" in state:
        print(f"[probe] full checkpoint, saved at episode "
              f"{state.get('episode', '?')}")
        state = state["model"]
    elif hasattr(state, "state_dict"):
        state = state.state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] {len(missing)} missing, {len(unexpected)} unexpected keys")
        print("       check LTAC_VARIANT and the encoder settings match "
              "the run that wrote this checkpoint")
    return model.to(device).eval()


@torch.no_grad()
def value_of(model, seq):
    """V_eta >= 0, in the sign convention of the merged construction."""
    v = model.value_only(seq)
    # value_only returns V^pi <= 0 for the cost arms; V_cost = -V^pi.
    return float(-v.reshape(-1)[0].item()) if M.IS_COST else float(v.reshape(-1)[0].item())


def rollout(model, env, device, n_episodes, seed0=10_000):
    """
    Deterministic rollouts, logging what the constants need.

    Uses model.act(..., True) so the policy is greedy: the certificate
    is a statement about the deployed controller, not about the
    exploring one.
    """
    rows, ep_stats = [], []
    for ep in range(n_episodes):
        torch.manual_seed(seed0 + ep)
        np.random.seed(seed0 + ep)
        env.place_devices()
        env.reset()
        x, y, yaw = env.ch.get_position()
        h = deque([M.obs(env, x, y, yaw, 0)] * M.SEQUENCE_LENGTH,
                  maxlen=M.SEQUENCE_LENGTH)

        # Per-episode validation accumulators, mirroring the fields the
        # training run's final evaluation writes, so this script is a
        # drop-in replacement for it at a chosen checkpoint rather than
        # a second, differently-defined measurement.
        ep_stat = {"episode": ep, "steps": 0, "total_reward": 0.0,
                   "min_batt": 100.0, "path_m": 0.0, "motion_mAh": 0.0,
                   "idle_mAh": 0.0, "solar_w": [], "abs_action": [],
                   "start_x": x, "start_y": y,
                   "start_batt": env.ch.get_battery()}

        for step in range(M.MAX_STEPS_PER_EPISODE):
            seq = M.seq_tensor(h, device)
            with torch.no_grad():
                a, _raw, _lp, _v = model.act(seq, True)
                latent = model.encode(seq).reshape(-1).detach().cpu().numpy()
            V = value_of(model, seq)
            soc = env.ch.get_battery() / 100.0

            a_np = a[0].detach().cpu().numpy()
            dx, dy = a_np * M.MAX_MOVE_PER_STEP
            tx = float(np.clip(x + dx, 0, env.dim - 1))
            ty = float(np.clip(y + dy, 0, env.dim - 1))
            tel, _ = env.step_simulation(step, tx, ty)
            x, y, yaw = env.ch.get_position()
            soc_next = env.ch.get_battery() / 100.0

            # Reward, recomputed exactly as the training run's
            # evaluation does, so the numbers are comparable rather
            # than merely similar.
            t_idx = min(step, len(env.times) - 1)
            sol = solarposition.get_solarposition(
                env.times[t_idx], env.lat_center + y * env.stp,
                env.long_center + x * env.stp)
            aft = env.get_obfuscation(x, y, t_idx, sol.azimuth.iloc[0],
                                      sol.apparent_zenith.iloc[0]).flatten()
            if M.IS_COST:
                rew = M.reward_fn_cost(aft, tel, soc_next)[0]
            else:
                rew = M.reward_fn(aft, tel, soc_next)[0]

            ep_stat["steps"] += 1
            ep_stat["total_reward"] += float(rew)
            ep_stat["min_batt"] = min(ep_stat["min_batt"], soc_next * 100.0)
            ep_stat["path_m"] += float(tel.get("path_m", 0.0))
            ep_stat["motion_mAh"] += float(tel.get("motion_mAh", 0.0))
            ep_stat["idle_mAh"] += float(tel.get("idle_mAh", 0.0))
            ep_stat["solar_w"].append(float(tel.get("solar_w", 0.0)))
            ep_stat["abs_action"].append(float(np.abs(a_np).mean()))
            h.append(M.obs(env, x, y, yaw, min(step + 1,
                                               M.MAX_STEPS_PER_EPISODE - 1)))

            # Both of these are forward passes and neither needs a
            # graph. Without no_grad the encode() output carries one,
            # and .numpy() refuses on a tensor that requires grad --
            # the first call above was guarded, this one was not.
            with torch.no_grad():
                seq_next = M.seq_tensor(h, device)
                V_next = value_of(model, seq_next)
                latent_next = model.encode(seq_next).reshape(-1).detach().cpu().numpy()

            rows.append({
                "episode": ep, "step": step,
                "soc": soc, "soc_next": soc_next,
                # distance to the target set, in SoC units
                "d": max(M.SOC_TARGET - soc, 0.0),
                "V": V, "V_next": V_next,
                "dV": V_next - V,
                # per-step displacement in the certified coordinates:
                # SoC alone, and SoC together with the latent
                "step_soc": abs(soc_next - soc),
                "step_full": float(np.sqrt((soc_next - soc) ** 2
                                           + np.sum((latent_next - latent) ** 2))),
                "alive": 1,
            })
            prev_latent = latent
            if env.ch.get_battery() <= 0.0:
                break

        disp = float(np.hypot(x - ep_stat["start_x"], y - ep_stat["start_y"]))
        ep_stat["final_batt"] = env.ch.get_battery()
        ep_stat["net_disp"] = disp
        # Reported alongside tortuosity because the denominator is
        # nearly constant here, and a station-keeping policy drives it
        # to zero while tortuosity diverges.
        ep_stat["disp_over_path"] = (disp / ep_stat["path_m"]
                                     if ep_stat["path_m"] > 0 else float("nan"))
        ep_stat["tortuosity"] = (ep_stat["path_m"] / disp
                                 if disp > 1e-6 else float("nan"))
        ep_stat["mean_solar_w"] = float(np.mean(ep_stat["solar_w"])) \
            if ep_stat["solar_w"] else float("nan")
        ep_stat["mean_abs_action"] = float(np.mean(ep_stat["abs_action"])) \
            if ep_stat["abs_action"] else float("nan")
        ep_stat["survived"] = int(ep_stat["steps"] >= M.MAX_STEPS_PER_EPISODE)
        for k in ("solar_w", "abs_action", "start_x", "start_y"):
            ep_stat.pop(k)
        ep_stats.append(ep_stat)

        print(f"  episode {ep+1}/{n_episodes}: steps={ep_stat['steps']} "
              f"minSOC={ep_stat['min_batt']:.2f} path={ep_stat['path_m']:.0f}",
              flush=True)
    return rows, ep_stats


def envelopes(rows, n_bins=24):
    """
    Lower and upper envelopes of V against distance.

    kappa_1 must bound V from BELOW at every state, so each bin
    contributes its minimum, not its mean. That is the whole reason
    this cannot come from the training CSVs, which log means and batch
    minima but never a minimum conditioned on distance.

    A bin minimum over finitely many samples is itself an estimate
    biased HIGH (the true infimum over the bin can only be lower), so
    the fitted slope is optimistic and should be reported as such.
    """
    d = np.array([r["d"] for r in rows])
    V = np.array([r["V"] for r in rows])
    keep = d > 0
    d, V = d[keep], V[keep]
    if len(d) < 10:
        return None

    edges = np.linspace(0.0, d.max(), n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 5:
            continue
        out.append({"d_lo": lo, "d_hi": hi, "d_mid": 0.5 * (lo + hi),
                    "n": int(m.sum()),
                    "V_min": float(V[m].min()),
                    "V_mean": float(V[m].mean()),
                    "V_max": float(V[m].max())})
    return out


def fit_linear_through_origin(x, y):
    """Least-squares slope for a class-K linear candidate k(d) = s*d."""
    x = np.asarray(x); y = np.asarray(y)
    denom = float((x * x).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="Certify_Probe")
    ap.add_argument("--bins", type=int, default=24)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = build_model(args.checkpoint, device)
    env = sim_env("test", 20, M.MAX_STEPS_PER_EPISODE)
    env.set_view_dist(M.VIEW_DISTANCE)

    print(f"variant={M.TRANSFORMER_VARIANT}  episodes={args.episodes}")
    rows, ep_stats = rollout(model, env, device, args.episodes)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    per_step = os.path.join(args.out, f"{stem}_steps.csv")
    with open(per_step, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Validation table: one row per episode, plus the aggregate the
    # results section reports. Emitted here so every number in the
    # paper -- performance and certification alike -- comes from the
    # SAME checkpoint. Mixing an episode-500 probe with an
    # end-of-training evaluation would certify a radius for a
    # controller other than the one whose performance is tabulated,
    # and the run lengths differ across seeds so there is no common
    # final episode to align on.
    val_path = os.path.join(args.out, f"{stem}_validation.csv")
    with open(val_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_stats[0].keys()))
        w.writeheader()
        w.writerows(ep_stats)

    def agg(key):
        v = np.array([e[key] for e in ep_stats], dtype=float)
        v = v[np.isfinite(v)]
        # ddof=1: the SAMPLE standard deviation, matching the
        # training run's validation summary, which uses
        # pandas .std(ddof=1). numpy defaults to the population form
        # (ddof=0), and the two differ by ~5% at n=10 -- enough for
        # the same quantity to print differently in two tables of the
        # same paper.
        if not len(v):
            return (float("nan"),) * 4
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        return float(v.mean()), sd, float(v.min()), float(v.max())

    validation = {
        "survived": int(sum(e["survived"] for e in ep_stats)),
        "episodes": len(ep_stats),
    }
    for key in ("min_batt", "final_batt", "path_m", "tortuosity",
                "disp_over_path", "mean_solar_w", "mean_abs_action",
                "motion_mAh", "total_reward", "steps"):
        m, sd, lo, hi = agg(key)
        validation[key] = {"mean": m, "std": sd, "min": lo, "max": hi}

    env_rows = envelopes(rows, args.bins)
    env_path = os.path.join(args.out, f"{stem}_envelope.csv")
    if env_rows:
        with open(env_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(env_rows[0].keys()))
            w.writeheader()
            w.writerows(env_rows)

    d = np.array([r["d"] for r in rows])
    V = np.array([r["V"] for r in rows])
    pos = d > 0

    # kappa_1 / kappa_2 as linear class-K candidates through the origin
    k1 = k2 = float("nan")
    if env_rows:
        k1 = fit_linear_through_origin([e["d_mid"] for e in env_rows],
                                       [e["V_min"] for e in env_rows])
        k2 = fit_linear_through_origin([e["d_mid"] for e in env_rows],
                                       [e["V_max"] for e in env_rows])

    # factorisation defect: spread of V about its conditional mean in SoC
    nu = float("nan")
    if env_rows:
        resid = []
        for e in env_rows:
            m = (d >= e["d_lo"]) & (d < e["d_hi"])
            resid.extend(np.abs(V[m] - e["V_mean"]).tolist())
        nu = float(np.percentile(resid, 95)) if resid else float("nan")

    D_soc = float(max(r["step_soc"] for r in rows))
    D_full = float(max(r["step_full"] for r in rows))

    # L_V in SoC coordinates: the steepest bin-to-bin slope of the
    # conditional mean. In the latent this is the spectral bound and is
    # not measurable here.
    L_V_soc = float("nan")
    if env_rows and len(env_rows) > 1:
        slopes = [abs(b["V_mean"] - a_["V_mean"]) / max(b["d_mid"] - a_["d_mid"], 1e-9)
                  for a_, b in zip(env_rows[:-1], env_rows[1:])]
        L_V_soc = float(max(slopes))

    summary = {
        "validation": validation,
        "checkpoint": args.checkpoint,
        "variant": M.TRANSFORMER_VARIANT,
        "episodes": args.episodes,
        "steps": len(rows),
        "states_outside_target": int(pos.sum()),
        "kappa1_linear_slope": k1,
        "kappa2_linear_slope": k2,
        "nu_p95": nu,
        "D_soc": D_soc,
        "D_full_soc_plus_latent": D_full,
        "L_V_soc_maxslope": L_V_soc,
        "V_mean": float(V.mean()), "V_min": float(V.min()),
        "d_max": float(d.max()),
    }
    with open(os.path.join(args.out, f"{stem}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("  VALIDATION (this checkpoint)")
    print(f"    survived                   "
          f"{validation['survived']}/{validation['episodes']}")
    for key in ("min_batt", "path_m", "tortuosity", "mean_solar_w",
                "mean_abs_action"):
        v = validation[key]
        print(f"    {key:26s} mean {v['mean']:8.3f}  sd {v['std']:7.3f}  "
              f"worst {v['min']:8.3f}")
    print()
    print("  CERTIFICATION CONSTANTS")
    for k, v in summary.items():
        if k in ("validation", "checkpoint"):
            continue
        print(f"    {k:26s} {v}")
    print()
    print(f"wrote {per_step}")
    print(f"wrote {val_path}")
    if env_rows:
        print(f"wrote {env_path}")
    print(f"wrote {os.path.join(args.out, stem + '_summary.json')}")
    print()
    print("To obtain R: c = varrho / (1 - alpha_hat) using the margin and")
    print("cert_alpha_q05 from the training CSV, then")
    print("    R = (c + L_V * D) / kappa1_linear_slope")
    print("with L_V the spectral bound if certifying in the latent, or")
    print("L_V_soc_maxslope with D_soc if certifying in SoC alone.")
    print()
    print("Report kappa1_linear_slope as an ESTIMATE: a per-bin minimum")
    print("over finitely many samples is biased above the true infimum,")
    print("so the fitted slope is optimistic and R correspondingly small.")


if __name__ == "__main__":
    main()
