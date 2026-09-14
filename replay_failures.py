#!/usr/bin/env python
"""
Replay captured failures with a planner, to decide whether they were.

    python replay_failures.py --policy exact Certify_Probe/failures/*.npz

WHAT THIS DECIDES

certify_probe excludes early-terminating episodes from D, on the
grounds that a trajectory leaving the certified region is not in the
supremum's domain. That is definitionally true but it is not an
argument that the failure was the environment's fault, and a reviewer
is right to press on it: excluding failures from a supremum used in a
safety bound needs more than domain membership.

The rollout alone cannot settle it. Harvest observed during an episode
is a function of where the policy went, not of what the map offered --
in the case that prompted this script the vehicle covered 29 m of an
800 m map, so its 3.5 W mean says nothing about the other 799 m.

What settles it is running a planner with full knowledge of the solar
field on the SAME draw. If the planner also depletes, no controller on
this platform survives that environment and the exclusion is a
statement about the draw. If the planner survives, the exclusion is a
policy failure being removed from a safety bound, and the paper should
say so.

ON REPRODUCIBILITY

The capture stores a SHA-256 of the realised topography and foliage.
Reseeding reproduces terrain, foliage and start position, which draw
from np.random; device placement draws from Python's random, seeded
once at construction, so a replay matches only if it makes the same
number of place_devices calls. This script recomputes the hash and
REFUSES to report a verdict on a mismatch rather than comparing two
different environments and calling it evidence. The fields themselves
are also stored, so a mismatch can be diagnosed rather than guessed
at.
"""

import argparse
import glob
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def env_hash(env):
    h = hashlib.sha256()
    h.update(np.asarray(env.topo_mask, dtype=np.float32).tobytes())
    h.update(np.asarray(env.foliage_mask, dtype=np.float32).tobytes())
    return h.hexdigest()


def rebuild(case, view_dist=20, steps=720):
    """Recreate the environment the failure was drawn from."""
    import random as _pyrandom
    import torch
    from environment import sim_env

    seed0 = int(case["seed0"])
    ep = int(case["episode"])
    env = sim_env("test", 20, steps)
    env.set_view_dist(view_dist)

    # Same sequence the probe performed: reseed, place, reset.
    torch.manual_seed(seed0 + ep)
    np.random.seed(seed0 + ep)
    _pyrandom.seed(seed0 + ep)
    env.place_devices()
    env.reset()
    return env


def force_state(env, case):
    """Put the vehicle where the failing episode started."""
    x = float(case["start_x"])
    y = float(case["start_y"])
    b = float(case["start_batt"])
    env.ch.update_telemetry(x, y, 0.0)
    # Battery is stored in mAh internally; set through state of charge
    # so the conversion stays in one place.
    env.ch.battery_mAh = env.ch.max_capacity_mAh * (b / 100.0)
    return x, y, b


def run_planner(env, steps, policy_type):
    """
    Step the environment under a planner for the full horizon.

    The planner consumes the same observation the learned arms do --
    act() takes a [batch, time, features] window and reads only the
    last timestep, being memoryless by construction. Building the
    observation through main.obs and main.seq_tensor rather than
    reimplementing it keeps the comparison honest: any difference in
    how the patch or scalars are assembled would compare two different
    problems.
    """
    import torch
    import main as M
    from pso_policy import PSOPolicy, ExactPolicy
    Planner = ExactPolicy if policy_type == "exact" else PSOPolicy
    # Constructed exactly as main.py does, so the planner sees the
    # same patch size and scalar block as every other arm.
    input_dim = (M.VIEW_DISTANCE * 2 + 1) ** 2 + M.SCALAR_DIM
    pol = Planner(input_dim=input_dim,
                  view_distance=M.VIEW_DISTANCE,
                  scalar_dim=M.SCALAR_DIM)

    from collections import deque
    device = torch.device("cpu")
    x, y, yaw = env.ch.get_position()
    hist = deque([M.obs(env, x, y, yaw, 0)] * M.SEQUENCE_LENGTH,
                 maxlen=M.SEQUENCE_LENGTH)

    min_b = env.ch.get_battery()
    path = 0.0
    for step in range(steps):
        seq = M.seq_tensor(hist, device)
        a = pol.act(seq, True)[0]
        dx, dy = np.asarray(a[0].detach().cpu().numpy(),
                            dtype=float) * M.MAX_MOVE_PER_STEP
        tx = float(np.clip(x + dx, 0, env.dim - 1))
        ty = float(np.clip(y + dy, 0, env.dim - 1))
        env.step_simulation(step, tx, ty)
        x, y, yaw = env.ch.get_position()
        hist.append(M.obs(env, x, y, yaw,
                          min(step + 1, M.MAX_STEPS_PER_EPISODE - 1)))
        path += float(env.ch.step_path_m)
        min_b = min(min_b, env.ch.get_battery())
        if env.ch.get_battery() <= 0.0:
            return False, step + 1, min_b, path
    return True, steps, min_b, path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--policy", choices=("exact", "pso"), default="exact")
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args()

    paths = []
    for p in args.cases:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    verdicts = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        case = np.load(p)
        print("\n=== %s ===" % os.path.basename(p))
        print("  learned policy: died at step %d, min charge %.2f%%, "
              "path %.1f m, mean harvest %.2f W"
              % (int(case["steps_survived"]), float(case["min_batt"]),
                 float(case["path_m"]), float(case["mean_solar_w"])))

        env = rebuild(case, steps=args.steps)
        got = env_hash(env)
        want = str(case["env_sha256"])
        if got != want:
            print("  ENVIRONMENT MISMATCH: replay produced a different draw.")
            print("    expected %s" % want[:16])
            print("    got      %s" % got[:16])
            print("  No verdict. The stored topo/foliage arrays are in the "
                  "case file for diagnosis; the likely cause is a differing "
                  "number of place_devices calls before reset.")
            verdicts.append((p, None))
            continue
        print("  environment verified (sha256 %s)" % got[:16])

        x, y, b = force_state(env, case)
        print("  replaying from x=%.1f y=%.1f b=%.2f%% under %s"
              % (x, y, b, args.policy))
        survived, nsteps, min_b, path = run_planner(env, args.steps, args.policy)
        print("  planner: %s at step %d, min charge %.2f%%, path %.1f m"
              % ("SURVIVED" if survived else "died", nsteps, min_b, path))
        verdicts.append((p, survived))

    print("\n" + "=" * 66)
    ok = [v for _, v in verdicts if v is not None]
    if not ok:
        print("no case could be verified; no conclusion available")
    else:
        n_surv = sum(1 for v in ok if v)
        print("planner survived %d of %d verified failure cases" % (n_surv, len(ok)))
        if n_surv == 0:
            print("  -> no controller on this platform survives these draws.")
            print("     The exclusion from D is a statement about the")
            print("     environment, and can be reported as such.")
        else:
            print("  -> the planner survives where the learned policy did not.")
            print("     These are controller failures, not infeasible draws,")
            print("     and excluding them from a safety supremum is not")
            print("     defensible on domain-membership grounds alone.")
    print("=" * 66)


if __name__ == "__main__":
    main()
