"""
Paired comparison of two trained checkpoints over identical environments.

WHY PAIRED
----------
Each episode randomizes terrain, foliage, device placement, start
position and start state of charge. Per-episode reward std measured on
the last run was 27.86, so with 10 independent episodes per arm the 95%
CI half-width is +-17 and only differences above ~35 reward are
resolvable at all.

Running BOTH models through the SAME seeded environments removes that
variance from the comparison. The statistic becomes the per-episode
DIFFERENCE, whose variance is typically several times smaller, so the
same number of episodes resolves a much smaller effect. This is a
paired design rather than two independent samples, and it is reported
with a paired t-test plus a bootstrap CI.

ARCHITECTURE DETECTION
----------------------
Checkpoints from different points in the project have different
architectures. The baseline in particular was rebuilt (round 19) to
match the Lyapunov trunk; earlier baseline checkpoints have neither
`position_embedding` nor `attention_pool` and use a 2-layer critic.
The loader inspects state_dict keys and instantiates whichever class
matches, so old and new checkpoints both load.

USAGE
-----
    python compare_checkpoints.py \
        --a  rl_csv_lyapunov_.../checkpoints/best.pt --a-name lyapunov \
        --b  rl_csv_normal_.../checkpoints/best.pt   --b-name baseline \
        --episodes 300

Writes paired_comparison.csv (one row per episode per arm) and
paired_comparison_summary.csv.
"""

import argparse
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import main as M

# --------------------------------------------------------------------
# --------------------------------------------------------------------
class LegacyTransformerActorCritic(nn.Module):
    """
    The pre-round-19 baseline: no positional embedding, no attention
    pooling, no closing LayerNorm, 2-layer critic, free log_std, and a
    CLAMPED sample scored under the unclamped Normal.

    Kept verbatim so checkpoints trained with it still load. Do not
    "fix" it -- it has to match the weights it is loading.
    """

    def __init__(self, view_dist, scalar_dim=8, action_dim=2, d_model=128,
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.patch_dim = (2 * int(view_dist) + 1) ** 2
        self.input_dim = self.patch_dim + scalar_dim
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, action_dim))
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, sequence):
        sequence = sequence.to(torch.float32)
        x = self.encoder(self.input_projection(sequence))[:, -1]
        return torch.tanh(self.actor(x)), self.critic(x).squeeze(-1)

    def act(self, sequence, deterministic=False):
        mean, value = self(sequence)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        raw = dist.mean if deterministic else dist.rsample()
        action = raw.clamp(-0.999, 0.999)
        return action, dist.log_prob(action).sum(-1), value

def load_checkpoint(path, device):
    """Instantiate whichever architecture the state_dict actually has."""
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    keys = set(sd.keys())

    if any(k.startswith("lyapunov.") for k in keys):
        kind = "lyapunov"
        from lyupnov_transformer import LyapunovTransformerActorCritic
        model = LyapunovTransformerActorCritic(
            view_dist=M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
            sequence_length=M.SEQUENCE_LENGTH)
    elif "position_embedding" in keys or "attention_pool.weight" in keys:
        kind = "baseline (round 19+)"
        from transformer import TransformerActorCritic
        model = TransformerActorCritic(
            M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
            sequence_length=M.SEQUENCE_LENGTH)
    else:
        kind = "baseline (legacy)"
        model = LegacyTransformerActorCritic(
            M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, kind, list(missing), list(unexpected)

def select_action(model, seq):
    """Normalize the differing act() return arities to just the action."""
    out = model.act(seq, True) if _accepts_deterministic(model) else model.act(seq)
    return out[0]

def _accepts_deterministic(model):
    import inspect
    return "deterministic" in inspect.signature(model.act).parameters

# --------------------------------------------------------------------
# --------------------------------------------------------------------
def seed_all(seed):
    """
    Both RNGs must be set: environment.py uses np.random for terrain,
    foliage and start position, and `random` for device placement;
    ugv_simulator.py uses `random` for the start state of charge.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_episode(model, env, device, seed):
    seed_all(seed)
    env.place_devices()
    env.reset()
    x, y, yaw = env.ch.get_position()
    h = deque([M.obs(env, x, y, yaw, 0)] * M.SEQUENCE_LENGTH,
              maxlen=M.SEQUENCE_LENGTH)

    start_batt = env.ch.get_battery()
    start_x, start_y = x, y
    total = directional = battery = movement = 0.0
    idle = motion = path = turn = 0.0
    socs = [start_batt]
    amags = []
    prev_action = None
    smoothness = 0.0
    steps = 0
    aft_batt = start_batt

    for step in range(M.MAX_STEPS_PER_EPISODE):
        seq = M.seq_tensor(h, device)
        with torch.no_grad():
            a = select_action(model, seq)
        current_action = a[0].detach().cpu().numpy()

        if prev_action is None:
            smoothness = 0.0
        else:
            smoothness = float(np.sum((current_action - prev_action) ** 2))

        tx = x + float(current_action[0]) * M.MAX_MOVE_PER_STEP
        ty = y + float(current_action[1]) * M.MAX_MOVE_PER_STEP
        b_batt = env.ch.get_battery()
        env.step(tx, ty, step)

        nx, ny, nyaw = env.ch.get_position()
        from pvlib import solarposition
        sol = solarposition.get_solarposition(
            env.times[min(step, len(env.times) - 1)],
            env.lat_center + ny * env.stp, env.long_center + nx * env.stp)
        aft = env.get_obfuscation(
            nx, ny, min(step, len(env.times) - 1),
            sol.azimuth.iloc[0], sol.apparent_zenith.iloc[0]).flatten()
        aft_batt = env.ch.get_battery()
        tel = env.ch.get_telemetry()

        rt, rd, rb, rm = M.reward_fn(aft, tel, aft_batt - b_batt, current_action)
        rew = rt - M.ACTION_SMOOTHNESS * smoothness

        total += rew
        directional += rd
        battery += rb
        movement += rm
        idle += float(env.ch.step_idle_mAh)
        motion += float(env.ch.step_motion_mAh)
        path += float(env.ch.step_path_m)
        turn += float(env.ch.step_turn_integral)
        socs.append(aft_batt)
        amags.append(float(np.hypot(current_action[0], current_action[1])))

        prev_action = current_action
        x, y, yaw = nx, ny, nyaw
        h.append(M.obs(env, x, y, yaw,
                       min(step + 1, M.MAX_STEPS_PER_EPISODE - 1)))
        steps = step + 1
        if aft_batt <= 0:
            break

    # SOC -- exact, and identical to certify_stability.py.
    soc = np.asarray(socs) / 100.0
    V = np.maximum(M.SOC_TARGET - soc, 0.0) / M.SOC_TARGET
    dV = np.diff(V)
    Vf = V[:-1]
    outside = Vf > M.LYAPUNOV_BALL
    slack = dV + M.LYAPUNOV_ALPHA * Vf + M.LYAPUNOV_MARGIN
    if outside.sum():
        viol = float((slack[outside] > 0).mean())
        mdV = float(dV[outside].mean())
        worst = float(slack[outside].max())
    else:
        viol, mdV, worst = 0.0, 0.0, float("nan")

    net = float(math.hypot(x - start_x, y - start_y))
    return dict(
        seed=seed, steps=steps, survived=int(steps >= M.MAX_STEPS_PER_EPISODE and aft_batt > 0),
        start_battery=start_batt, final_battery=aft_batt, min_battery=float(min(socs)),
        total_reward=total, directional_reward=directional,
        battery_reward=battery, movement_penalty=movement,
        idle_mAh=idle, motion_mAh=motion, path_m=path, turn_integral=turn,
        net_displacement_m=net, tortuosity=path / max(net, 1e-6),
        mean_abs_action=float(np.mean(amags)) if amags else 0.0,
        lyap_violation_rate=viol, lyap_mean_dV=mdV, lyap_worst_slack=worst,
        lyap_in_ball_rate=float(1.0 - outside.mean()) if len(Vf) else 1.0,
    )

# --------------------------------------------------------------------
def paired_stats(a, b, label):
    d = a - b
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("nan")
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wins = int((d > 0).sum())
    return dict(metric=label, mean_a=a.mean(), mean_b=b.mean(),
                mean_diff=mean, paired_sd=sd, t=t,
                ci_lo=lo, ci_hi=hi, a_wins=wins, n=n,
                significant=bool(lo > 0 or hi < 0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=20260729)
    ap.add_argument("--out", default="paired_comparison")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    models = {}
    for name, path in ((args.a_name, args.a), (args.b_name, args.b)):
        m, kind, missing, unexpected = load_checkpoint(path, device)
        print(f"{name:12s} {path}")
        print(f"{'':12s} architecture: {kind}")
        if missing:
            print(f"{'':12s} WARNING missing keys ({len(missing)}): {missing[:4]}")
        if unexpected:
            print(f"{'':12s} WARNING unexpected keys ({len(unexpected)}): {unexpected[:4]}")
        models[name] = m
    print()

    seeds = [args.seed_base + i for i in range(args.episodes)]
    rows = []
    for i, seed in enumerate(seeds, 1):
        line = f"  [{i:3d}/{len(seeds)}] seed {seed}"
        for name, model in models.items():
            env = M.sim_env()
            r = run_episode(model, env, device, seed)
            r["model"] = name
            rows.append(r)
            line += f" | {name}: rew {r['total_reward']:8.2f} batt {r['final_battery']:5.1f}%"
        print(line, flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}.csv", index=False)

    A = df[df.model == args.a_name].set_index("seed").sort_index()
    B = df[df.model == args.b_name].set_index("seed").sort_index()

    metrics = ["total_reward", "final_battery", "min_battery", "survived",
               "mean_abs_action", "path_m", "motion_mAh", "tortuosity",
               "lyap_violation_rate", "lyap_in_ball_rate"]
    stats = [paired_stats(A[m].values.astype(float),
                          B[m].values.astype(float), m) for m in metrics]
    sdf = pd.DataFrame(stats)
    sdf.to_csv(f"{args.out}_summary.csv", index=False)

    print("\n" + "=" * 100)
    print(f"PAIRED COMPARISON over {args.episodes} identical environments")
    print("=" * 100)
    print(f"{'metric':22s} {args.a_name:>12s} {args.b_name:>12s} "
          f"{'diff':>10s} {'95% CI':>20s} {'wins':>7s} {'sig':>5s}")
    for s in stats:
        ci = f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"
        print(f"{s['metric']:22s} {s['mean_a']:12.4f} {s['mean_b']:12.4f} "
              f"{s['mean_diff']:+10.4f} {ci:>20s} "
              f"{s['a_wins']:3d}/{s['n']:<3d} {'YES' if s['significant'] else '':>5s}")

    print()
    print("diff = A - B, paired per environment. 'wins' counts environments")
    print("where A scored higher. 'sig' means the bootstrap 95% CI excludes 0.")
    print()
    print("Efficiency of pairing on total_reward:")
    ind = math.sqrt(A.total_reward.var(ddof=1) + B.total_reward.var(ddof=1))
    par = (A.total_reward.values - B.total_reward.values).std(ddof=1)
    print(f"  unpaired sd of difference : {ind:8.2f}")
    print(f"  paired   sd of difference : {par:8.2f}")
    if par > 0:
        print(f"  variance reduction        : {ind / par:8.2f}x "
              f"(equivalent to {(ind / par) ** 2:.1f}x more episodes)")

    print(f"\nwrote {args.out}.csv and {args.out}_summary.csv")

    try:
        if M.OUT.exists() and not any(M.OUT.iterdir()):
            M.OUT.rmdir()
    except OSError:
        pass

if __name__ == "__main__":
    main()
