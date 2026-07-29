import csv, math, os, random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pvlib import solarposition
from environment import sim_env, MIN_USABLE_ELEVATION
from lyupnov_transformer import LyapunovTransformerActorCritic
import datetime
import time

TransformerActorCritic = None
ChebyshevTransformer = None
ChebyshevLyapunovTransformerActorCritic = None
PSOPolicy = None

# ============================================================
# Set POLICY_TYPE = "transformer" or "pso"
POLICY_TYPE = "transformer"
if POLICY_TYPE == "transformer":
    # Set TRANSFORMER_VARIANT = "normal" or "chaotic" or "lyapunov"
    #
    #     LTAC_VARIANT=lyapunov python main.py
    #     LTAC_VARIANT=normal   python main.py
    TRANSFORMER_VARIANT = os.environ.get("LTAC_VARIANT", "lyapunov")
    # Set TRANSFORMER_INIT = "normal" or "chaotic"
    if TRANSFORMER_VARIANT == "lyapunov":
        TRANSFORMER_INIT = "normal"
else:
    TRANSFORMER_VARIANT = "normal"
# ============================================================

TOTAL_EPISODES=1000; MAX_STEPS_PER_EPISODE=720; VIEW_DISTANCE=20
SEQUENCE_LENGTH=32; UPDATE_EVERY_EPISODES=2; GAMMA=.99; GAE_LAMBDA=.95
LR=3e-4; MAX_MOVE_PER_STEP=20.0; ENTROPY_COEF=.01; VALUE_COEF=.5

###############################################################
# Lyapunov Hyperparameters
###############################################################
LYAPUNOV_COEF = 0.25; DYNAMICS_COEF = 0.015
BARRIER_COEF = 0.20; ACTION_SMOOTHNESS = 0.01
LYAPUNOV_MARGIN = 0.0001
BATTERY_MARGIN = 0.10; BOUNDARY_MARGIN = 0.10; VEGETATION_MARGIN = 0.05
VELOCITY_MARGIN = 0.05; COMM_MARGIN = 0.10; CLIP_EPS =0.20
LYAPUNOV_ALPHA = 0.001  # see LYAPUNOV_MARGIN above
# Constant-force hinge pull on raw_log_std toward RAW_LOG_STD_TARGET --
# see evaluate_actions()'s raw_log_std_reg docstring. Replaces the
# earlier L2 penalty (whose gradient was proportional to raw_log_std,
# and so weakened exactly as it approached the target -- observed
# twice as a recurring deceleration: doubling that coefficient bought
# a better starting position, but the same slowdown re-emerged closer
# to the target as the entropy bonus's recovering leverage caught up
# again). This form's gradient is a constant +-1 (times this
# coefficient) everywhere above RAW_LOG_STD_TARGET, not proportional
# to raw_log_std itself, so it does not taper as raw_log_std declines.
RAW_LOG_STD_REG_COEF = 3e-2

###############################################################
# PPO optimization budget
#
# meaningful from the second minibatch onward.
###############################################################
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
TARGET_KL = 0.03

KL_EMERGENCY_MULT = 4.0

MEAN_SATURATION_COEF = 1e-2

# Deterministic entropy-temperature schedule, replacing the learned
# (SAC-style) alpha. Decays exponentially from ALPHA_START to
# ALPHA_END over ALPHA_DECAY_EPISODES, then holds at ALPHA_END. 300
# matches the existing curriculum boundary (full Lyapunov/barrier
# weights activate at episode 300), so exploration pressure is
# already low by the time that harder phase begins, same intent as
# before -- just guaranteed by a formula instead of hoped for from a
# gradient that might not converge in time.
ALPHA_START = 1.0
ALPHA_END = 0.01
ALPHA_DECAY_EPISODES = 300
# Reactive safety net (see the entropy check in update()): if a
# batch's mean entropy drops below this floor, alpha jumps to at
# least SAFETY_ALPHA_BOOST regardless of the schedule. Floor sits
# between the achievable range's true minimum (~-1.16, full
# collapse) and the original learned-alpha target (~1.338, the
# smooth bound's midpoint) -- low enough to only trigger on a real
# problem, not on ordinary convergence toward a reasonable
# exploitation level.
SAFETY_STD_FLOOR = 0.02
SAFETY_ALPHA_BOOST = 0.5

###############################################################
# Convergence Monitoring / Checkpointing
#
# Rather than guessing an episode count up front: track moving
# averages of reward and of the Lyapunov/barrier penalties, save
# the best-so-far model whenever the reward average improves, and
# flag (and optionally stop on) convergence once the stability
# constraints are comfortably satisfied AND reward has stopped
# improving for a while. Convergence is only ever checked once the
# full-weight regime (ep>=300) is active -- checking earlier would
# just be measuring the pre-curriculum warmup, not real convergence.
###############################################################
REWARD_WINDOW = 50              # episodes averaged for the reward moving average
STABILITY_WINDOW = 20           # update() calls averaged for Lyapunov/barrier stability
CONVERGENCE_PATIENCE = 10       # retained for logging only; see PLATEAU_SLOPE_FRAC
LYAPUNOV_STABLE_THRESHOLD = 2 * LYAPUNOV_MARGIN

###############################################################
# Ultimate boundedness
###############################################################
SOC_TARGET = 0.90
LYAPUNOV_BALL = 0.05

LYAPUNOV_MAX_VIOLATION_RATE = 0.30

LYAPUNOV_IN_BALL_SUFFICIENT = 0.95
LYAPUNOV_V_SPREAD_FLOOR = 0.05
LYAPUNOV_SPREAD_COEF = 1.0
LYAPUNOV_COLLAPSE_COEF = 1.0

LYAPUNOV_COLLAPSE_STD = 0.02

LYAPUNOV_COLLAPSE_BAND = 0.05 * LYAPUNOV_MARGIN

BARRIER_COLLAPSE_FLOOR = 1e-5
BARRIER_STABLE_THRESHOLD = 0.05

BARRIER_ANCHOR_COEF = 0.5

BARRIER_COLLAPSE_STD = 0.02
CHECKPOINT_EVERY = 100          # periodic safety-net checkpoint, regardless of performance
VALIDATION_EPISODES = 10

AUTO_STOP_ON_CONVERGENCE = False # set True to re-enable early stopping once the criteria are recalibrated for how fast training now converges
# Reward readings are not trustworthy evidence of a real plateau while
# Std is still pinned near the exploration ceiling (LOG_STD_MAX=0.5 ->
# Std~1.65) -- a flat reward there can mean "hasn't started learning
# yet" just as easily as "found the optimum," and the two are
# indistinguishable from reward alone. Require avg_std to have
# actually come down substantially before convergence is even
# eligible to fire, on top of the existing Lyapunov/barrier/reward
# checks.
STD_CLEARED_THRESHOLD = 1.0
# Lyapunov/barrier being stable and reward "not improving" are both
# satisfied just as easily by "found the optimum" as by "hasn't
# started learning yet" -- as seen at episode 378, where reward was
# still indistinguishable from the pre-fix stuck baseline. Require
# avg_reward to have actually cleared a real bar, not just plateaued
# anywhere, before convergence can fire.
REWARD_CONVERGENCE_THRESHOLD = -150.0

###############################################################
# Plateau test
###############################################################
PLATEAU_WINDOW = 200
PLATEAU_T_CRIT = 1.65      # one-sided 95%
PLATEAU_CONSECUTIVE = 10

###############################################################
# Degradation guard
#
# instead of running to 1000.
###############################################################
DEGRADATION_FRAC = 0.15
DEGRADATION_PATIENCE = 10
ACTION_SATURATION_THRESHOLD = 0.97

###############################################################
# Reward coefficients
###############################################################
DIRECTIONAL_COEF = 0.25   # ~33% of the battery span (was 0.05, ~6.5%)
BATTERY_COEF = 2.0        # the objective; unchanged
MOVEMENT_COEF = 0.001     # base rate; scaled by exposure in reward_fn

PARK_COEF = 0.30
PARK_DISTANCE_SCALE = 4.0   # cells; bonus decays as exp(-distance/scale)
PARK_ACTION_SCALE = 0.40
MOVEMENT_EXPOSURE_SCALE = 3.0

# Format as YYYY-MM-DD_HH-MM-SS
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_variant_tag = TRANSFORMER_VARIANT if POLICY_TYPE == "transformer" else POLICY_TYPE
OUT=Path(f"rl_csv_{_variant_tag}"); OUT.mkdir(exist_ok=True)

size = 2 * VIEW_DISTANCE + 1
y, x = np.mgrid[-VIEW_DISTANCE:VIEW_DISTANCE+1,
                -VIEW_DISTANCE:VIEW_DISTANCE+1]
KERNEL_SIGMA = 2.0
kernel = np.exp(-(x**2 + y**2)/(2*KERNEL_SIGMA**2))
kernel /= kernel.sum()
GAUSSIAN_KERNEL = kernel.flatten()

def log_status(ep, total_episodes, steps, avg_reward, final_batt, loss, is_eval=False):
    prefix = "[EVALUATION]" if is_eval else f"[Episode {ep}/{total_episodes}]"
    loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else loss
    print(f"{prefix} Steps: {steps:3} | Reward: {avg_reward:7.2f} | Battery: {final_batt:6.2f}% | Loss: {loss_str}")

SCALAR_DIM = 8

SOC_OBS_INDEX = -(SCALAR_DIM - 4)
X_OBS_INDEX = -SCALAR_DIM
Y_OBS_INDEX = -(SCALAR_DIM - 1)

# Mirror of environment.py's geometry, used to build the boundary
GRID_DIM = 800
BOUNDARY_CENTER = GRID_DIM / 2.0
BOUNDARY_RADIUS = 250.0

def obs(env, x, y, yaw, step):
    sol=solarposition.get_solarposition(env.times[min(step,len(env.times)-1)], env.lat_center+y*env.stp, env.long_center+x*env.stp)
    patch=env.get_obfuscation(x,y,min(step,len(env.times)-1),sol.azimuth.iloc[0],sol.apparent_zenith.iloc[0]).flatten()
    potential = 1.0 - patch

    elevation = 90.0 - sol.apparent_zenith.iloc[0]
    elevation_norm = max(0.0, elevation) / 90.0
    sun_usable = 1.0 if elevation >= MIN_USABLE_ELEVATION else 0.0

    scalars=np.array([x/(env.dim-1),y/(env.dim-1),math.sin(yaw),math.cos(yaw),
                      env.ch.get_battery()/100, sol.azimuth.iloc[0]/360,
                      elevation_norm, sun_usable],np.float32)
    return np.concatenate([potential.astype(np.float32),scalars])

def seq_tensor(history, device):
    return torch.tensor(np.asarray(history),dtype=torch.float32,device=device).unsqueeze(0)

def reward_fn(after, telemetry, delta_batt, action=None):
    # -------
    #
    # --------
    potential_after = 1.0 - after
    score_after = float(np.dot(GAUSSIAN_KERNEL, potential_after))

    px, py, _ = telemetry["previous_position"]
    nx, ny, _ = telemetry["new_position"]

    distance = math.hypot(nx - px, ny - py)

    # -------
    action_magnitude = 0.0
    if action is not None:
        flat = np.asarray(
            action.detach().cpu() if hasattr(action, "detach") else action,
            dtype=np.float64,
        ).reshape(-1)
        if flat.size >= 2:
            action_magnitude = float(math.hypot(flat[0], flat[1]))

    park_factor = math.exp(
        -(distance / PARK_DISTANCE_SCALE)
        - (action_magnitude / PARK_ACTION_SCALE)
    )

    directional_reward = (
        DIRECTIONAL_COEF * score_after
        + PARK_COEF * park_factor * score_after
    )

    movement_penalty = (
        MOVEMENT_COEF * distance
        * (1.0 + MOVEMENT_EXPOSURE_SCALE * score_after)
    )

    battery_reward = BATTERY_COEF * delta_batt

    total = float(directional_reward + battery_reward - movement_penalty)

    return total, float(directional_reward), float(battery_reward), float(movement_penalty)

class RunningVariance:
    """
    Welford-style running mean/variance estimator, updated in
    batches. Used to adaptively rescale value_loss's effective
    weight in the total loss as the raw scale of returns shifts
    over training (return variance isn't fixed -- it changes as the
    policy changes), rather than relying on one fixed VALUE_COEF
    guessed against a single snapshot.

    Deliberately does NOT normalize `values`/`returns` themselves --
    only the loss term's weighting. The critic's raw output
    (`values`, collected via `v.item()` during rollout) is mixed
    directly with raw-scale `r["rewards"]` in GAE
    (`gae=r["rewards"][i]+GAMMA*nxt-r["values"][i]+...`), computed
    entirely outside update(). If the critic were trained to predict
    a normalized target instead, its raw output would silently drift
    into a different scale than GAE assumes, corrupting every
    advantage estimate -- and therefore the policy gradient itself.
    Rescaling only how much the (still raw-scale) MSE contributes to
    the total loss avoids that risk entirely.
    """
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = float(x.mean())
        batch_var = float(x.var(unbiased=False))
        batch_count = x.numel()

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

def compute_batch(rollouts, device):
    """
    Flattens a list of episode rollouts into aligned, batched tensors.

    THE ORDERING BUG THIS FIXES
    ---------------------------
    The previous version built the advantage and return lists with
    `adv.insert(0, gae)` while building the state list with
    `states += r["states"]`. insert(0, ...) PREPENDS; += APPENDS.

    With a single episode per update those agree by accident. With
    UPDATE_EVERY_EPISODES=2 they do not:

        states  ['A0','A1','A2','B0','B1','B2']
        adv     [ B advantages... , A advantages... ]

    Every state/action was therefore paired with an advantage from
    the OTHER episode, and every value prediction was regressed
    against the other episode's return. Because both episodes were
    always exactly MAX_STEPS_PER_EPISODE long, the tensor shapes
    matched and nothing ever raised -- the policy gradient was
    simply noise, and the critic was trained on wrong targets.

    Advantages are accumulated per-episode and then extended onto
    the batch in the same order the states are, so the two can no
    longer drift apart regardless of how many episodes are batched.

    RETURNS
    -------
    `returns` is now adv + values (the TD(lambda) return) rather
    than a raw Monte Carlo discounted sum seeded at zero.

    The old MC form told the critic "zero future reward after the
    final step" on every trajectory. Since episodes essentially
    always hit the 720-step cap rather than dying (steps == 720 for
    all 1000 episodes of the previous run), that bootstrap was wrong
    every single time -- and it disagreed with GAE, which did
    bootstrap correctly from r["bootstrap_value"]. Deriving returns
    from the advantages makes the two consistent by construction.
    """
    states=[]; next_states=[]; actions=[]; oldlp=[]; adv=[]; returns=[]

    for r in rollouts:
        T = len(r["rewards"])
        ep_adv = []
        gae = 0.0
        for i in reversed(range(T)):
            nxt = r.get("bootstrap_value", 0.0) if i == T - 1 else r["values"][i + 1]
            gae = r["rewards"][i] + GAMMA * nxt - r["values"][i] + GAMMA * GAE_LAMBDA * gae
            ep_adv.insert(0, gae)

        adv += ep_adv
        returns += [a + v for a, v in zip(ep_adv, r["values"])]
        states += r["states"]
        next_states += r["next_states"]
        actions += r["actions"]
        oldlp += r["logps"]

    adv = torch.tensor(adv, device=device, dtype=torch.float32)
    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    actions = torch.tensor(np.asarray(actions), dtype=torch.float32, device=device)
    oldlp = torch.tensor(oldlp, device=device, dtype=torch.float32)
    states = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device)

    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    return states, next_states, actions, oldlp, adv, returns

def update(model,opt,rollouts,device, ep, metrics_writer=None, return_var_tracker=None,
           control_params=None, auxiliary_param_list=None):
    model.train()

    if control_params is None:
        control_params = list(model.parameters())
        auxiliary_param_list = []

    states, next_states, actions, oldlp, adv, returns = compute_batch(rollouts, device)

    if TRANSFORMER_VARIANT == "lyapunov":
        if ep < 100:
            lyap_weight = 0.0
            barrier_weight = 0.0
            dynamics_weight = 0.0
        elif ep < 300:
            lyap_weight = 0.10
            barrier_weight = 0.05
            # Reduced from 0.05: weighted dynamics_loss was measured
            # at 6.56x policy_loss even at this "light" weight
            # (dynamics_loss ~1.4, policy_loss ~0.01) -- the encoder's
            # gradient was dominated by dynamics prediction, not
            # reward, essentially the whole time this phase has been
            # active. Kept proportional to DYNAMICS_COEF below (half),
            # same relationship as before.
            dynamics_weight = 0.0075
        else:
            lyap_weight = LYAPUNOV_COEF
            barrier_weight = BARRIER_COEF
            dynamics_weight = DYNAMICS_COEF

        # alpha follows a fixed schedule (see ALPHA_START/ALPHA_END/
        # ALPHA_DECAY_EPISODES below), not a learned parameter --
        # its value at this episode is known in advance, with no
        # dependence on gradient noise or how fast anything else
        # happens to be converging.
        decay_frac = min(ep, ALPHA_DECAY_EPISODES) / ALPHA_DECAY_EPISODES
        scheduled_alpha = ALPHA_START * (ALPHA_END / ALPHA_START) ** decay_frac

        if return_var_tracker is not None:
            return_var_tracker.update(returns.detach())
            return_scale = return_var_tracker.var + 1e-8
        else:
            return_scale = 1.0

        n = states.shape[0]
        indices = np.arange(n)
        accum = {}
        n_minibatches = 0
        last_loss = 0.0
        stop_early = False
        first_mb_kl = None
        epoch_kl = []

        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            for start_idx in range(0, n, MINIBATCH_SIZE):
                mb = torch.as_tensor(
                    indices[start_idx:start_idx + MINIBATCH_SIZE],
                    dtype=torch.long, device=device,
                )

                mb_states = states[mb]
                mb_adv = adv[mb]
                mb_returns = returns[mb]

                mb_soc = mb_states[:, -1, SOC_OBS_INDEX]
                mb_soc_next = next_states[mb][:, -1, SOC_OBS_INDEX]

                V_true = F.relu(SOC_TARGET - mb_soc) / SOC_TARGET
                V_true_next = F.relu(SOC_TARGET - mb_soc_next) / SOC_TARGET
                delta_V_true = V_true_next - V_true

                (lp, entropy, values, lyapunov, barrier, latent,
                 predicted_latent, mean_std, mean_raw_log_std,
                 raw_log_std_reg, raw_mean) = model.evaluate_actions(mb_states, actions[mb])

                V_next, actual_latent = model.evaluate_next(next_states[mb])
                V_now = lyapunov
                delta_V = V_next - V_now

                log_ratio = lp - oldlp[mb]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((torch.abs(ratio - 1.0) > CLIP_EPS).float().mean())
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - CLIP_EPS,
                    1.0 + CLIP_EPS,
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                outside_ball = (V_true > LYAPUNOV_BALL).float()
                n_outside = outside_ball.sum().clamp(min=1.0)

                lyapunov_penalty = (
                    F.relu(delta_V + LYAPUNOV_ALPHA * V_now + LYAPUNOV_MARGIN)
                    * outside_ball
                ).sum() / n_outside

                with torch.no_grad():
                    stability_slack = (
                        delta_V_true + LYAPUNOV_ALPHA * V_true + LYAPUNOV_MARGIN
                    )
                    viol = ((stability_slack > 0).float() * outside_ball).sum()
                    lyap_violation_rate = viol / n_outside
                    lyap_mean_dV = (delta_V_true * outside_ball).sum() / n_outside
                    lyap_worst_slack = torch.where(
                        outside_ball > 0, stability_slack,
                        torch.full_like(stability_slack, -1e9)
                    ).max()
                    lyap_frac_outside = outside_ball.mean()
                    lyap_in_ball_rate = 1.0 - lyap_frac_outside

                    lyap_v_rmse = torch.sqrt(
                        F.mse_loss(V_now, V_true) + 1e-12
                    )

                lyapunov_anchor_loss = F.mse_loss(V_now, V_true)

                lyapunov_collapse_penalty = (
                    lyapunov_anchor_loss
                    + LYAPUNOV_SPREAD_COEF * F.relu(
                        LYAPUNOV_V_SPREAD_FLOOR - V_now.std()
                    )
                )

                value_loss_raw = F.mse_loss(values, mb_returns.detach())
                value_loss = value_loss_raw / return_scale

                dynamics_loss = F.mse_loss(
                    predicted_latent,
                    actual_latent.detach(),
                )

                battery_loss = torch.relu(BATTERY_MARGIN - barrier[:, 0]).mean()
                boundary_loss = torch.relu(BOUNDARY_MARGIN - barrier[:, 1]).mean()
                vegetation_loss = torch.relu(VEGETATION_MARGIN - barrier[:, 2]).mean()
                velocity_loss = torch.relu(VELOCITY_MARGIN - barrier[:, 3]).mean()
                communication_loss = torch.relu(COMM_MARGIN - barrier[:, 4]).mean()

                mb_x = mb_states[:, -1, X_OBS_INDEX]
                mb_y = mb_states[:, -1, Y_OBS_INDEX]
                dx_c = mb_x * (GRID_DIM - 1) - BOUNDARY_CENTER
                dy_c = mb_y * (GRID_DIM - 1) - BOUNDARY_CENTER
                radial = torch.sqrt(dx_c * dx_c + dy_c * dy_c + 1e-8)
                boundary_target = (1.0 - radial / BOUNDARY_RADIUS).clamp(-1.0, 1.0)

                barrier_anchor = (
                    F.mse_loss(barrier[:, 0], mb_soc)
                    + F.mse_loss(barrier[:, 1], boundary_target)
                )

                barrier_loss = (1.0 * battery_loss + 0.5 * boundary_loss + 0.25 * vegetation_loss
                                + 0.25 * velocity_loss + 0.75 * communication_loss
                                + BARRIER_ANCHOR_COEF * barrier_anchor)

                saturation_loss = raw_mean.pow(2).mean()

                current_std = mean_std.item()
                if current_std < SAFETY_STD_FLOOR:
                    alpha = max(scheduled_alpha, SAFETY_ALPHA_BOOST)
                else:
                    alpha = scheduled_alpha

                loss = (policy_loss
                        + VALUE_COEF * value_loss
                        - alpha * entropy.mean()
                        + RAW_LOG_STD_REG_COEF * raw_log_std_reg
                        + MEAN_SATURATION_COEF * saturation_loss
                        + lyap_weight * lyapunov_penalty
                        + lyap_weight * LYAPUNOV_COLLAPSE_COEF * lyapunov_collapse_penalty
                        + dynamics_weight * dynamics_loss
                        + barrier_weight * barrier_loss)

                opt.zero_grad()
                loss.backward()

                control_norm = torch.nn.utils.clip_grad_norm_(control_params, 1.0)
                if auxiliary_param_list:
                    aux_norm = torch.nn.utils.clip_grad_norm_(auxiliary_param_list, 1.0)
                else:
                    aux_norm = torch.zeros((), device=control_norm.device)
                opt.step()

                with torch.no_grad():
                    mean_abs_action = torch.tanh(raw_mean).abs().mean()

                batch_metrics = {
                    "mean_V": float(V_now.mean().item()),
                    "std_V": float(V_now.std().item()),
                    "std_barrier": float(barrier[:, :2].std(dim=0).mean().item()),
                    "lyap_violation_rate": float(lyap_violation_rate.item()),
                    "lyap_in_ball_rate": float(lyap_in_ball_rate.item()),
                    "lyap_v_rmse": float(lyap_v_rmse.item()),
                    "grad_norm_control": float(control_norm.item()),
                    "grad_norm_aux": float(aux_norm.item()),
                    "lyap_mean_dV": float(lyap_mean_dV.item()),
                    "lyap_worst_slack": float(lyap_worst_slack.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss_raw.item()),
                    "lyap_penalty": float(lyapunov_penalty.item()),
                    "dynamics_loss": float(dynamics_loss.item()),
                    "barrier_loss": float(barrier_loss.item()),
                    "approx_kl": float(approx_kl.item()),
                    "clip_fraction": float(clip_fraction.item()),
                    "mean_std": float(mean_std.item()),
                    "mean_raw_log_std": float(mean_raw_log_std.item()),
                    "mean_abs_action": float(mean_abs_action.item()),
                    "alpha": float(alpha),
                }
                for k, v in batch_metrics.items():
                    accum[k] = accum.get(k, 0.0) + v
                n_minibatches += 1
                last_loss = float(loss.item())

                if first_mb_kl is None:
                    first_mb_kl = abs(float(approx_kl.item()))
                    if first_mb_kl > 1e-4:
                        print(
                            f"  [warn] ep {ep}: first-minibatch approx_kl = "
                            f"{first_mb_kl:.6f}, expected ~0. The policy is not "
                            f"reproducible between rollout and update (dropout, "
                            f"BatchNorm, or another nondeterministic layer), or "
                            f"the action distribution is saturated."
                        )

                epoch_kl.append(float(approx_kl.item()))

                if float(approx_kl.item()) > KL_EMERGENCY_MULT * TARGET_KL:
                    stop_early = True
                    break

            if epoch_kl:
                mean_epoch_kl = sum(epoch_kl) / len(epoch_kl)
                epoch_kl = []
                if mean_epoch_kl > TARGET_KL:
                    stop_early = True

            if stop_early:
                break

        avg = {k: v / max(n_minibatches, 1) for k, v in accum.items()}

        print(
            f"Policy {avg['policy_loss']:.4f} | "
            f"Value {avg['value_loss']:.4f} | "
            f"Lyap {avg['lyap_penalty']:.4f} | "
            f"Dynamics {avg['dynamics_loss']:.4f} | "
            f"Barrier {avg['barrier_loss']:.4f} | "
            f"KL {avg['approx_kl']:.5f} | "
            f"CF {avg['clip_fraction']:.4f} | "
            f"Std {avg['mean_std']:.4f} | "
            f"RawLogStd {avg['mean_raw_log_std']:.4f} | "
            f"|a| {avg['mean_abs_action']:.4f} | "
            f"Alpha {avg['alpha']:.4f} | "
            f"MB {n_minibatches}/{PPO_EPOCHS * math.ceil(n / MINIBATCH_SIZE)}"
        )
        if metrics_writer is not None:
            row = {"episode": ep, "minibatches": n_minibatches,
                   "minibatches_possible": PPO_EPOCHS * math.ceil(n / MINIBATCH_SIZE),
                   "first_mb_kl": first_mb_kl if first_mb_kl is not None else 0.0}
            row.update(avg)
            metrics_writer.writerow(row)

        diag_lyap = avg["lyap_penalty"]
        diag_barrier = avg["barrier_loss"]
        diag_std = avg["mean_std"]
        diag_abs_action = avg["mean_abs_action"]
        diag_mean_V = avg["mean_V"]
        diag_std_V = avg["std_V"]
        diag_std_barrier = avg["std_barrier"]
        diag_lyap_violation = avg["lyap_violation_rate"]
        diag_lyap_mean_dV = avg["lyap_mean_dV"]
        diag_lyap_in_ball = avg["lyap_in_ball_rate"]
        loss_value = last_loss
    else:
        decay_frac = min(ep, ALPHA_DECAY_EPISODES) / ALPHA_DECAY_EPISODES
        alpha = ALPHA_START * (ALPHA_END / ALPHA_START) ** decay_frac

        if return_var_tracker is not None:
            return_var_tracker.update(returns.detach())
            return_scale = return_var_tracker.var + 1e-8
        else:
            return_scale = 1.0

        n = states.shape[0]
        indices = np.arange(n)
        accum = {}
        n_minibatches = 0
        last_loss = 0.0
        stop_early = False
        epoch_kl = []

        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            for start_idx in range(0, n, MINIBATCH_SIZE):
                mb = torch.as_tensor(
                    indices[start_idx:start_idx + MINIBATCH_SIZE],
                    dtype=torch.long, device=device,
                )

                (lp, entropy, values, mean_std, mean_raw_log_std,
                 raw_log_std_reg, raw_mean) = model.evaluate_actions(
                    states[mb], actions[mb])
                log_ratio = lp - oldlp[mb]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((torch.abs(ratio - 1.0) > CLIP_EPS).float().mean())
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio,
                                    1.0 - CLIP_EPS,
                                    1.0 + CLIP_EPS) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss_raw = F.mse_loss(values, returns[mb].detach())
                value_loss = value_loss_raw / return_scale

                saturation_loss = raw_mean.pow(2).mean()

                loss = (policy_loss
                        + VALUE_COEF * value_loss
                        - alpha * entropy.mean()
                        + RAW_LOG_STD_REG_COEF * raw_log_std_reg
                        + MEAN_SATURATION_COEF * saturation_loss)

                opt.zero_grad()
                loss.backward()
                control_norm = torch.nn.utils.clip_grad_norm_(control_params, 1.0)
                opt.step()
                last_loss = float(loss.item())

                with torch.no_grad():
                    mb_std = mean_std
                    mb_absa = torch.tanh(raw_mean).abs().mean()

                batch_metrics = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss_raw.item()),
                    "approx_kl": float(approx_kl.item()),
                    "clip_fraction": float(clip_fraction.item()),
                    "mean_std": float(mb_std.item()),
                    "mean_raw_log_std": float(mean_raw_log_std.item()),
                    "mean_abs_action": float(mb_absa.item()),
                    "grad_norm_control": float(control_norm.item()),
                    "grad_norm_aux": 0.0,
                    "alpha": float(alpha),
                }
                for k, v in batch_metrics.items():
                    accum[k] = accum.get(k, 0.0) + v
                n_minibatches += 1

                epoch_kl.append(float(approx_kl.item()))
                if float(approx_kl.item()) > KL_EMERGENCY_MULT * TARGET_KL:
                    stop_early = True
                    break

            if epoch_kl:
                if sum(epoch_kl) / len(epoch_kl) > TARGET_KL:
                    stop_early = True
                epoch_kl = []

            if stop_early:
                break

        avg = {k: v / max(n_minibatches, 1) for k, v in accum.items()}
        print(
            f"Policy {avg.get('policy_loss', 0):.4f} | "
            f"Value {avg.get('value_loss', 0):.4f} | "
            f"KL {avg.get('approx_kl', 0):.5f} | "
            f"CF {avg.get('clip_fraction', 0):.4f} | "
            f"Std {avg.get('mean_std', 0):.4f} | "
            f"|a| {avg.get('mean_abs_action', 0):.4f} | "
            f"Alpha {avg.get('alpha', 0):.4f} | "
            f"MB {n_minibatches}/{PPO_EPOCHS * math.ceil(n / MINIBATCH_SIZE)}"
        )
        if metrics_writer is not None:
            row = {"episode": ep, "minibatches": n_minibatches,
                   "minibatches_possible": PPO_EPOCHS * math.ceil(n / MINIBATCH_SIZE),
                   "first_mb_kl": 0.0}
            row.update(avg)
            metrics_writer.writerow(row)

        diag_lyap = None
        diag_barrier = None
        diag_std = avg.get("mean_std")
        diag_abs_action = avg.get("mean_abs_action")
        diag_mean_V = None
        diag_std_V = None
        diag_std_barrier = None
        diag_lyap_violation = None
        diag_lyap_mean_dV = None
        diag_lyap_in_ball = None
        loss_value = last_loss

    return (loss_value, diag_lyap, diag_barrier, diag_std,
            diag_abs_action, diag_mean_V, diag_std_V, diag_std_barrier,
            diag_lyap_violation, diag_lyap_mean_dV, diag_lyap_in_ball)

def run():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=sim_env("test",20,MAX_STEPS_PER_EPISODE); env.set_view_dist(VIEW_DISTANCE)

    # 1. Initialization Logic
    if POLICY_TYPE == "transformer":
        if TRANSFORMER_VARIANT == "lyapunov":
            if TRANSFORMER_INIT == "chaotic":
                from chaotic_lyupnov_transformer import ChebyshevLyapunovTransformerActorCritic
                model = ChebyshevLyapunovTransformerActorCritic(
                    view_dist=VIEW_DISTANCE,
                    scalar_dim=SCALAR_DIM,
                    sequence_length=SEQUENCE_LENGTH,
                ).to(device)
            else:
                model = LyapunovTransformerActorCritic(
                    view_dist=VIEW_DISTANCE,
                    scalar_dim=SCALAR_DIM,
                    sequence_length=SEQUENCE_LENGTH,
                ).to(device)
            actor_params = list(model.actor.parameters())

            log_std_params = [model.log_std_param]

            critic_params = list(model.critic.parameters())

            transformer_params = (
                    list(model.input_projection.parameters()) +
                    list(model.encoder.parameters()) +
                    list(model.attention_pool.parameters()) +
                    [model.position_embedding]
            )

            auxiliary_params = (
                    list(model.energy_encoder.parameters()) +
                    list(model.lyapunov.parameters()) +
                    list(model.barrier.parameters()) +
                    list(model.dynamics.parameters())
            )

            control_params = transformer_params + actor_params + critic_params + log_std_params
            auxiliary_param_list = auxiliary_params

            opt = optim.AdamW([{"params": transformer_params, "lr": 1e-3, "weight_decay":1e-5},
                               {"params": actor_params,       "lr": 3e-4, "weight_decay":1e-5},
                               {"params": critic_params,      "lr": 3e-4, "weight_decay":1e-5},
                               {"params": auxiliary_params,   "lr": 2e-4, "weight_decay":1e-5},
                               # log_std_param is now a fixed, non-
                               # state-dependent parameter (see the
                               # model file), not a Linear layer
                               # reading latent. This removes the
                               # encoder-coupling problem that caused
                               # persistent oscillation through every
                               # previous attempt (a plain 1e-3 LR,
                               # then 3e-4, then detaching latent's
                               # gradient -- a full run confirmed
                               # detach alone didn't resolve it,
                               # since it only cut the backward path
                               # and log_std_head still read a
                               # constantly-moving forward-pass
                               # target). With no latent dependency
                               # left at all, the entropy bonus vs.
                               # hinge regularizer tug-of-war should
                               # now behave as the clean two-force
                               # system the math always predicted.
                               # Kept at the same conservative 3e-4 as
                               # a starting point for this genuinely
                               # different dynamic -- no evidence yet
                               # either way on whether it needs
                               # adjustment now that the moving-target
                               # problem is gone.
                               {"params": log_std_params,     "lr": 3e-4, "weight_decay":1e-5},],
                              eps=1e-5,)
        elif TRANSFORMER_VARIANT == "chaotic":
            from chebyshev_transformer import ChebyshevTransformer
            model = ChebyshevTransformer(VIEW_DISTANCE).to(device)
            opt = optim.Adam(model.parameters(), lr=LR)
            control_params = list(model.parameters())
            auxiliary_param_list = []
        else:
            from transformer import TransformerActorCritic
            model = TransformerActorCritic(
                VIEW_DISTANCE, scalar_dim=SCALAR_DIM,
                sequence_length=SEQUENCE_LENGTH).to(device)
            opt = optim.AdamW([
                {"params": (list(model.input_projection.parameters())
                            + list(model.encoder.parameters())
                            + list(model.attention_pool.parameters())
                            + [model.position_embedding]),
                 "lr": 1e-3, "weight_decay": 1e-5},
                {"params": list(model.actor.parameters()),
                 "lr": 3e-4, "weight_decay": 1e-5},
                {"params": list(model.critic.parameters()),
                 "lr": 3e-4, "weight_decay": 1e-5},
                {"params": [model.log_std_param],
                 "lr": 3e-4, "weight_decay": 1e-5},
            ])
            control_params = list(model.parameters())
            auxiliary_param_list = []
    else:
        from pso_policy import PSOPolicy
        # Assuming observation dimension is flattened patch size + scalars
        # Calculate based on your obs function (e.g., 20x20 patch + 7 scalars)
        input_dim = (VIEW_DISTANCE * 2 + 1) ** 2 + SCALAR_DIM
        model = PSOPolicy(input_dim=input_dim).to(device)
        opt = None  # PSO does not use torch.optim

    ###############################################################
    # Timing instrumentation
    ###############################################################
    total_inference_time = 0.0   # policy forward passes only
    total_inference_steps = 0    # counted per env step, never per episode
    total_rollout_time = 0.0     # whole step loop: inference + env + reward
    total_update_time = 0.0      # update() only
    total_update_calls = 0
    total_checkpoint_time = 0.0  # torch.save() only
    total_episode_time = 0.0     # end-to-end per episode
    run_start = time.perf_counter()
    epfile=open(OUT/"episode_metrics.csv","w",newline="")
    epw=csv.DictWriter(epfile,fieldnames=["episode","steps","final_battery","total_reward",
                                          "total_directional_reward","total_battery_reward",
                                          "total_movement_penalty","loss",
                                          "episode_time","rollout_time","inference_time",
                                          "env_time","update_time",
                                          "peak_solar_w","mean_solar_w",
                                          "start_battery","min_battery",
                                          "idle_mAh","motion_mAh","path_m","turn_integral",
                                          "net_displacement_m"])
    epw.writeheader()
    rollouts=[]

    # Same numbers as the training print line, written to CSV so they
    # don't have to be read back out of console/log output by hand.
    metrics_file = open(OUT/"training_metrics.csv","w",newline="")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=[
        "episode","minibatches","minibatches_possible","first_mb_kl",
        "policy_loss","value_loss","mean_V","std_V","std_barrier",
        "lyap_violation_rate","lyap_in_ball_rate","lyap_mean_dV","lyap_worst_slack",
        "lyap_v_rmse","grad_norm_control","grad_norm_aux",
        "lyap_penalty","dynamics_loss",
        "barrier_loss","approx_kl","clip_fraction","mean_std","mean_raw_log_std",
        "mean_abs_action","alpha",
    ])
    metrics_writer.writeheader()

    # Same numbers as the [Convergence check] print line.
    convergence_file = open(OUT/"convergence_checks.csv", "w", newline="")
    convergence_writer = csv.DictWriter(convergence_file, fieldnames=[
        "episode","avg_reward","best_avg_reward","reward_above_threshold",
        "avg_lyap","avg_barrier","avg_std","avg_abs_action","avg_mean_V","avg_std_V","avg_std_barrier","avg_lyap_violation","avg_lyap_mean_dV","avg_lyap_in_ball","lyap_stable",
        "lyap_collapsed","barrier_collapsed","exploration_cleared",
        "reward_trend_per_window","reward_trend_t","plateau_streak","plateaued",
        "stable","checks_since_best",
    ])
    convergence_writer.writeheader()

    # Convergence monitoring / checkpointing state
    ckpt_dir = OUT / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    reward_history = deque(maxlen=REWARD_WINDOW)
    lyap_history = deque(maxlen=STABILITY_WINDOW)
    barrier_history = deque(maxlen=STABILITY_WINDOW)
    std_history = deque(maxlen=STABILITY_WINDOW)
    abs_action_history = deque(maxlen=STABILITY_WINDOW)
    mean_V_history = deque(maxlen=STABILITY_WINDOW)
    std_V_history = deque(maxlen=STABILITY_WINDOW)
    std_barrier_history = deque(maxlen=STABILITY_WINDOW)
    lyap_violation_history = deque(maxlen=STABILITY_WINDOW)
    lyap_mean_dV_history = deque(maxlen=STABILITY_WINDOW)
    lyap_in_ball_history = deque(maxlen=STABILITY_WINDOW)
    plateau_history = deque(maxlen=PLATEAU_WINDOW)
    plateau_streak = 0
    degrade_streak = 0
    return_var_tracker = RunningVariance() if POLICY_TYPE == "transformer" else None
    best_avg_reward = -float("inf")
    checks_since_best = 0
    converged = False

    for ep in range(1,TOTAL_EPISODES+1):
        ep_start = time.perf_counter()
        ep_inference_time = 0.0
        ep_update_time = 0.0
        env.place_devices(); env.reset(); x,y,yaw=env.ch.get_position()
        ep_start_x, ep_start_y = x, y
        h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH);total=0;
        total_directional=0.0; total_battery_reward=0.0; total_movement_penalty=0.0
        r={"states":[],"next_states":[],"actions":[],"logps":[],"values":[],"rewards":[],"lyapunov":[], "barrier":[]};
        previous_action = None; smoothness = 0.0
        solar_samples = []
        min_battery = 100.0
        ep_idle_mAh = 0.0
        ep_motion_mAh = 0.0
        ep_path_m = 0.0
        ep_turn = 0.0
        start_battery = env.ch.get_battery()
        rollout_start = time.perf_counter()
        for step in range(MAX_STEPS_PER_EPISODE):
            s=seq_tensor(h,device)

            start = time.perf_counter()
            # 2. Action Selection Logic
            if POLICY_TYPE == "transformer":
                model.eval()
                with torch.no_grad():
                    if TRANSFORMER_VARIANT == "lyapunov":
                        (a, raw_a, lp, v, lyapunov, barrier, latent, predicted_next) = model.act(s)
                        current_action = a[0].detach().cpu().numpy()
                        stored_action = raw_a
                        if previous_action is None:
                            smoothness = 0.0
                        else:
                            smoothness = float(np.sum((current_action - previous_action) ** 2))
                        r["lyapunov"].append(lyapunov.cpu().numpy()); r["barrier"].append(barrier.cpu().numpy())
                    else:
                        a, raw_a_b, lp, v = model.act(s)
                        lyapunov = None
                        barrier = None
                        current_action = a[0].detach().cpu().numpy()
                        stored_action = raw_a_b
                        if previous_action is None:
                            smoothness = 0.0
                        else:
                            smoothness = float(
                                np.sum((current_action - previous_action) ** 2))
            else:
                # Set the current particle for the forward pass
                model.current_particle = (ep - 1) % model.swarm_size
                with torch.no_grad():
                    a = model(s)
                    lp, v = torch.tensor(0.0), torch.tensor(0.0)  # Dummy values
                    current_action = a[0].detach().cpu().numpy()

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            total_inference_time += elapsed
            ep_inference_time += elapsed
            total_inference_steps += 1

            # normalized action -> local, physically scaled target; no global-coordinate clipping mismatch
            dx,dy=a[0].cpu().numpy()*MAX_MOVE_PER_STEP
            tx=float(np.clip(x+dx,0,env.dim-1)); ty=float(np.clip(y+dy,0,env.dim-1))

            b_batt = env.ch.get_battery()

            tel, nxt= env.step_simulation(step,tx,ty)

            x_new, y_new, yaw_new = env.ch.get_position()
            sol = solarposition.get_solarposition(env.times[min(step, len(env.times) - 1)],
                                                  env.lat_center + y_new * env.stp, env.long_center + x_new * env.stp)
            after = env.get_obfuscation(x_new, y_new, min(step, len(env.times) - 1), sol.azimuth.iloc[0],
                                         sol.apparent_zenith.iloc[0]).flatten()
            aft_batt = env.ch.get_battery()
            solar_samples.append(float(env.ch.solar_potential))
            min_battery = min(min_battery, aft_batt)
            ep_idle_mAh += float(env.ch.step_idle_mAh)
            ep_motion_mAh += float(env.ch.step_motion_mAh)
            ep_path_m += float(env.ch.step_path_m)
            ep_turn += float(env.ch.step_turn_integral)

            rew_total, rew_directional, rew_battery, rew_movement = reward_fn(
                after, tel, aft_batt-b_batt, current_action)
            rew = rew_total - ACTION_SMOOTHNESS*smoothness

            total+=rew; previous_action=current_action
            total_directional+=rew_directional; total_battery_reward+=rew_battery; total_movement_penalty+=rew_movement
            r["states"].append(np.asarray(h)); r["actions"].append(stored_action[0].cpu().numpy())
            r["logps"].append(lp.item()); r["values"].append(v.item()); r["rewards"].append(rew)
            x,y,yaw=env.ch.get_position(); h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            r["next_states"].append(np.asarray(h))
            if aft_batt<=0: break

        rollout_elapsed = time.perf_counter() - rollout_start
        total_rollout_time += rollout_elapsed

        # GAE needs a bootstrap value for whatever comes after the last
        # recorded step. If the battery genuinely died (aft_batt<=0),
        # there truly is no future reward -- 0 is correct. If the loop
        # only ended because MAX_STEPS_PER_EPISODE was reached, the
        # environment did not actually terminate; bootstrapping with 0
        # would tell the value function "no more reward is possible
        # here" when that isn't true. Use the critic's own estimate of
        # the real next state in that case instead.
        if POLICY_TYPE == "transformer":
            if aft_batt <= 0:
                bootstrap_value = 0.0
            else:
                model.eval()
                with torch.no_grad():
                    dist_out = model.distribution(seq_tensor(h, device))
                    bootstrap_value_t = dist_out[1]
                bootstrap_value = bootstrap_value_t.item()
            r["bootstrap_value"] = bootstrap_value

        rollouts.append(r); loss=""
        reward_history.append(total)
        plateau_history.append(total)

        # 3. Training Update Logic
        if POLICY_TYPE == "transformer":
            if ep % UPDATE_EVERY_EPISODES == 0:
                update_start = time.perf_counter()
                (loss, diag_lyap, diag_barrier, diag_std,
                 diag_abs_action, diag_mean_V, diag_std_V,
                 diag_std_barrier, diag_lyap_violation,
                 diag_lyap_mean_dV, diag_lyap_in_ball) = update(
                    model, opt, rollouts, device, ep, metrics_writer, return_var_tracker,
                    control_params, auxiliary_param_list)
                ep_update_time = time.perf_counter() - update_start
                total_update_time += ep_update_time
                total_update_calls += 1
                metrics_file.flush()
                rollouts = []

                if diag_lyap is not None:
                    lyap_history.append(diag_lyap)
                    barrier_history.append(diag_barrier)
                    std_history.append(diag_std)
                    abs_action_history.append(diag_abs_action)
                    mean_V_history.append(diag_mean_V)
                    std_V_history.append(diag_std_V)
                    std_barrier_history.append(diag_std_barrier)
                    lyap_violation_history.append(diag_lyap_violation)
                    lyap_mean_dV_history.append(diag_lyap_mean_dV)
                    lyap_in_ball_history.append(diag_lyap_in_ball)

                    ready = (
                        ep >= 300
                        and len(lyap_history) == STABILITY_WINDOW
                        and len(reward_history) == REWARD_WINDOW
                    )
                    if ready:
                        avg_lyap = sum(lyap_history) / len(lyap_history)
                        avg_barrier = sum(barrier_history) / len(barrier_history)
                        avg_reward = sum(reward_history) / len(reward_history)
                        avg_std = sum(std_history) / len(std_history)
                        avg_abs_action = sum(abs_action_history) / len(abs_action_history)
                        avg_mean_V = sum(mean_V_history) / len(mean_V_history)
                        avg_std_V = sum(std_V_history) / len(std_V_history)
                        avg_std_barrier = sum(std_barrier_history) / len(std_barrier_history)
                        avg_lyap_violation = sum(lyap_violation_history) / len(lyap_violation_history)
                        avg_lyap_mean_dV = sum(lyap_mean_dV_history) / len(lyap_mean_dV_history)
                        avg_lyap_in_ball = sum(lyap_in_ball_history) / len(lyap_in_ball_history)
                        exploration_cleared = (
                            avg_std <= STD_CLEARED_THRESHOLD
                            and avg_abs_action <= ACTION_SATURATION_THRESHOLD
                        )
                        reward_above_threshold = avg_reward >= REWARD_CONVERGENCE_THRESHOLD

                        lyap_collapsed = (
                            avg_std_V < LYAPUNOV_COLLAPSE_STD
                            or abs(avg_lyap - LYAPUNOV_MARGIN) < LYAPUNOV_COLLAPSE_BAND
                        )
                        barrier_collapsed = avg_std_barrier < BARRIER_COLLAPSE_STD
                        collapsed = lyap_collapsed or barrier_collapsed

                        lyap_stable = (
                            avg_lyap_in_ball >= LYAPUNOV_IN_BALL_SUFFICIENT
                            or (
                                avg_lyap_violation <= LYAPUNOV_MAX_VIOLATION_RATE
                                and avg_lyap_mean_dV < 0.0
                            )
                        )

                        stable = (
                            lyap_stable
                            and avg_barrier <= BARRIER_STABLE_THRESHOLD
                            and not collapsed
                            and exploration_cleared
                            and reward_above_threshold
                        )

                        trend_slope = float("nan")
                        trend_t = float("nan")
                        plateau_now = False
                        if len(plateau_history) == PLATEAU_WINDOW:
                            ys = np.asarray(plateau_history, dtype=float)
                            n_pts = len(ys)
                            xs = np.arange(n_pts, dtype=float)
                            x_c = xs - xs.mean()
                            s_xx = float((x_c ** 2).sum())
                            if s_xx > 0 and n_pts > 2:
                                trend_slope = float((x_c * (ys - ys.mean())).sum() / s_xx)
                                resid = ys - (ys.mean() + trend_slope * x_c)
                                se = math.sqrt(
                                    float((resid ** 2).sum()) / (n_pts - 2) / s_xx
                                ) + 1e-12
                                trend_t = trend_slope / se
                                plateau_now = trend_t < PLATEAU_T_CRIT

                        plateau_streak = plateau_streak + 1 if plateau_now else 0
                        plateaued = plateau_streak >= PLATEAU_CONSECUTIVE

                        if avg_reward > best_avg_reward:
                            best_avg_reward = avg_reward
                            ckpt_start = time.perf_counter()
                            torch.save(model.state_dict(), ckpt_dir / "best.pt")
                            total_checkpoint_time += time.perf_counter() - ckpt_start

                        checks_since_best = 0 if avg_reward >= best_avg_reward else checks_since_best + 1

                        print(
                            f"[Convergence check] ep {ep} | "
                            f"avg_reward(last {REWARD_WINDOW}) {avg_reward:.2f} "
                            f"(best {best_avg_reward:.2f}, "
                            f"above_threshold={reward_above_threshold}) | "
                            f"avg_lyap(last {STABILITY_WINDOW}) {avg_lyap:.4f} "
                            f"(V={avg_mean_V:.3f}+-{avg_std_V:.3f}, "
                            f"viol={100*avg_lyap_violation:.1f}%, "
                            f"dV={avg_lyap_mean_dV:+.6f}, "
                            f"inball={100*avg_lyap_in_ball:.0f}%, "
                            f"collapsed={collapsed}) | "
                            f"avg_barrier(last {STABILITY_WINDOW}) {avg_barrier:.4f} | "
                            f"avg_std(last {STABILITY_WINDOW}) {avg_std:.4f} | "
                            f"avg|a|(last {STABILITY_WINDOW}) {avg_abs_action:.4f} "
                            f"(cleared={exploration_cleared}) | "
                            f"stable={stable} | "
                            f"trend {trend_slope * (PLATEAU_WINDOW - 1):+.2f}/window "
                            f"t={trend_t:+.2f} "
                            f"(plateau_streak={plateau_streak}/{PLATEAU_CONSECUTIVE}) | "
                            f"checks_since_best={checks_since_best}"
                        )
                        convergence_writer.writerow({
                            "episode": ep,
                            "avg_reward": avg_reward,
                            "best_avg_reward": best_avg_reward,
                            "reward_above_threshold": reward_above_threshold,
                            "avg_lyap": avg_lyap,
                            "avg_barrier": avg_barrier,
                            "avg_std": avg_std,
                            "avg_abs_action": avg_abs_action,
                            "avg_mean_V": avg_mean_V,
                            "avg_std_V": avg_std_V,
                            "avg_std_barrier": avg_std_barrier,
                            "avg_lyap_violation": avg_lyap_violation,
                            "avg_lyap_mean_dV": avg_lyap_mean_dV,
                            "avg_lyap_in_ball": avg_lyap_in_ball,
                            "lyap_stable": lyap_stable,
                            "lyap_collapsed": lyap_collapsed,
                            "barrier_collapsed": barrier_collapsed,
                            "exploration_cleared": exploration_cleared,
                            "reward_trend_per_window": trend_slope * (PLATEAU_WINDOW - 1),
                            "reward_trend_t": trend_t,
                            "plateau_streak": plateau_streak,
                            "plateaued": plateaued,
                            "stable": stable,
                            "checks_since_best": checks_since_best,
                        })
                        convergence_file.flush()

                        # auxiliary heads are doing. best.pt is already
                        if best_avg_reward > 0:
                            regression = (best_avg_reward - avg_reward) / abs(best_avg_reward)
                        else:
                            regression = 0.0
                        degrading = regression > DEGRADATION_FRAC
                        degrade_streak = degrade_streak + 1 if degrading else 0

                        if degrade_streak >= DEGRADATION_PATIENCE:
                            print(
                                f"\n[STOP] avg_reward {avg_reward:.2f} is "
                                f"{100 * regression:.1f}% below the best "
                                f"({best_avg_reward:.2f}) for "
                                f"{degrade_streak} consecutive checks. "
                                f"Halting; best.pt holds the peak policy."
                            )
                            converged = True
                            break

                        if stable and plateaued:
                            print(
                                f"\n[Convergence] Lyapunov/barrier stable and reward "
                                f"plateaued at episode {ep} -- "
                                f"avg_reward={avg_reward:.2f}, avg_lyap={avg_lyap:.4f}, "
                                f"avg_barrier={avg_barrier:.4f}."
                            )
                            ckpt_start = time.perf_counter()
                            torch.save(model.state_dict(), ckpt_dir / "converged.pt")
                            total_checkpoint_time += time.perf_counter() - ckpt_start
                            converged = True
        else:
            # PSO Update: Evaluate current particle and update swarm
            model.evaluate_particle(model.current_particle, total)
            if ep % model.swarm_size == 0:
                model.update_swarm()
            loss = "N/A (PSO)"

        if ep % CHECKPOINT_EVERY == 0:
            ckpt_start = time.perf_counter()
            torch.save(model.state_dict(), ckpt_dir / f"episode_{ep}.pt")
            total_checkpoint_time += time.perf_counter() - ckpt_start

        ep_elapsed = time.perf_counter() - ep_start
        total_episode_time += ep_elapsed

        ep_env_time = rollout_elapsed - ep_inference_time

        steps_taken = len(r["rewards"]); log_status(ep, TOTAL_EPISODES, steps_taken, total, aft_batt, loss)
        epw.writerow(dict(episode=ep,steps=len(r["rewards"]),final_battery=aft_batt,total_reward=total,
                          total_directional_reward=total_directional,total_battery_reward=total_battery_reward,
                          total_movement_penalty=total_movement_penalty,loss=loss,
                          episode_time=ep_elapsed, rollout_time=rollout_elapsed,
                          inference_time=ep_inference_time, env_time=ep_env_time,
                          update_time=ep_update_time,
                          peak_solar_w=max(solar_samples) if solar_samples else 0.0,
                          mean_solar_w=(sum(solar_samples)/len(solar_samples)) if solar_samples else 0.0,
                          start_battery=start_battery,
                          min_battery=min_battery,
                          idle_mAh=ep_idle_mAh, motion_mAh=ep_motion_mAh,
                          path_m=ep_path_m, turn_integral=ep_turn,
                          net_displacement_m=float(math.hypot(
                              x - ep_start_x, y - ep_start_y)))); epfile.flush()

        if converged and AUTO_STOP_ON_CONVERGENCE:
            print(f"[Convergence] Stopping training early at episode {ep}/{TOTAL_EPISODES}.")
            break

    ###############################################################
    # Validation phase
    #
    # per-episode summary is written alongside it.
    ###############################################################
    with open(OUT/"final_evaluation_steps.csv","w",newline="") as f:
        fields=["episode","step","x_before","y_before","target_x","target_y","x_after","y_after","battery_before",
                "battery_after","battery_delta","reward","directional_reward","battery_reward",
                "movement_penalty","action_dx_norm","action_dy_norm",
                "solar_w","exposure_score",
                "idle_mAh","motion_mAh","path_m","turn_integral"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()

        final_inference_time = 0.0
        final_inference_steps = 0
        final_eval_start = time.perf_counter()
        validation_rows = []

        for val_ep in range(1, VALIDATION_EPISODES + 1):
          env.place_devices(); env.reset()
          x,y,yaw=env.ch.get_position()
          h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH)
          val_start_batt = env.ch.get_battery()
          val_start_x, val_start_y = x, y
          val_total = 0.0; val_directional = 0.0; val_battery = 0.0; val_movement = 0.0
          val_idle = 0.0; val_motion = 0.0; val_path = 0.0; val_turn = 0.0
          val_min_batt = 100.0; val_solar = []; val_amag = []; val_steps = 0
          for step in range(MAX_STEPS_PER_EPISODE):
            start = time.perf_counter()
            if POLICY_TYPE == "transformer":
                model.eval()
                with torch.no_grad():
                    if TRANSFORMER_VARIANT == "lyapunov":
                        (a, raw_a, lp, v, lyapunov, barrier, latent, predicted_next) = model.fast_act(seq_tensor(h,device))
                    else:
                        a, _raw_a_b, lp, v = model.act(seq_tensor(h,device),True)
            else:
                # Set the current particle for the forward pass
                model.current_particle = (TOTAL_EPISODES - 1) % model.swarm_size
                with torch.no_grad():
                    a = model(seq_tensor(h,device))
                    lp, v = torch.tensor(0.0), torch.tensor(0.0)  # Dummy values

            elapsed = time.perf_counter() - start
            total_inference_time += elapsed
            total_inference_steps += 1
            final_inference_time += elapsed
            final_inference_steps += 1

            dx,dy=a[0].cpu().numpy()*MAX_MOVE_PER_STEP; tx=float(np.clip(x+dx,0,env.dim-1)); ty=float(np.clip(y+dy,0,env.dim-1))

            b_batt = env.ch.get_battery()

            tel,_=env.step_simulation(step,tx,ty)

            nx, ny, nyaw = env.ch.get_position()
            sol = solarposition.get_solarposition(env.times[min(step, len(env.times) - 1)],
                                                  env.lat_center + ny * env.stp, env.long_center + nx * env.stp)
            aft = env.get_obfuscation(nx, ny, min(step, len(env.times) - 1), sol.azimuth.iloc[0],
                                        sol.apparent_zenith.iloc[0]).flatten()
            aft_batt = env.ch.get_battery()

            rew, rew_directional, rew_battery, rew_movement = reward_fn(
                aft, tel, aft_batt-b_batt, a)

            w.writerow(dict(step=step,x_before=x,y_before=y,target_x=tx,target_y=ty,x_after=nx,y_after=ny,
                            battery_before=b_batt,battery_after=aft_batt,battery_delta=aft_batt-b_batt,reward=rew,
                            directional_reward=rew_directional,battery_reward=rew_battery,
                            movement_penalty=rew_movement,
                            action_dx_norm=a[0,0].item(),action_dy_norm=a[0,1].item(),
                            solar_w=float(env.ch.solar_potential),
                            exposure_score=float(np.dot(GAUSSIAN_KERNEL, 1.0 - aft)),
                            idle_mAh=float(env.ch.step_idle_mAh),
                            motion_mAh=float(env.ch.step_motion_mAh),
                            path_m=float(env.ch.step_path_m),
                            turn_integral=float(env.ch.step_turn_integral),
                            episode=val_ep))

            val_total += rew; val_directional += rew_directional
            val_battery += rew_battery; val_movement += rew_movement
            val_idle += float(env.ch.step_idle_mAh)
            val_motion += float(env.ch.step_motion_mAh)
            val_path += float(env.ch.step_path_m)
            val_turn += float(env.ch.step_turn_integral)
            val_solar.append(float(env.ch.solar_potential))
            val_amag.append(float(math.hypot(a[0,0].item(), a[0,1].item())))
            val_min_batt = min(val_min_batt, aft_batt)
            val_steps = step + 1

            x,y,yaw=nx,ny,nyaw; h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            if aft_batt<=0: break

          net_disp = float(math.hypot(x - val_start_x, y - val_start_y))
          validation_rows.append({
              "episode": val_ep,
              "steps": val_steps,
              "survived": val_steps >= MAX_STEPS_PER_EPISODE and aft_batt > 0,
              "start_battery": val_start_batt,
              "final_battery": aft_batt,
              "min_battery": val_min_batt,
              "total_reward": val_total,
              "directional_reward": val_directional,
              "battery_reward": val_battery,
              "movement_penalty": val_movement,
              "idle_mAh": val_idle,
              "motion_mAh": val_motion,
              "path_m": val_path,
              "turn_integral": val_turn,
              "net_displacement_m": net_disp,
              "tortuosity": val_path / max(net_disp, 1e-6),
              "mean_solar_w": float(np.mean(val_solar)) if val_solar else 0.0,
              "mean_abs_action": float(np.mean(val_amag)) if val_amag else 0.0,
          })
          print(f"  [validation {val_ep}/{VALIDATION_EPISODES}] steps {val_steps:3d} | "
                f"battery {val_start_batt:5.1f}% -> {aft_batt:5.1f}% (min {val_min_batt:5.1f}%) | "
                f"reward {val_total:8.2f} | |a| {validation_rows[-1]['mean_abs_action']:.3f} | "
                f"path {val_path:7.0f} m")

        final_eval_time = time.perf_counter() - final_eval_start
    epfile.close()
    metrics_file.close()
    convergence_file.close()
    # ADD THIS SECTION:
    import pandas as pd

    val_df = pd.DataFrame(validation_rows)
    val_df.to_csv(OUT / "validation_summary.csv", index=False)

    def _stat(col):
        s = val_df[col]
        return s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0, s.min(), s.max()

    print("\n" + "=" * 78)
    print(f"VALIDATION COMPLETE -- {len(val_df)} independent deterministic episodes")
    print("=" * 78)
    print(f"{'metric':<22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    for col, label in [
        ("total_reward",      "reward"),
        ("final_battery",     "final battery %"),
        ("min_battery",       "min battery %"),
        ("steps",             "steps"),
        ("mean_solar_w",      "solar W"),
        ("mean_abs_action",   "|action|"),
        ("path_m",            "path m"),
        ("tortuosity",        "tortuosity"),
        ("motion_mAh",        "motion mAh"),
        ("idle_mAh",          "idle mAh"),
    ]:
        m, sd, lo, hi = _stat(col)
        print(f"{label:<22} {m:10.2f} {sd:10.2f} {lo:10.2f} {hi:10.2f}")

    survived = int(val_df["survived"].sum())
    print()
    print(f"survived full {MAX_STEPS_PER_EPISODE} steps : {survived}/{len(val_df)}")
    print(f"ended below 20% battery         : {int((val_df.final_battery < 20).sum())}/{len(val_df)}")
    print(f"ended above 70% battery         : {int((val_df.final_battery > 70).sum())}/{len(val_df)}")

    if len(val_df) > 1:
        m, sd, _, _ = _stat("total_reward")
        se = sd / math.sqrt(len(val_df))
        print(f"\nreward 95% CI: {m:.2f} +- {1.96 * se:.2f}  ({m - 1.96 * se:.2f} to {m + 1.96 * se:.2f})")

    df_eval = pd.read_csv(OUT / "final_evaluation_steps.csv")
    total_steps = len(df_eval)
    print(f"\ntotal step rows written: {total_steps}")

    run_time = time.perf_counter() - run_start
    training_env_time = total_rollout_time - (total_inference_time - final_inference_time)
    other_time = run_time - total_rollout_time - total_update_time \
                 - total_checkpoint_time - final_eval_time

    def pct(x):
        return 100.0 * x / run_time if run_time > 0 else 0.0

    print("\n" + "=" * 46)
    print("WALL CLOCK BREAKDOWN")
    print("=" * 46)
    print(f"Total run time        : {run_time:10.2f} s")
    print(f"  Rollout (training)  : {total_rollout_time:10.2f} s  ({pct(total_rollout_time):5.1f}%)")
    print(f"    - policy inference: {total_inference_time - final_inference_time:10.2f} s  "
          f"({pct(total_inference_time - final_inference_time):5.1f}%)")
    print(f"    - env + reward    : {training_env_time:10.2f} s  ({pct(training_env_time):5.1f}%)")
    print(f"  PPO update()        : {total_update_time:10.2f} s  ({pct(total_update_time):5.1f}%)")
    print(f"  Checkpoint saves    : {total_checkpoint_time:10.2f} s  ({pct(total_checkpoint_time):5.1f}%)")
    print(f"  Final evaluation    : {final_eval_time:10.2f} s  ({pct(final_eval_time):5.1f}%)")
    print(f"  Other / logging     : {other_time:10.2f} s  ({pct(other_time):5.1f}%)")

    print("\n" + "=" * 46)
    print("PER-CALL AVERAGES")
    print("=" * 46)
    if total_inference_steps:
        print(f"Policy inference      : {total_inference_steps:8d} calls | "
              f"{1000 * total_inference_time / total_inference_steps:8.3f} ms/step")
    if final_inference_steps:
        print(f"  final eval only     : {final_inference_steps:8d} calls | "
              f"{1000 * final_inference_time / final_inference_steps:8.3f} ms/step")
    if total_update_calls:
        print(f"PPO update()          : {total_update_calls:8d} calls | "
              f"{1000 * total_update_time / total_update_calls:8.3f} ms/update")
    if TOTAL_EPISODES:
        print(f"Episode               : {TOTAL_EPISODES:8d} eps   | "
              f"{total_episode_time / TOTAL_EPISODES:8.3f} s/episode")

    if run_time > 0:
        print(f"\nupdate() share of wall clock: {pct(total_update_time):.1f}%")
if __name__=="__main__": run()