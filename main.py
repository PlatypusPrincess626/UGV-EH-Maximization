import csv, math, random
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

# The alternative policy variants are optional -- importing them
# eagerly means a missing module breaks the lyapunov path too, even
# though it never touches them. Imported on demand in run() instead.
TransformerActorCritic = None
ChebyshevTransformer = None
ChebyshevLyapunovTransformerActorCritic = None
PSOPolicy = None

# ============================================================
# Set POLICY_TYPE = "transformer" or "pso"
POLICY_TYPE = "transformer"
if POLICY_TYPE == "transformer":
    # Set TRANSFORMER_VARIANT = "normal" or "chaotic" or "lyapunov"
    TRANSFORMER_VARIANT = "lyapunov"
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
BARRIER_COEF = 0.20; ACTION_SMOOTHNESS = 0.01; LYAPUNOV_MARGIN = 0.01
# (LATENT_COEF was defined here and never referenced anywhere -- removed.)
BATTERY_MARGIN = 0.10; BOUNDARY_MARGIN = 0.10; VEGETATION_MARGIN = 0.05
VELOCITY_MARGIN = 0.05; COMM_MARGIN = 0.10; CLIP_EPS =0.20; LYAPUNOV_ALPHA = 0.02
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
# Previously update() took exactly ONE full-batch gradient step per
# rollout: 1000 episodes / UPDATE_EVERY_EPISODES=2 = 500 optimizer
# steps for the entire run. A 2-layer transformer over a 1688-dim
# input cannot learn anything in 500 steps.
#
# It also meant the clipped surrogate was decorative: with a single
# step per batch, `lp` and `oldlp` come from identical parameters,
# so ratio == 1 by construction and PPO degenerates to vanilla A2C.
#
# Epochs * minibatches turns the same collected data into ~24
# optimizer steps per update (1440 samples / 256 per minibatch = 6,
# times 4 epochs), i.e. ~12000 for the run, and makes the ratio
# meaningful from the second minibatch onward.
###############################################################
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
# Early-stop the epoch loop once the updated policy has moved too
# far from the behaviour policy. Standard PPO practice; matters more
# here than usual because the rollout is only two episodes.
TARGET_KL = 0.015

# Penalty on the pre-squash mean, applied to keep tanh out of
# saturation. Nothing else in this system constrains |raw_mean|.
#
# In the previous run the deterministic policy sat at
# action_dx_norm = -0.99 for all 720 evaluation steps, i.e.
# raw_mean ~ -2.7. At that point tanh's output is flat: the
# environment cannot distinguish raw_mean=-2.7 from -3.5, so the
# reward gradient vanishes, while the log-prob term keeps pushing
# the mean further out. Exploration was effectively dead --
# with the logged pre-squash Std of 0.47, sampled actions spanned
# tanh([-3.6, -1.8]) = [-0.998, -0.947], about 1% of the action
# range -- even though that Std reads as perfectly healthy and
# even satisfied the STD_CLEARED_THRESHOLD convergence gate.
#
# This is the same term SAC uses for the same reason.
# Raised 1e-3 -> 1e-2. At 1e-3 the penalty was too weak to hold the
# mean off the boundary: mean_abs_action reached exactly 1.0 and
# stayed there for the back half of the run, i.e. the deterministic
# policy was a constant, fully saturated action. Saturation also
# destabilizes the importance ratio (see the dropout note in the
# model file), so this term protects the KL early-stop as well as
# exploration.
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
ALPHA_END = 0.05
ALPHA_DECAY_EPISODES = 300
# Reactive safety net (see the entropy check in update()): if a
# batch's mean entropy drops below this floor, alpha jumps to at
# least SAFETY_ALPHA_BOOST regardless of the schedule. Floor sits
# between the achievable range's true minimum (~-1.16, full
# collapse) and the original learned-alpha target (~1.338, the
# smooth bound's midpoint) -- low enough to only trigger on a real
# problem, not on ordinary convergence toward a reasonable
# exploitation level.
SAFETY_ENTROPY_FLOOR = 0.3
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
CONVERGENCE_PATIENCE = 10       # consecutive stability checks with no new best reward
LYAPUNOV_STABLE_THRESHOLD = 2 * LYAPUNOV_MARGIN
BARRIER_STABLE_THRESHOLD = 0.05
CHECKPOINT_EVERY = 100          # periodic safety-net checkpoint, regardless of performance
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
# Companion to STD_CLEARED_THRESHOLD, on the OTHER half of the
# exploration question. STD_CLEARED_THRESHOLD looks at the
# pre-squash sigma; this looks at mean |tanh(raw_mean)|. The
# previous run passed the sigma test throughout while executing a
# constant, fully-saturated action, because a healthy-looking
# sigma of 0.47 around a raw_mean of -2.7 still spans only ~1% of
# the action range. A policy whose mean action magnitude averages
# above this is pinned against the action bounds, not converged.
ACTION_SATURATION_THRESHOLD = 0.97

# Format as YYYY-MM-DD_HH-MM-SS
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT=Path("rl_csv_"+timestamp); OUT.mkdir(exist_ok=True)

size = 2 * VIEW_DISTANCE + 1
y, x = np.mgrid[-VIEW_DISTANCE:VIEW_DISTANCE+1,
                -VIEW_DISTANCE:VIEW_DISTANCE+1]
sigma = 6.0
kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
kernel /= kernel.sum()
GAUSSIAN_KERNEL = kernel.flatten()

def log_status(ep, total_episodes, steps, avg_reward, final_batt, loss, is_eval=False):
    prefix = "[EVALUATION]" if is_eval else f"[Episode {ep}/{total_episodes}]"
    loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else loss
    print(f"{prefix} Steps: {steps:3} | Reward: {avg_reward:7.2f} | Battery: {final_batt:6.2f}% | Loss: {loss_str}")

# Number of scalar (non-patch) features appended by obs(). Must match
# the `scalar_dim` passed to the model -- keep them derived from this
# constant rather than duplicating the literal in two files.
SCALAR_DIM = 8

def obs(env, x, y, yaw, step):
    sol=solarposition.get_solarposition(env.times[min(step,len(env.times)-1)], env.lat_center+y*env.stp, env.long_center+x*env.stp)
    patch=env.get_obfuscation(x,y,min(step,len(env.times)-1),sol.azimuth.iloc[0],sol.apparent_zenith.iloc[0]).flatten()
    potential = 1.0 - patch

    # Solar elevation instead of raw zenith/90.
    #
    # zenith/90 was intended to normalize to [0,1] but exceeds 1
    # whenever the sun is below the horizon (zenith > 90, reaching
    # ~1.5 at the old run's 20:00 endpoint). That was one of only
    # seven non-patch features, out of range for the entire dark
    # portion of the episode.
    #
    # sun_usable is an explicit flag for whether get_obfuscation is
    # returning real geometry or its below-threshold short-circuit.
    # Without it the policy has to infer "there is no sun" from the
    # patch having gone uniformly zero (potential = 1 - 1), which is
    # indistinguishable from "fully shaded but the sun is up" and
    # gives it nothing to switch behaviour on.
    elevation = 90.0 - sol.apparent_zenith.iloc[0]
    elevation_norm = max(0.0, elevation) / 90.0
    sun_usable = 1.0 if elevation >= MIN_USABLE_ELEVATION else 0.0

    scalars=np.array([x/(env.dim-1),y/(env.dim-1),math.sin(yaw),math.cos(yaw),
                      env.ch.get_battery()/100, sol.azimuth.iloc[0]/360,
                      elevation_norm, sun_usable],np.float32)
    return np.concatenate([potential.astype(np.float32),scalars])

def seq_tensor(history, device):
    return torch.tensor(np.asarray(history),dtype=torch.float32,device=device).unsqueeze(0)

def reward_fn(after, telemetry, delta_batt):
    # Dense, scaled reward: energy gain, movement cost, survival, and boundary discouragement.
    #
    # directional_reward is now the ABSOLUTE, instantaneous exposure
    # score at the current position, not the difference between this
    # step's before/after scores. The differenced form telescopes to
    # exactly (score_final - score_initial) when summed over an
    # episode -- every intermediate term cancels, so it could never
    # reward *sustained* good positioning, only the net change from
    # start to end (confirmed directly: total_directional_reward was
    # ~0.1-0.2 per full 720-step episode, regardless of how the
    # episode was actually navigated). This is textbook potential-
    # based reward shaping (Ng, Harada & Russell 1999) -- valid, but
    # by construction it doesn't change what the optimal policy
    # achieves, only how fast training converges to it, and it was
    # the *only* positional signal here with no absolute term
    # alongside it.
    #
    # Coefficient recalibrated from 10 to 0.05: the old value was
    # tuned for a step-to-step DIFFERENCE (small, since two nearby
    # positions' scores are usually close). An ABSOLUTE score is
    # typically ~0.7-0.9, a much larger and always-positive quantity
    # -- left at 10, this would total on the order of several
    # thousand over a 720-step episode, dwarfing battery_reward
    # (~-53 total) and movement_penalty (~-22 total) by two orders of
    # magnitude. 0.05 targets a comparable order of magnitude to
    # those two instead (~0.05 * 0.8avg * 720steps =~ 29 total) so
    # positioning can actually compete for influence rather than
    # either vanishing (the old bug) or swamping everything else (the
    # naive fix). Worth checking total_directional_reward against
    # total_battery_reward/total_movement_penalty after the next run
    # and adjusting further if the balance still isn't right --
    # this is a reasoned starting point, not a precisely derived one.
    potential_after = 1.0 - after
    score_after = float(np.dot(GAUSSIAN_KERNEL, potential_after))

    directional_reward = 0.05 * score_after

    px, py, _ = telemetry["previous_position"]
    nx, ny, _ = telemetry["new_position"]

    distance = math.hypot(nx - px, ny - py)

    movement_penalty = 0.002 * distance

    battery_reward = 2.0 * delta_batt

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

        # Same order as everything else appended below.
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

    # Normalized once across the complete rollout batch, after
    # returns have been derived from the raw advantages.
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    return states, next_states, actions, oldlp, adv, returns


def update(model,opt,rollouts,device, ep, metrics_writer=None, return_var_tracker=None):
    # Dropout must be ACTIVE for the update (it is a regularizer on
    # the gradient step) and INACTIVE during rollout, where its only
    # effect is to make the recorded log-probs and values refer to a
    # policy that does not exist. See the model.eval() calls in run().
    model.train()

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
            #
            # Note this weighting matters much less now: the auxiliary
            # heads read a DETACHED latent (see encode_state in the
            # model file), so dynamics_loss no longer competes with
            # policy_loss for control of the shared encoder at all.
            # It only trains the dynamics head itself.
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

        # Return-scale tracker is updated once per rollout batch, not
        # once per minibatch -- otherwise the same data would be
        # counted PPO_EPOCHS * n_minibatches times and the running
        # count would race ahead of the actual sample count.
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

                (lp, entropy, values, lyapunov, barrier, latent,
                 predicted_latent, mean_std, mean_raw_log_std,
                 raw_log_std_reg, raw_mean) = model.evaluate_actions(mb_states, actions[mb])

                # Only ONE extra encoder pass, on next_states. V(s) and
                # the predicted next latent already came back above --
                # calling evaluate_transition here would re-encode
                # mb_states for identical results, and that waste would
                # be multiplied by epochs * minibatches.
                V_next, actual_latent = model.evaluate_next(next_states[mb])
                V_now = lyapunov
                delta_V = V_next - V_now

                ratio = torch.exp(lp - oldlp[mb])
                approx_kl = (oldlp[mb] - lp).mean()
                clip_fraction = ((torch.abs(ratio - 1.0) > CLIP_EPS).float().mean())
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - CLIP_EPS,
                    1.0 + CLIP_EPS,
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                lyapunov_penalty = F.relu(delta_V
                                          + LYAPUNOV_ALPHA * V_now
                                          + LYAPUNOV_MARGIN).mean()

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
                barrier_loss = (1.0 * battery_loss + 0.5 * boundary_loss + 0.25 * vegetation_loss
                                + 0.25 * velocity_loss + 0.75 * communication_loss)

                # Keeps tanh out of saturation -- see
                # MEAN_SATURATION_COEF. Without this the mean drifts
                # outward indefinitely, because past |tanh| ~ 0.99 the
                # environment cannot tell one raw_mean from another
                # and only the log-prob term still has an opinion.
                saturation_loss = raw_mean.pow(2).mean()

                # Safety net the pure schedule gives up: a learned alpha
                # would have noticed entropy collapsing too far and
                # raised itself back up on its own. A schedule has no
                # such awareness -- it decays regardless of what entropy
                # is actually doing. This reactive check restores that
                # protection without reintroducing a convergence-
                # dependent mechanism: it's a direct threshold, not a
                # gradient, so there's no "will it correct in time"
                # question.
                current_entropy = entropy.mean().item()
                if current_entropy < SAFETY_ENTROPY_FLOOR:
                    alpha = max(scheduled_alpha, SAFETY_ALPHA_BOOST)
                else:
                    alpha = scheduled_alpha

                loss = (policy_loss
                        + VALUE_COEF * value_loss
                        - alpha * entropy.mean()
                        + RAW_LOG_STD_REG_COEF * raw_log_std_reg
                        + MEAN_SATURATION_COEF * saturation_loss
                        + lyap_weight * lyapunov_penalty
                        + dynamics_weight * dynamics_loss
                        + barrier_weight * barrier_loss)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                # tanh(raw_mean) is the diagnostic that actually tracks
                # exploration. The pre-squash Std does not: at
                # raw_mean=-2.7 a Std of 0.47 still yields actions
                # spanning only ~1% of the action range.
                with torch.no_grad():
                    mean_abs_action = torch.tanh(raw_mean).abs().mean()

                batch_metrics = {
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

                # Self-check on the very first minibatch of the first
                # epoch: no optimizer step has touched the parameters
                # since the rollout, so lp and oldlp are computed from
                # IDENTICAL weights and approx_kl must be ~0. Anything
                # else means the forward pass is not reproducible
                # between rollout and update -- which previously came
                # from dropout being off in eval() and on in train(),
                # and made the KL early stop fire immediately, costing
                # the entire epoch/minibatch budget without any
                # visible error.
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

                # Standard PPO early stop. Matters more than usual here
                # because the rollout is only UPDATE_EVERY_EPISODES
                # episodes: without it, later epochs can push the policy
                # a long way from the behaviour policy that generated
                # the data, and the importance ratios stop being
                # trustworthy.
                if abs(float(approx_kl.item())) > TARGET_KL:
                    stop_early = True
                    break

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
        loss_value = last_loss
    else:
        n = states.shape[0]
        indices = np.arange(n)
        last_loss = 0.0
        stop_early = False

        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            for start_idx in range(0, n, MINIBATCH_SIZE):
                mb = torch.as_tensor(
                    indices[start_idx:start_idx + MINIBATCH_SIZE],
                    dtype=torch.long, device=device,
                )

                lp, entropy, values = model.evaluate_actions(states[mb], actions[mb])
                ratio = torch.exp(lp - oldlp[mb])
                approx_kl = (oldlp[mb] - lp).mean()
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio,
                                    1.0 - CLIP_EPS,
                                    1.0 + CLIP_EPS) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                loss = (policy_loss
                        + VALUE_COEF * F.mse_loss(values, returns[mb])
                        - ENTROPY_COEF * entropy.mean())

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                last_loss = float(loss.item())

                if abs(float(approx_kl.item())) > TARGET_KL:
                    stop_early = True
                    break

            if stop_early:
                break

        diag_lyap = None
        diag_barrier = None
        diag_std = None
        diag_abs_action = None
        loss_value = last_loss

    return loss_value, diag_lyap, diag_barrier, diag_std, diag_abs_action

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
                # scalar_dim passed explicitly: obs() now appends 8
                # scalars, not 7 (solar elevation replaced the
                # out-of-range zenith/90 term, and a sun-usable flag
                # was added). The model default is still 7, so leaving
                # this implicit would silently mis-slice the input.
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

            opt = optim.AdamW([{"params": transformer_params, "lr": 1e-3, "weight_decay":1e-5},
                               {"params": actor_params,       "lr": 3e-4, "weight_decay":1e-5},
                               {"params": critic_params,      "lr": 3e-4, "weight_decay":1e-5},
                               {"params": auxiliary_params,   "lr": 5e-5, "weight_decay":1e-5},
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
        else:
            from transformer import TransformerActorCritic
            model = TransformerActorCritic(VIEW_DISTANCE).to(device)
            opt = optim.Adam(model.parameters(), lr=LR)
    else:
        from pso_policy import PSOPolicy
        # Assuming observation dimension is flattened patch size + scalars
        # Calculate based on your obs function (e.g., 20x20 patch + 7 scalars)
        input_dim = (VIEW_DISTANCE * 2 + 1) ** 2 + SCALAR_DIM
        model = PSOPolicy(input_dim=input_dim).to(device)
        opt = None  # PSO does not use torch.optim

    ###############################################################
    # Timing instrumentation
    #
    # Previously a single (total_inference_time, total_inference_steps)
    # pair accumulated three different things at two different units:
    # the per-env-step policy forward pass, and -- via a timer opened
    # before the training block and closed after the checkpoint save --
    # the per-episode update() call and torch.save() as well, with the
    # step counter incremented once per env step AND once per episode.
    # The reported "Average inference ms/step" was therefore a mix of
    # inference, optimization and disk I/O divided by a count that
    # mixed episodes and steps.
    #
    # Each phase now gets its own accumulator and its own counter, in
    # consistent units, so the rollout-vs-update split can actually be
    # read off a run rather than guessed at.
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
                                          "env_time","update_time"])
    epw.writeheader()
    rollouts=[]

    # Same numbers as the training print line, written to CSV so they
    # don't have to be read back out of console/log output by hand.
    metrics_file = open(OUT/"training_metrics.csv","w",newline="")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=[
        "episode","minibatches","minibatches_possible","first_mb_kl",
        "policy_loss","value_loss","lyap_penalty","dynamics_loss",
        "barrier_loss","approx_kl","clip_fraction","mean_std","mean_raw_log_std",
        "mean_abs_action","alpha",
    ])
    metrics_writer.writeheader()

    # Same numbers as the [Convergence check] print line.
    convergence_file = open(OUT/"convergence_checks.csv", "w", newline="")
    convergence_writer = csv.DictWriter(convergence_file, fieldnames=[
        "episode","avg_reward","best_avg_reward","reward_above_threshold",
        "avg_lyap","avg_barrier","avg_std","avg_abs_action","exploration_cleared",
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
    return_var_tracker = RunningVariance() if TRANSFORMER_VARIANT == "lyapunov" else None
    best_avg_reward = -float("inf")
    checks_since_best = 0
    converged = False

    for ep in range(1,TOTAL_EPISODES+1):
        ep_start = time.perf_counter()
        ep_inference_time = 0.0
        ep_update_time = 0.0
        env.place_devices(); env.reset(); x,y,yaw=env.ch.get_position()
        h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH);total=0;
        total_directional=0.0; total_battery_reward=0.0; total_movement_penalty=0.0
        r={"states":[],"next_states":[],"actions":[],"logps":[],"values":[],"rewards":[],"lyapunov":[], "barrier":[]};
        previous_action = None; smoothness = 0.0
        rollout_start = time.perf_counter()
        for step in range(MAX_STEPS_PER_EPISODE):
            s=seq_tensor(h,device)

            start = time.perf_counter()
            # 2. Action Selection Logic
            if POLICY_TYPE == "transformer":
                # eval() disables dropout for action selection.
                #
                # Without it, TransformerEncoderLayer(dropout=0.1) is
                # active here AND again in evaluate_actions() with a
                # DIFFERENT mask, so the recorded log_prob and value
                # describe a policy that never existed. The previous
                # run proved this was firing: with one gradient step
                # per batch, lp and oldlp come from identical
                # parameters, so approx_kl and clip_fraction must be
                # exactly 0 -- they logged 0.083 / 0.458 at episode 2
                # and 0.006 / 0.053 at episode 1000. Every one of
                # those clipped samples was clipped by dropout noise,
                # and r["values"] (the GAE baseline) was corrupted the
                # same way.
                model.eval()
                with torch.no_grad():
                    if TRANSFORMER_VARIANT == "lyapunov":
                        (a, raw_a, lp, v, lyapunov, barrier, latent, predicted_next) = model.act(s)
                        current_action = a[0].detach().cpu().numpy()
                        if previous_action is None:
                            smoothness = 0.0
                        else:
                            # current_action is already normalized to
                            # (-1,1) by the tanh squash, so the extra
                            # division by MAX_MOVE_PER_STEP made this
                            # term ~1e-5 after ACTION_SMOOTHNESS --
                            # numerically dead. Compare normalized
                            # actions directly.
                            smoothness = float(np.sum((current_action - previous_action) ** 2))
                        r["lyapunov"].append(lyapunov.cpu().numpy()); r["barrier"].append(barrier.cpu().numpy())
                    else:
                        a, lp, v = model.act(s)
                        lyapunov = None
                        barrier = None
                        current_action = a[0].detach().cpu().numpy()
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

            rew_total, rew_directional, rew_battery, rew_movement = reward_fn(after, tel, aft_batt-b_batt)
            rew = rew_total - ACTION_SMOOTHNESS*smoothness
            total+=rew; previous_action=current_action
            total_directional+=rew_directional; total_battery_reward+=rew_battery; total_movement_penalty+=rew_movement
            r["states"].append(np.asarray(h)); r["actions"].append(raw_a[0].cpu().numpy())
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
        if POLICY_TYPE == "transformer" and TRANSFORMER_VARIANT == "lyapunov":
            if aft_batt <= 0:
                bootstrap_value = 0.0
            else:
                model.eval()
                with torch.no_grad():
                    (_, bootstrap_value_t, _, _, _, _, _) = model.distribution(seq_tensor(h, device))
                bootstrap_value = bootstrap_value_t.item()
            r["bootstrap_value"] = bootstrap_value

        rollouts.append(r); loss=""
        reward_history.append(total)

        # 3. Training Update Logic
        if POLICY_TYPE == "transformer":
            if ep % UPDATE_EVERY_EPISODES == 0:
                update_start = time.perf_counter()
                loss, diag_lyap, diag_barrier, diag_std, diag_abs_action = update(
                    model, opt, rollouts, device, ep, metrics_writer, return_var_tracker)
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
                        # avg_std alone was not a sufficient test. The
                        # previous run satisfied avg_std <= 1.0 for its
                        # entire length while the policy was in fact a
                        # constant, fully saturated action -- the
                        # pre-squash Std says nothing about spread in
                        # ACTION space once tanh is flat. Require the
                        # mean action to be off the boundary too.
                        exploration_cleared = (
                            avg_std <= STD_CLEARED_THRESHOLD
                            and avg_abs_action <= ACTION_SATURATION_THRESHOLD
                        )
                        reward_above_threshold = avg_reward >= REWARD_CONVERGENCE_THRESHOLD
                        stable = (
                            avg_lyap <= LYAPUNOV_STABLE_THRESHOLD
                            and avg_barrier <= BARRIER_STABLE_THRESHOLD
                            and exploration_cleared
                            and reward_above_threshold
                        )

                        if avg_reward > best_avg_reward:
                            best_avg_reward = avg_reward
                            checks_since_best = 0
                            ckpt_start = time.perf_counter()
                            torch.save(model.state_dict(), ckpt_dir / "best.pt")
                            total_checkpoint_time += time.perf_counter() - ckpt_start
                        else:
                            checks_since_best += 1

                        print(
                            f"[Convergence check] ep {ep} | "
                            f"avg_reward(last {REWARD_WINDOW}) {avg_reward:.2f} "
                            f"(best {best_avg_reward:.2f}, "
                            f"above_threshold={reward_above_threshold}) | "
                            f"avg_lyap(last {STABILITY_WINDOW}) {avg_lyap:.4f} | "
                            f"avg_barrier(last {STABILITY_WINDOW}) {avg_barrier:.4f} | "
                            f"avg_std(last {STABILITY_WINDOW}) {avg_std:.4f} | "
                            f"avg|a|(last {STABILITY_WINDOW}) {avg_abs_action:.4f} "
                            f"(cleared={exploration_cleared}) | "
                            f"stable={stable} | "
                            f"checks_since_best={checks_since_best}/{CONVERGENCE_PATIENCE}"
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
                            "exploration_cleared": exploration_cleared,
                            "stable": stable,
                            "checks_since_best": checks_since_best,
                        })
                        convergence_file.flush()

                        if stable and checks_since_best >= CONVERGENCE_PATIENCE:
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

        # env_time is rollout minus policy inference: environment
        # stepping, obs(), get_obfuscation and the reward computation.
        # This is the number that decides whether update() is worth
        # optimizing at all.
        ep_env_time = rollout_elapsed - ep_inference_time

        steps_taken = len(r["rewards"]); log_status(ep, TOTAL_EPISODES, steps_taken, total, aft_batt, loss)
        epw.writerow(dict(episode=ep,steps=len(r["rewards"]),final_battery=aft_batt,total_reward=total,
                          total_directional_reward=total_directional,total_battery_reward=total_battery_reward,
                          total_movement_penalty=total_movement_penalty,loss=loss,
                          episode_time=ep_elapsed, rollout_time=rollout_elapsed,
                          inference_time=ep_inference_time, env_time=ep_env_time,
                          update_time=ep_update_time)); epfile.flush()

        if converged and AUTO_STOP_ON_CONVERGENCE:
            print(f"[Convergence] Stopping training early at episode {ep}/{TOTAL_EPISODES}.")
            break

    # final deterministic evaluation, step-level telemetry CSV
    env.place_devices(); env.reset(); x,y,yaw=env.ch.get_position(); h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH)
    with open(OUT/"final_evaluation_steps.csv","w",newline="") as f:
        fields=["step","x_before","y_before","target_x","target_y","x_after","y_after","battery_before",
                "battery_after","battery_delta","reward","directional_reward","battery_reward",
                "movement_penalty","action_dx_norm","action_dy_norm"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()

        final_inference_time = 0.0
        final_inference_steps = 0
        final_eval_start = time.perf_counter()
        for step in range(MAX_STEPS_PER_EPISODE):
            start = time.perf_counter()
            if POLICY_TYPE == "transformer":
                # Deterministic evaluation must have dropout off, or
                # it is not deterministic.
                model.eval()
                with torch.no_grad():
                    if TRANSFORMER_VARIANT == "lyapunov":
                        (a, raw_a, lp, v, lyapunov, barrier, latent, predicted_next) = model.fast_act(seq_tensor(h,device))
                    else:
                        a, lp, v = model.act(seq_tensor(h,device),True)
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

            rew, rew_directional, rew_battery, rew_movement = reward_fn(aft, tel, aft_batt-b_batt)

            w.writerow(dict(step=step,x_before=x,y_before=y,target_x=tx,target_y=ty,x_after=nx,y_after=ny,
                            battery_before=b_batt,battery_after=aft_batt,battery_delta=aft_batt-b_batt,reward=rew,
                            directional_reward=rew_directional,battery_reward=rew_battery,
                            movement_penalty=rew_movement,
                            action_dx_norm=a[0,0].item(),action_dy_norm=a[0,1].item()))
            x,y,yaw=nx,ny,nyaw; h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            if aft_batt<=0: break
        final_eval_time = time.perf_counter() - final_eval_start
    epfile.close()
    metrics_file.close()
    convergence_file.close()
    # ADD THIS SECTION:
    import pandas as pd
    df_eval = pd.read_csv(OUT / "final_evaluation_steps.csv")
    total_steps = len(df_eval)

    print("\n" + "=" * 46)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 46)
    print(f"Total Steps Performed: {total_steps}")
    print(f"Final Battery Level: {df_eval['battery_after'].iloc[-1]:.2f}%")

    run_time = time.perf_counter() - run_start
    training_env_time = total_rollout_time - (total_inference_time - final_inference_time)
    other_time = run_time - total_rollout_time - total_update_time \
                 - total_checkpoint_time - final_eval_time

    def pct(x):
        return 100.0 * x / run_time if run_time > 0 else 0.0

    # Each line is one phase, in consistent units, and the phases sum
    # to the wall clock. The old single "inference" number mixed the
    # policy forward pass with update() and torch.save(), divided by a
    # count that mixed env steps with episodes.
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

    # The number that decides whether the PPO epoch/minibatch budget is
    # worth tuning. If update() is a small slice, raising PPO_EPOCHS is
    # nearly free; if it is a large slice, PPO_EPOCHS=3 with
    # MINIBATCH_SIZE=512 recovers most of the cost.
    if run_time > 0:
        print(f"\nupdate() share of wall clock: {pct(total_update_time):.1f}%")
if __name__=="__main__": run()