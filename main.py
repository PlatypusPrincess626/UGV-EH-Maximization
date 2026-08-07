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

# The cost family. All three use reward_fn_cost and differ ONLY in
# the critic construction, so every reward-scale constant, threshold
# and certification path below keys off membership in this set rather
# than the literal "cost".
#
#   cost         beta*softplus head, spectral-normalized  (reference)
#   cost_linear  unconstrained head, spectral-normalized  (isolates
#                                                          the head)
#   cost_plain   unconstrained head, no spectral norm     (isolates
#                                                          both)
#
# See ablation_transformer.py for what each one gives up.
COST_VARIANTS = ("cost", "cost_linear", "cost_plain")
IS_COST = TRANSFORMER_VARIANT in COST_VARIANTS


###############################################################
# Run configuration from the environment
#
#     LTAC_VARIANT=lyapunov LTAC_SEED=3 LTAC_EPISODES=400 python main.py
#
# LTAC_SEED seeds EVERY random source the run touches. Before this,
# nothing seeded numpy at all: terrain, foliage and start position
# came from OS entropy per process, so two runs were never comparable
# on the environments they saw. Meanwhile `random` WAS seeded, from
# EPISODE_DATE alone in sim_env.__init__, which made start SOC and
# device placement identical in every run ever launched. That split
# is why validation `start_battery` matched exactly across arms while
# terrain did not.
#
# With a shared LTAC_SEED both arms train and validate on identical
# environments, which is what makes a paired comparison valid. The
# same variable is read by environment.py.
#
# LTAC_EPISODES allows short paired sweeps: training-stability
# questions are answered by the learning phase, so 300-400 episodes
# is usually enough and makes multi-seed designs affordable.
###############################################################
RUN_SEED = int(os.environ.get("LTAC_SEED", "0"))

TOTAL_EPISODES = int(os.environ.get("LTAC_EPISODES", "1000"))
MAX_STEPS_PER_EPISODE=720; VIEW_DISTANCE=20

# Auxiliary-head curriculum, as fractions of the run so that short
# sweeps keep the same shape. At TOTAL_EPISODES=1000 these are 100 and
# 300, matching the previous hardcoded values.
CURRICULUM_WARMUP_EPISODES = max(1, int(0.10 * TOTAL_EPISODES))
CURRICULUM_FULL_EPISODES = max(2, int(0.30 * TOTAL_EPISODES))
# Seed every source, before anything draws from them. sim_env.__init__
# reseeds `random` from (EPISODE_DATE, RUN_SEED) later; that is
# deliberate and still deterministic.
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)
torch.manual_seed(RUN_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RUN_SEED)
# Determinism in cuDNN costs essentially nothing here -- the
# environment is ~98% of wall clock and the network is small.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEQUENCE_LENGTH=32; GAMMA=.99; GAE_LAMBDA=.95
# 2 -> 4 episodes per update (1440 -> 2880 samples).
#
# The KL early stop was discarding 26% of the gradient budget and
# worsening: 8890 of 12000 possible minibatches taken, with the
# over-target rate climbing 7% -> 18% -> 30% across the run.
#
# A larger batch cuts gradient noise by sqrt(2), so KL per step roughly
# halves and the stop stops binding. The total budget is unchanged
# (250 updates x 48 minibatches = 12000) but far more of it gets used.
# It also gives 12 minibatches per epoch instead of 6, so minibatches
# are less correlated, and better statistics for the advantage
# normalization and the return-variance tracker.
#
# The cost is negligible: update() measured 0.3% of wall clock (53.8 s
# of 21,134) against the environment's 95.5%.
#
# HALVED FROM 4. At TOTAL_EPISODES=400 the previous cadence gave 100
# optimizer updates for the whole run -- a transformer policy trained
# on 100 gradient steps. Doubling to 200 costs ~54 s on a six-hour
# run, because the batch was sized for a bottleneck that does not
# exist here.
#
# Nothing downstream is keyed to the update count: the LR decay and
# the entropy schedule are both functions of EPISODE, and
# STABILITY_WINDOW is already documented as tracking a fixed span of
# episodes rather than of calls. The batch halves from ~2880 to ~1440
# samples, which at MINIBATCH_SIZE=256 is still 5-6 minibatches per
# epoch.
UPDATE_EVERY_EPISODES=2
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
# For a cost MDP the Bellman equation gives
#     dV = -c(s) - (1 - gamma) * V(s')
# so (1 - gamma) IS the decay rate. Using anything else would certify
# against a bound the discount already contradicts.
COST_LYAPUNOV_ALPHA = 1.0 - GAMMA
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
# Expressed as a FRACTION of TOTAL_EPISODES so short sweeps keep the
# same schedule shape. At the default 1000 episodes this reproduces
# the previous fixed value of 300 exactly.
ALPHA_DECAY_EPISODES = max(1, int(0.30 * TOTAL_EPISODES))
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
# update() calls averaged for Lyapunov/barrier stability. Sized to
# cover a fixed span of EPISODES, not of calls, so it tracks
# UPDATE_EVERY_EPISODES: 20 calls x 2 episodes = the same 40-episode
# stretch that 10 calls x 4 episodes covered before.
STABILITY_WINDOW = 20
CONVERGENCE_PATIENCE = 10       # retained for logging only; see PLATEAU_SLOPE_FRAC
LYAPUNOV_STABLE_THRESHOLD = 2 * LYAPUNOV_MARGIN

###############################################################
# Ultimate boundedness
###############################################################
SOC_TARGET = 0.90
# The cost variant measures deficit QUADRATICALLY, so a ball of the
# same radius in V units means a much looser tolerance in SOC:
# V <= 0.05 is SOC >= 69.9% under the square, against 85.5% under the
# linear form. 0.0025 = 0.05^2 restores the intended meaning.
COST_LYAPUNOV_BALL = 0.0025
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
# Set per variant, because the two arms' rewards are on different
# scales and neither threshold means anything on the other.
#
# The old cost value (-100) was written against an assumed return of
# -50 to -110. Measured, the UNSCALED cost arm returned -624 (best
# episode) to -1354 (worst), typically ~-1000, so -100 could never
# have fired. Under COST_REWARD_SCALE = 1 - GAMMA those become -6.24
# to -13.5, typically ~-10. -7.0 is therefore a real bar: a 50-episode
# average has to beat the single luckiest untrained episode.
#
# Hardcoded rather than written as COST_REWARD_SCALE * something,
# because COST_REWARD_SCALE is defined further down with the rest of
# the cost-MDP block and this constant is read before it.
REWARD_CONVERGENCE_THRESHOLD = -7.0 if TRANSFORMER_VARIANT in COST_VARIANTS else -150.0

###############################################################
# Plateau test
###############################################################
###############################################################
# Learning-rate decay
#
# Cosine decay of the CONTROL parameter groups (trunk, actor, critic,
# log_std) from their initial values to LR_DECAY_FINAL x those values
# over TOTAL_EPISODES.
#
# The KL early stop tightens on its own as training progresses. KL
# scales roughly as (delta_mu / sigma)^2, and mean_std fell 0.175 ->
# 0.133 over the last run, so a FIXED parameter step produces 1.73x
# the KL by the end. Measured KL growth was 3.54x, so sigma shrinkage
# explains about half and genuinely larger steps the rest. A constant
# learning rate against a constant TARGET_KL therefore trips the stop
# more and more often -- 7% of updates early, 30% late.
#
# The AUXILIARY group is deliberately excluded. It has its own
# gradient clip, is under no KL constraint, and its certificate was
# still improving at the end of the run (lyap_v_rmse 0.247 -> 0.102),
# so decaying it would slow the one thing that had not converged.
LR_DECAY_FINAL = 0.30

PLATEAU_WINDOW = 200
PLATEAU_T_CRIT = 1.65      # one-sided 95%
PLATEAU_CONSECUTIVE = 10

###############################################################
# Degradation guard
#
# instead of running to 1000.
###############################################################
DEGRADATION_FRAC = 0.15
# Floor on the denominator of the regression ratio, so a
# best_avg_reward near zero cannot make it explode.
#
# Per-variant for the same reason as REWARD_CONVERGENCE_THRESHOLD: a
# floor of 10.0 against scaled cost returns of ~-10 would dominate
# the denominator at every check and mute the guard entirely. 0.1
# keeps it at the same ~1% of a typical return that 10.0 was on the
# reward arms.
DEGRADATION_SCALE_FLOOR = 0.1 if TRANSFORMER_VARIANT in COST_VARIANTS else 10.0
DEGRADATION_PATIENCE = 10
ACTION_SATURATION_THRESHOLD = 0.97

###############################################################
# Cost-MDP variant
#
# Every reward component becomes a non-negative COST that is zero at
# its ideal, so r = -c <= 0 and the value function is the negative
# cost-to-go. V_cost = -value is then a Lyapunov candidate by
# construction: non-negative, zero exactly when all three costs are
# zero forever (parked, in sun, at target charge), and decreasing by
# the Bellman equation.
#
# Why each component had to change rather than only the battery term:
# leaving directional_reward and PARK_COEF positive keeps the reward
# mixed-sign, which makes the value maximal at the goal and defeats
# the whole construction.
#
# Weights are set so the three costs contribute comparably at the
# scales actually observed:
#     shade   mean 0.228   (1 - exposure)
#     deficit mean 0.124   quadratic; squaring SHRINKS values on [0,1]
#     move    mean 0.033   distance / MAX_MOVE_PER_STEP
# The deficit weight is raised to compensate for the squaring, which
# roughly halved it relative to the linear form.
###############################################################
# Shade cost relative to the best exposure in view, rather than to
# perfect sun. See reward_fn_cost for the measurement that motivated
# it. False restores the absolute form used by every cost run up to
# and including the cost_plain ablation -- those runs are NOT
# comparable to relative-shade runs, since this changes the reward.
COST_SHADE_RELATIVE = True

# Shade weight. Left at 1.00 deliberately for this comparison run.
#
# Lowering it improves the critic's certificate sharply -- measured
# non-decrease rate for the true cost-to-go at n=1, on the code's own
# mask, with ABSOLUTE shade:
#
#     COST_SHADE_W    1.00    0.25    0.10    0.00
#     dV>=0 (n=1)     .122    .048    ~.01    .002
#
# But COST_SHADE_RELATIVE attacks the SAME defect -- the part of the
# shade cost no action can reduce -- and does it in a targeted way
# rather than by shrinking the whole term, so it keeps the gradient
# toward sun that solves the deficit -> SOC -> sun credit assignment.
# Stacking both on one run would confound them, and the numbers above
# were measured WITHOUT the relative form, so they do not predict the
# combination.
#
# Order of operations: run with relative shade at 1.00, read
# `crit dV>=0` off it, and drop the weight to 0.25 only if that
# number is still high. Going to 0.00 buys another factor of ~24 on
# the certificate but removes the sun-seeking signal entirely, which
# trades a real learning signal for a metric.
COST_SHADE_W = 1.00
COST_DEFICIT_W = 2.00
COST_MOVE_W = 0.50

# Global scale on the cost-MDP reward.
#
# WHY THIS EXISTS
#
# The critic's layers are spectral-normalized, so critic_body is
# 1-Lipschitz in the pooled latent. The encoder ends in a LayerNorm,
# which fixes every token's norm at sqrt(d_model) ~ 11.3, and the
# attention pool is a convex combination of tokens, so ||latent||
# <= ~11.3. The critic's entire reachable output range is therefore
# ~+-11 around whatever offset the (unconstrained) biases supply.
#
# The unscaled costs ran -1.39 per step. The critic is trained on a
# TD(lambda) return, whose scale is r_bar / (1 - gamma*lambda) =
# -1.39 / (1 - 0.99*0.95) = -23.4. Reaching -23.4 from a -Softplus
# head that starts near -0.7 requires ~23 units of pure bias travel,
# and AdamW moves a bias by ~lr = 3e-4 per optimizer step. At 400
# episodes / UPDATE_EVERY_EPISODES=2 = 200 updates x ~24 minibatches
# there are only 4800 steps in the whole run. Measured: value_loss
# opened at 416 -- exactly (23.4 - 0.7)^2 -- and had only reached
# ~200 by episode 144. The critic never left its initialization, so
# GAE degenerated to delta_t ~ r_t with no state-dependent baseline
# and the policy gradient was pure variance (Policy ~0.00x,
# KL ~0.003, |a| stuck near the 0.01 output gain).
#
# WHY 1 - GAMMA IS THE RIGHT SCALE
#
# 1 - gamma is exactly the factor that turns a discounted SUM into a
# discounted AVERAGE: the return of a constant per-step cost c is
# c / (1 - gamma), so scaling the cost by (1 - gamma) makes the
# return land at c, i.e. on the same scale as one step's cost.
# Returns move from ~-23 to ~-0.23, comfortably inside the critic's
# ~+-11 range and ~800 bias-steps away rather than 78,000.
#
# WHY IT IS SAFE FOR THE CERTIFICATE
#
# A positive scalar multiple of a Lyapunov function is a Lyapunov
# function: V >= 0, V = 0 at the goal, and dV < 0 are all invariant
# under k*V for k > 0. The decay rate is unchanged too -- the
# Bellman identity dV = -c(s) - (1-gamma)*V(s') scales uniformly on
# both sides, so COST_LYAPUNOV_ALPHA = 1 - GAMMA still holds exactly.
# Advantages are normalized to zero mean and unit variance in
# compute_batch, and value_loss is divided by the running return
# variance, so both the policy gradient and the value-loss weight
# are already scale-invariant. Nothing else needed rescaling.
COST_REWARD_SCALE = 1.0 - GAMMA

# Critic head shaping. V_cost = beta * softplus(g / beta), whose gain
# at value V is exactly 1 - exp(-V/beta) -- a function of the RATIO
# only. So a fixed beta does not hold the head's conditioning fixed:
# V_cost shrinks as the policy improves (1.08 -> 0.65 over cost seed
# 1's 400 episodes), and the gain drifts down with it.
#
# COST_BETA_GAIN_TARGET makes beta slide to hold that gain instead,
# via beta = V_q10 / -ln(1 - target), EMA'd in log space once per
# update. Set it to None for a fixed beta.
#
# 0.90 is deliberately not higher. Pushing toward 1.0 drives beta
# small, and beta is also the width of the region where the head is
# meaningfully curved -- shrink it too far and the head is a ReLU
# with a dead zone in all but name.
COST_BETA_INIT = 0.30
COST_BETA_GAIN_TARGET = 0.90

# Decay rate alpha: measured, not asserted.
#
# The condition dV + alpha*V + margin <= 0 inverts to
# alpha <= (-dV - margin)/V, so every sample carries the largest
# alpha IT supports. The q-th quantile of that is exactly the largest
# alpha at which the violation rate would have been q.
#
# In "adaptive" mode alpha tracks that quantile, EMA'd in log space
# once per update. The violation rate is then ~COST_ALPHA_QUANTILE BY
# CONSTRUCTION and carries no information -- the RESULT is alpha_hat
# itself. Report it that way: "the certified decay rate is alpha_hat,
# at a 5% violation rate", not "the violation rate is 5%". Choosing
# alpha to minimise violations and then reporting the low violation
# rate would be circular; reporting the rate you can certify is not.
#
# The measured alpha is calibrated on the ANALYTIC V_true -- a
# property of the physical system, not of any network -- and the same
# value is then applied to the merged critic's check. That keeps
# v_critic_violation free to move, so it still measures whether the
# critic reproduces the analytic certificate rather than being pinned
# to the target by construction. On the 3-seed sweep those two rates
# were 61% and 63%, and their agreement was the best evidence the
# merge holds; pinning both would have destroyed it.
#
# Set COST_ALPHA_MODE = "fixed" to restore the asserted 1 - GAMMA.
# Cost-to-go charged when the battery dies.
#
# THE FAILURE STATE IS ABSORBING, AND UNDER A COST MDP IT IS THE MOST
# EXPENSIVE STATE THERE IS -- not the cheapest.
#
# GAE bootstraps the last step of a terminated episode. Bootstrapping
# with 0 is right for the reward arms: dying forfeits future positive
# reward, so 0 is a penalty. Under r = -c it is exactly inverted. Every
# return is <= 0, so V = 0 is the BEST value in the state space, and a
# policy that drives the battery flat at step 100 books ~1/7th the
# accumulated cost of one that survives all 720. Suicide becomes
# optimal, in the advantage as well as the logged total.
#
# The fix is to charge what the agent would actually accrue. A dead
# robot sits at SOC 0 for the rest of the episode, so its per-step
# cost is the deficit term at maximum:
#
#     c_fail = COST_DEFICIT_W * ((SOC_TARGET - 0)/SOC_TARGET)^2
#            = COST_DEFICIT_W
#
# discounted over the steps that remain. Shade and move are left out
# deliberately -- they depend on where the corpse is, and the deficit
# term alone already dominates.
#
# Sizing check: c_fail * COST_REWARD_SCALE / (1 - GAMMA) = 2.0 against
# a measured healthy V_cost of ~0.65, so death is ~3x worse than the
# worst normal state. Finite-horizon rather than 1/(1-gamma) so that
# dying at step 719 is not charged the same as dying at step 10.
COST_DEATH_COST = COST_DEFICIT_W * COST_REWARD_SCALE


def death_cost_to_go(step, horizon=None):
    """
    DISCOUNTED cost of sitting dead from `step` to the horizon. This is
    what the critic represents, so it is the right bootstrap value.
    """
    horizon = MAX_STEPS_PER_EPISODE if horizon is None else horizon
    remaining = max(int(horizon) - int(step), 0)
    if remaining <= 0:
        return 0.0
    return COST_DEATH_COST * (1.0 - GAMMA ** remaining) / (1.0 - GAMMA)


def death_cost_logged(step, horizon=None):
    """
    UNDISCOUNTED cost of the same thing, for the logged episode total.

    The two differ and both are needed. The logged total is a plain sum
    of per-step costs, so charging it the discounted figure leaves
    dying attractive anyway: at gamma=0.99 the discounted charge caps
    at 2.00, while a surviving episode accumulates ~4.9 undiscounted.
    Dying at step 50 would book -2.34 against -4.90 -- still the better
    outcome, just less obviously.

    The critic keeps the discounted form because that is what a value
    function is; only the reporting side uses this one.
    """
    horizon = MAX_STEPS_PER_EPISODE if horizon is None else horizon
    return COST_DEATH_COST * max(int(horizon) - int(step), 0)


COST_ALPHA_MODE = "adaptive"
COST_ALPHA_QUANTILE = 0.05
COST_ALPHA_EMA = 0.05
COST_ALPHA_MIN = 0.0          # 0 is still a Lyapunov claim: V non-increasing
COST_ALPHA_MAX = 1.0 - GAMMA  # the original assertion, as a ceiling

# Mutable so update() can slide it; read through ALPHA_STATE["alpha"]
# everywhere rather than closing over the constant.
ALPHA_STATE = {"alpha": COST_LYAPUNOV_ALPHA}

# The smoothness penalty is applied OUTSIDE reward_fn_cost, so it has
# to be scaled alongside it or it silently becomes ~100x heavier
# relative to the costs it sits next to (it was ~0.06% of a step's
# reward before scaling; left alone it would be ~6%).
EFFECTIVE_ACTION_SMOOTHNESS = (
    ACTION_SMOOTHNESS * COST_REWARD_SCALE
    if TRANSFORMER_VARIANT in COST_VARIANTS
    else ACTION_SMOOTHNESS
)

# Elevation below which no realistically attainable position sustains
# the robot, so the descent condition cannot hold and is not claimed.
#
# Derived empirically: the 95th percentile of ACHIEVED panel power
# crosses the 15.92 W breakeven at ~18.5 deg. Note this is the
# attainable figure, not the unshaded one -- an earlier estimate of
# 20 deg assumed unshaded operation, and a linear extrapolation of
# power against sin(elevation) gave a nonsense 6.5 deg because the
# fit's intercept is not physical.
#
# At 18.5 deg the certified region covers 96% of the episode. The
# three ball escapes observed in the lyapunov run sit at 20.3-30.0
# deg, INSIDE this region, so they remain counted as the genuine
# policy failures they are.
CERTIFY_MIN_ELEVATION = 18.5


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
# Seed in the directory name: a paired sweep launches many runs that
# differ only by seed, and timestamps alone make them hard to match up.
OUT=Path(f"rl_csv_{_variant_tag}_s{RUN_SEED}_{timestamp}"); OUT.mkdir(exist_ok=True)

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
# elevation_norm is scalar 6 of 8 (see obs()).
ELEV_OBS_INDEX = -(SCALAR_DIM - 6)
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

def reward_fn_cost(after, telemetry, soc_after):
    """
    Pure cost. Every term is >= 0 and zero at its ideal, so the
    returned reward is <= 0 and the value function is the negative
    cost-to-go.

    Differences from reward_fn that matter:

      * LEVEL, not DELTA. reward_fn's battery term is
        BATTERY_COEF * delta_batt, whose episode sum telescopes to
        2*(final - start) -- verified exactly on all ten validation
        episodes. That is path-independent: two episodes ending at the
        same charge score identically whether one reached target at
        step 334 or never. The deficit cost integrates over time
        instead, so reaching target sooner and staying is rewarded.

      * The movement cost does NOT scale with exposure. reward_fn
        charges movement at (1 + 3*exposure), which made moving
        CHEAPEST exactly when energy was scarcest -- 60% of the midday
        rate in the evening, for the same energy per metre. Measured on
        the lyapunov run, 88% of all motion energy was spent in windows
        where no position could sustain the robot.

      * Quadratic, not linear or quartic. Deficit is normalized to
        [0,1] where higher powers shrink both value and gradient:
        quartic is weaker than quadratic everywhere below deficit
        0.707 (SOC 26.4%), which is the entire operating range.
    """
    exposure = float(np.dot(GAUSSIAN_KERNEL, 1.0 - after))

    px, py, _ = telemetry["previous_position"]
    nx, ny, _ = telemetry["new_position"]
    distance = math.hypot(nx - px, ny - py)

    if COST_SHADE_RELATIVE:
        # Shade cost RELATIVE to what is reachable right now, not
        # absolute.
        #
        # 1 - exposure has an exogenous floor: late in the episode the
        # sun is low and shadows lengthen, so no position recovers full
        # exposure. Measured on cost seed 1's validation episodes, the
        # BEST exposure any episode achieved in t in [600,700] was
        # 0.62-0.88, against 0.94-1.00 in t in [350,400].
        #
        # A cost that cannot reach zero makes a cost-to-go that cannot
        # reach zero, and V_cost = 0 at the goal is a Lyapunov
        # condition, not a nicety. It showed up directly: the
        # cost-to-go at t=380 sat at 11.63 with the shade term and 0.80
        # without it, and the non-decrease rate went 0.263 -> 0.139.
        # V_cost was not failing to fit; it was fitting a function that
        # does not converge.
        #
        # Subtracting the best exposure in view removes the floor
        # without removing the gradient: the cost is still zero only at
        # the sunniest reachable spot, at any sun height, and still
        # points toward sun everywhere else. relu() guards the case
        # where the kernel-weighted exposure edges above the patch max.
        #
        # potential.max() is an upper bound on the kernel-weighted
        # exposure attainable nearby -- the patch is already 1 - after,
        # so this costs one max() and no new hyperparameter.
        c_shade = max(float((1.0 - after).max()) - exposure, 0.0)
    else:
        c_shade = 1.0 - exposure
    c_deficit = (max(SOC_TARGET - soc_after, 0.0) / SOC_TARGET) ** 2
    c_move = distance / MAX_MOVE_PER_STEP

    total = -(COST_SHADE_W * c_shade
              + COST_DEFICIT_W * c_deficit
              + COST_MOVE_W * c_move)

    # COST_REWARD_SCALE applies to the total AND to every component,
    # so the CSV breakdown still sums to the logged reward. See the
    # constant's definition for why the critic needs it.
    k = COST_REWARD_SCALE

    # Signs match reward_fn's convention so the episode CSV columns
    # stay comparable in magnitude: directional and battery are
    # contributions to the reward, movement is reported positive.
    return (float(k * total),
            float(k * -COST_SHADE_W * c_shade),
            float(k * -COST_DEFICIT_W * c_deficit),
            float(k * COST_MOVE_W * c_move))


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

# Window for the ANALYTIC certificate, in steps.
#
# V_true is a function of SOC, which jitters step to step but rises
# monotonically on long timescales, so the single-step condition
# fails far more often than the trajectory warrants. Measured on the
# exact (critic-free) values from cost seed 1's validation episodes,
# the fraction of samples with dV >= 0:
#
#     n        1     10     50    100    150    200
#     V_true  .120   .096   .050   .010   .001   .000
#
# 150 keeps 570 of every 720 samples while driving the rate to ~0.1%.
# Larger n buys little and discards more of each episode's tail.
#
# THE CRITIC IS CERTIFIED AT n = 1, NOT HERE. The two candidates want
# opposite windows. Measured on the code's own mask (V_true > ball,
# elevation gate), non-decrease rate by window:
#
#     n                  1     10     50    150
#     V_true          .134   .104   .050   .001    improves with n
#     V_cost          .122   .135   .186   .303    degrades with n
#
# V_true jitters step to step but rises monotonically over hours, so
# it wants a long window. V_cost is a discounted sum whose long-gap
# differences vanish into fluctuation, so it wants a short one. At
# n=1 the two are comparable (.122 vs .134); certifying V_cost at
# n=150 was measuring the wrong thing.
#
# NOTE this does NOT transfer to V_cost, which moves the other way
# (.150 at n=1 rising to .360 at n=200) because a discounted
# cost-to-go converges to a nonzero steady state and long-gap
# differences vanish into fluctuation. The critic's check stays at
# one step and fixes its noise a different way -- see
# critic_certification().
CERT_NSTEP = 150

CERT_QUANTILES = (0.05, 0.25, 0.50)


def _masked_quantiles(x, mask, qs=CERT_QUANTILES):
    """Quantiles of x over the entries where mask > 0; nan if empty."""
    sel = x[mask > 0]
    if sel.numel() == 0:
        return [float("nan")] * len(qs)
    q = torch.quantile(sel.float(),
                       torch.tensor(qs, device=sel.device, dtype=torch.float32))
    return [float(v) for v in q]


def analytic_certification(soc, soc_next, elev_deg, valid=None,
                           nstep=1, alpha=None,
                           ball=COST_LYAPUNOV_BALL,
                           margin=LYAPUNOV_MARGIN):
    """
    The analytic Lyapunov certificate, computed IDENTICALLY for every
    variant so the three arms produce comparable columns.

    Deliberately independent of any network: V_true is a function of
    measured SOC alone, so this says the same thing whether the arm
    has a Lyapunov head, a merged critic, or neither. The lyapunov
    arm's own head-specific metrics are left untouched alongside
    these; they answer a different question.

    WHY THE ALPHA QUANTILES EXIST

    Reporting a violation rate at one fixed alpha makes the number
    uninterpretable across arms and across run lengths. On the 3-seed
    sweep the cost arm used alpha = 1 - gamma = 0.01 and the lyapunov
    arm used 0.001, and their 61% and 33% violation rates were
    therefore not comparable at all.

    Instead, invert the condition. For each sample outside the ball,

        dV + alpha*V + margin <= 0   <=>   alpha <= (-dV - margin)/V

    so alpha_feasible_i is the largest decay rate that sample
    supports. Its q-th quantile is exactly the largest alpha at which
    the violation rate would have been q. Logging the quantiles gives
    the whole violation-vs-alpha curve for free, from which any alpha
    can be read off afterwards without retraining.

    This also makes the certificate a REPORTED RESULT rather than a
    hyperparameter: state "the largest alpha for which 95% of sampled
    states satisfy the decrease condition is alpha_q05". Choosing
    alpha to minimize violations and then reporting the low violation
    rate would be circular; reporting the quantile is not.

    None of it touches training -- every caller is inside no_grad.
    """
    if alpha is None:
        alpha = ALPHA_STATE["alpha"]
    V = (F.relu(SOC_TARGET - soc) / SOC_TARGET) ** 2
    V_next = (F.relu(SOC_TARGET - soc_next) / SOC_TARGET) ** 2
    dV = V_next - V

    # Over an n-step window the exponential condition compounds:
    # V(s_{t+n}) <= (1-alpha)^n V(s_t). alpha stays the PER-STEP rate
    # everywhere it is stored or reported, so runs at different
    # CERT_NSTEP remain comparable; only the test uses alpha_n.
    alpha_n = 1.0 - (1.0 - alpha) ** nstep

    # Descent is asserted only outside the goal ball and inside the
    # certifiable region: below CERTIFY_MIN_ELEVATION no attainable
    # position sustains the robot, so dV < 0 is physically impossible
    # there and claiming it would be dishonest.
    certifiable = (elev_deg >= CERTIFY_MIN_ELEVATION).float()

    # TWO masks, because the two candidates are certified at DIFFERENT
    # windows and must not inherit each other's sample restrictions.
    #
    #   base    ball + elevation. What the CRITIC's n=1 check uses.
    #   outside base, minus samples whose n-step window ran past the
    #           end of their episode. What the ANALYTIC check uses.
    #
    # Before this split the critic inherited `outside` and silently
    # threw away every sample in the last CERT_NSTEP steps of each
    # episode -- 21% of the batch at n=150 -- for a reason that does
    # not apply to a one-step check.
    base = (V > ball).float() * certifiable
    outside = base * valid.float() if valid is not None else base
    n_out = outside.sum().clamp(min=1.0)
    n_base = base.sum().clamp(min=1.0)

    slack = dV + alpha_n * V + margin
    # Invert to the per-step rate so the quantiles mean the same thing
    # at any nstep: alpha_n <= (-dV - margin)/V, then
    # alpha = 1 - (1 - alpha_n)^(1/n).
    alpha_n_feasible = (-dV - margin) / V.clamp(min=1e-8)
    alpha_feasible = 1.0 - (1.0 - alpha_n_feasible.clamp(max=1.0 - 1e-9)) ** (
        1.0 / float(nstep))
    q05, q25, q50 = _masked_quantiles(alpha_feasible, outside)

    return {
        "cert_violation": float((((slack > 0).float() * outside).sum() / n_out).item()),
        "cert_mean_dV": float(((dV * outside).sum() / n_out).item()),
        "cert_mean_V": float(((V * outside).sum() / n_out).item()),
        "cert_worst_slack": float(torch.where(
            outside > 0, slack, torch.full_like(slack, -1e9)).max().item()),
        "cert_in_ball_rate": float((1.0 - (V > ball).float().mean()).item()),
        "cert_certifiable_frac": float(certifiable.mean().item()),
        "cert_alpha_q05": q05,
        "cert_alpha_q25": q25,
        "cert_alpha_q50": q50,
        "cert_alpha_used": float(alpha),
        "cert_nstep": float(nstep),
        "cert_n_samples": float(outside.sum().item()),
        # Fraction of certified samples with dV >= -margin, i.e.
        # where V is not decreasing at all.
        #
        # The Lyapunov condition REQUIRES dV < 0, so this is not a
        # tolerance to tune around -- it is the rate at which the
        # candidate fails outright, and no alpha >= 0 can rescue it.
        # It is broken out from cert_violation so the two failure
        # modes stay separate: cert_violation counts "did not decrease
        # FAST ENOUGH for the claimed alpha", this counts "did not
        # decrease at all". Only the first is about alpha.
        #
        # Expect this to be large for V_true, and note that it is a
        # statement about the CANDIDATE, not about the controller. A
        # function of SOC alone cannot be a Lyapunov function for this
        # system: reaching sun costs charge, so any trajectory that
        # has to traverse shade drives SOC down and V_true up while
        # moving strictly toward the goal. Compare against
        # v_critic_alpha_nonpos_frac below -- the merged critic's
        # V_cost is a discounted cost-to-go over shade AND deficit AND
        # motion, so it can fall while SOC falls. If it does, that is
        # the result: the critic is a valid candidate where the
        # analytic one is not.
        "cert_alpha_nonpos_frac": float(
            (((alpha_feasible <= 0).float() * outside).sum() / n_out).item()),
    }, outside, n_out, alpha_feasible, base, n_base


def adapt_alpha(alpha_feasible, outside):
    """
    Slide ALPHA_STATE["alpha"] to the decay rate the system actually
    supports. No-op unless COST_ALPHA_MODE == "adaptive".

    Called ONCE per update(), after the minibatch loop -- never inside
    it. Changing alpha changes what the check asserts, so minibatches
    of the same update must all be evaluated at one value.

    EMA in log space, because alpha spans orders of magnitude (the
    asserted 0.01 against a measured ~0.0012) and a linear EMA crawls
    at the small end. Free to move down AND up: a ratchet that only
    tightened would overclaim after a late-run regression.
    """
    if COST_ALPHA_MODE != "adaptive":
        return ALPHA_STATE["alpha"]

    sel = alpha_feasible[outside > 0]
    if sel.numel() == 0:
        return ALPHA_STATE["alpha"]

    target = float(torch.quantile(sel.float(), COST_ALPHA_QUANTILE).item())
    target = min(max(target, COST_ALPHA_MIN), COST_ALPHA_MAX)

    cur = ALPHA_STATE["alpha"]
    if target <= 0.0 or cur <= 0.0:
        # Log space is undefined at zero. alpha = 0 is a legitimate
        # claim (V non-increasing) and the system can genuinely sit
        # there early in training, so fall back to a linear step.
        new = (1.0 - COST_ALPHA_EMA) * cur + COST_ALPHA_EMA * target
    else:
        new = math.exp((1.0 - COST_ALPHA_EMA) * math.log(cur)
                       + COST_ALPHA_EMA * math.log(target))
    ALPHA_STATE["alpha"] = min(max(new, COST_ALPHA_MIN), COST_ALPHA_MAX)
    return ALPHA_STATE["alpha"]


def critic_certification(V_cost, V_cost_next, cost, outside, n_out,
                         alpha=None, margin=LYAPUNOV_MARGIN):
    """
    The decrease condition for the MERGED critic, computed two ways.

    Evaluated on the same `outside` mask as analytic_certification, so
    the rates are directly comparable.

    WHY TWO FORMS

    The obvious estimator, dV = V(s') - V(s), differences two noisy
    critic outputs. Measured on cost seed 1: the critic's RMSE is
    0.0372 while the true per-step dV is -0.00191 +- 0.00226 -- the
    error is 16x the signal. Injecting Gaussian noise at that level
    into a direct difference reproduces the observed 0.463
    non-decrease rate almost exactly (0.467 at zero error
    correlation), against a true rate of 0.15. The direct form was
    measuring its own noise.

    Aggregating does not rescue it. dV has lag-1 autocorrelation 0.969
    along a trajectory, a 64x variance inflation, so even pooling all
    288,000 samples of a 400-episode run reaches only t = -2.4.

    The Bellman form uses the cost MDP's own recursion,
    V(s) = c(s) + gamma*V(s'), rearranged:

        dV = [ (1 - gamma)*V(s) - c(s) ] / gamma

    c(s) is measured exactly, and V(s) enters weighted by (1 - gamma).
    The critic's error is therefore attenuated 100x -- 0.00037 against
    a 0.00191 signal. Verified exact on the analytic values (max
    deviation 1.9e-16), and it recovers 0.279 against a truth of
    0.283, independent of error correlation.

    WHAT EACH ONE CERTIFIES

    They differ by exactly the Bellman residual:

        delta(s)   = V(s) - c(s) - gamma*V(s')
        dV_direct  = dV_bellman - delta/gamma

    so the Bellman form leans on V only lightly, and the honest claim
    is conditional rather than absolute:

        V is a Lyapunov function wherever the Bellman-form condition
        holds with margin m + |delta|/gamma.

    That is why delta's distribution is logged here and not just its
    mean -- q95 is what turns the conditional into a stated
    confidence level. Shrinking |delta| tightens the certificate, and
    |delta| is the same quantity the value loss already minimizes, so
    fit quality and certificate quality stop being separate goals.

    The direct form is still reported, labelled, so the noise gap
    stays visible rather than being quietly dropped.
    """
    if alpha is None:
        alpha = ALPHA_STATE["alpha"]

    dV_direct = V_cost_next - V_cost
    dV_bellman = ((1.0 - GAMMA) * V_cost - cost) / GAMMA
    delta = V_cost - cost - GAMMA * V_cost_next

    slack_b = dV_bellman + alpha * V_cost + margin
    slack_d = dV_direct + alpha * V_cost + margin
    alpha_feasible = (-dV_bellman - margin) / V_cost.clamp(min=1e-8)
    q05, q25, q50 = _masked_quantiles(alpha_feasible, outside)

    sel = delta[outside > 0]
    if sel.numel() == 0:
        d_mean = d_std = d_q95 = float("nan")
    else:
        d_mean = float(sel.mean().item())
        d_std = float(sel.std().item()) if sel.numel() > 1 else 0.0
        d_q95 = float(torch.quantile(sel.abs().float(), 0.95).item())

    frac = lambda t: float(((t.float() * outside).sum() / n_out).item())
    return {
        "v_critic_violation": frac(slack_b > 0),
        "v_critic_violation_direct": frac(slack_d > 0),
        "v_critic_mean": float(V_cost.mean().item()),
        "v_critic_min": float(V_cost.min().item()),
        "v_critic_alpha_q05": q05,
        "v_critic_alpha_q25": q25,
        "v_critic_alpha_q50": q50,
        "v_critic_alpha_nonpos_frac": frac(alpha_feasible <= 0),
        "v_critic_nonpos_direct": frac(dV_direct >= -margin),
        # Bellman residual: the exact gap between the two forms, and
        # the error term in the conditional certificate above.
        "v_bellman_delta_mean": d_mean,
        "v_bellman_delta_std": d_std,
        "v_bellman_delta_q95": d_q95,
    }


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
    # c(s) for the Bellman-form dV, and the index of the state
    # CERT_NSTEP ahead WITHIN THE SAME EPISODE (-1 where the window
    # would run past the end). Both are certification-only; neither
    # touches the loss.
    costs=[]; nstep_idx=[]
    offset = 0

    for r in rollouts:
        T = len(r["rewards"])
        costs += [-x for x in r["rewards"]]
        nstep_idx += [offset + t + CERT_NSTEP if t + CERT_NSTEP < T else -1
                      for t in range(T)]
        offset += T
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

    costs = torch.tensor(costs, device=device, dtype=torch.float32)
    nstep_idx = torch.tensor(nstep_idx, device=device, dtype=torch.long)

    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    return (states, next_states, actions, oldlp, adv, returns,
            costs, nstep_idx)

def update(model,opt,rollouts,device, ep, metrics_writer=None, return_var_tracker=None,
           control_params=None, auxiliary_param_list=None):
    model.train()

    if control_params is None:
        control_params = list(model.parameters())
        auxiliary_param_list = []

    # Cosine LR decay on the control groups. Groups are tagged with
    # "initial_lr" at construction; any group without the tag (the
    # auxiliary heads) is left at a constant rate.
    decay_frac = min(ep, TOTAL_EPISODES) / max(TOTAL_EPISODES, 1)
    lr_scale = LR_DECAY_FINAL + (1.0 - LR_DECAY_FINAL) * 0.5 * (
        1.0 + math.cos(math.pi * decay_frac)
    )
    for group in opt.param_groups:
        if "initial_lr" in group:
            group["lr"] = group["initial_lr"] * lr_scale

    (states, next_states, actions, oldlp, adv, returns,
     costs, nstep_idx) = compute_batch(rollouts, device)
    # SOC for every state in the batch, so the n-step analytic check
    # can index forward without a second forward pass.
    soc_all = states[:, -1, SOC_OBS_INDEX]

    if TRANSFORMER_VARIANT == "lyapunov":
        # Curriculum boundaries scale with TOTAL_EPISODES. Hardcoded at
        # 100 and 300 they were 10% and 30% of a 1000-episode run; on a
        # 400-episode paired sweep they would have been 25% and 75%,
        # leaving the auxiliary heads at full weight for only the last
        # quarter. The fractions below reproduce 100 and 300 exactly at
        # the default length.
        if ep < CURRICULUM_WARMUP_EPISODES:
            lyap_weight = 0.0
            barrier_weight = 0.0
            dynamics_weight = 0.0
        elif ep < CURRICULUM_FULL_EPISODES:
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
        last_alpha_feasible = None
        accum = {}
        # Per-key counts, because the alpha quantiles are NaN in any
        # minibatch with no samples outside the ball. A single NaN in
        # a running sum poisons the average for the rest of the
        # update, so those entries are skipped rather than added.
        accum_n = {}
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

                # Same helper the other two arms call, on the same
                # squared V_true and the same elevation gate, so all
                # three runs emit identical cert_* columns. The
                # lyapunov head's own metrics below are kept as-is --
                # they use this arm's linear V_true and its own alpha,
                # and answer a different question.
                with torch.no_grad():
                    mb_elev_cert = mb_states[:, -1, ELEV_OBS_INDEX] * 90.0
                    mb_fwd = nstep_idx[mb]
                    mb_valid = mb_fwd >= 0
                    mb_soc_fwd = soc_all[mb_fwd.clamp(min=0)]
                    (cert_metrics, _cert_outside, _cert_n_out,
                     _cert_alpha_feasible, _cert_base,
                     _cert_n_base) = analytic_certification(
                        mb_soc, mb_soc_fwd, mb_elev_cert,
                        valid=mb_valid, nstep=CERT_NSTEP)
                    last_alpha_feasible = (_cert_alpha_feasible.detach(),
                                           _cert_outside.detach())

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
                    "lr_scale": float(lr_scale),
                    "lyap_mean_dV": float(lyap_mean_dV.item()),
                    "lyap_worst_slack": float(lyap_worst_slack.item()),
                    **cert_metrics,
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
                    if v != v:      # NaN
                        continue
                    accum[k] = accum.get(k, 0.0) + v
                    accum_n[k] = accum_n.get(k, 0) + 1
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

        avg = {k: v / max(accum_n.get(k, n_minibatches), 1)
               for k, v in accum.items()}

        # Slide alpha once per update, from the last minibatch's
        # feasible-alpha distribution. Applied AFTER the loop, so this
        # update's reported violation rates were all computed at one
        # alpha; the new value takes effect next update.
        if last_alpha_feasible is not None:
            avg["cert_alpha_next"] = adapt_alpha(*last_alpha_feasible)

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

        # Certification line, identical across all three arms so the
        # runs can be diffed directly. a*: the largest decay rate the
        # 5th/50th-percentile sample supports -- read a violation rate
        # of 5% off the first and 50% off the second. Compare a*_q05
        # against whatever alpha is being claimed; if the claim
        # exceeds it, the violation rate is measuring the claim, not
        # the controller.
        print(
            f"  cert | viol {avg.get('cert_violation', float('nan')):.3f}"
            f" | dV {avg.get('cert_mean_dV', float('nan')):+.6f}"
            f" | V {avg.get('cert_mean_V', float('nan')):.4f}"
            f" | inball {avg.get('cert_in_ball_rate', float('nan')):.3f}"
            f" | a*_q05 {avg.get('cert_alpha_q05', float('nan')):.5f}"
            f" | a*_q50 {avg.get('cert_alpha_q50', float('nan')):.5f}"
            f" | dV>=0 {avg.get('cert_alpha_nonpos_frac', float('nan')):.3f}"
            f" | n={CERT_NSTEP}"
            f" | a_used {avg.get('cert_alpha_used', float('nan')):.5f}"
            f" -> {avg.get('cert_alpha_next', float('nan')):.5f}"
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
        last_v_cost = None
        last_alpha_feasible = None
        accum = {}
        # Per-key counts, because the alpha quantiles are NaN in any
        # minibatch with no samples outside the ball. A single NaN in
        # a running sum poisons the average for the rest of the
        # update, so those entries are skipped rather than added.
        accum_n = {}
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

                # Lyapunov certification, cost variant only.
                #
                # Certified on the ANALYTIC V from measured state of
                # charge, not on the critic. A neural V cannot be
                # certified over a 1689-dimensional observation: the
                # covering argument needs (1/delta)^d samples, and the
                # analytic form is a function of ONE variable. The
                # critic's own descent and its agreement with the
                # analytic V are logged alongside, which is the
                # evidence that it learned a valid certificate rather
                # than only a value estimate.
                # Runs for BOTH non-lyapunov arms. The analytic half
                # needs only SOC, so the normal arm gets the same
                # certification columns as the cost arm and the three
                # can finally be compared on one scale.
                with torch.no_grad():
                    mb_soc = states[mb][:, -1, SOC_OBS_INDEX]
                    mb_elev = states[mb][:, -1, ELEV_OBS_INDEX] * 90.0
                    # n-step ahead SOC, from the precomputed index.
                    # -1 marks a window that ran off the end of its
                    # episode; clamped so the gather is safe and then
                    # masked out via `valid`.
                    mb_fwd = nstep_idx[mb]
                    mb_valid = mb_fwd >= 0
                    mb_soc_next = soc_all[mb_fwd.clamp(min=0)]
                    (cert_metrics, cert_outside, cert_n_out,
                     cert_alpha_feasible, cert_base,
                     cert_n_base) = analytic_certification(
                        mb_soc, mb_soc_next, mb_elev,
                        valid=mb_valid, nstep=CERT_NSTEP)
                    last_alpha_feasible = (cert_alpha_feasible.detach(),
                                           cert_outside.detach())

                if TRANSFORMER_VARIANT in COST_VARIANTS:
                    with torch.no_grad():
                        # V_cost = -value is the merged critic acting
                        # as the Lyapunov function. Evaluated on the
                        # SAME `cert_outside` mask as the analytic
                        # certificate above, so the two violation
                        # rates are directly comparable -- that
                        # comparison is the evidence that the merge
                        # holds.
                        V_true = (F.relu(SOC_TARGET - mb_soc) / SOC_TARGET) ** 2
                        V_cost = -values
                        V_cost_next = -model.value_only(next_states[mb])
                        # cert_base, NOT cert_outside: the critic is
                        # certified at n=1, so the n-step window's
                        # episode-boundary restriction does not apply
                        # to it. Same ball and elevation gate.
                        critic_metrics = critic_certification(
                            V_cost, V_cost_next, costs[mb],
                            cert_base, cert_n_base)
                        # Kept for the once-per-update beta slide
                        # below. Deliberately NOT applied inside the
                        # loop: changing beta changes the head, so
                        # values from different minibatches of the
                        # same update would not be comparable.
                        last_v_cost = V_cost.detach()

                        # Correlation rather than RMSE: V_cost is a
                        # discounted SUM of costs while V_true is a
                        # single-step deficit, so they differ by a
                        # factor near 1/(1-gamma) and an RMSE between
                        # them would measure that scale, not agreement.
                        if V_cost.numel() > 1 and V_true.std() > 1e-8:
                            v_agreement = float(torch.corrcoef(
                                torch.stack([V_cost, V_true]))[0, 1].item())
                        else:
                            v_agreement = 0.0
                        critic_metrics["v_agreement"] = v_agreement

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
                    "lr_scale": float(lr_scale),
                    "alpha": float(alpha),
                }
                # Both arms log the analytic certificate; only cost
                # has a critic to certify.
                batch_metrics.update(cert_metrics)
                if TRANSFORMER_VARIANT in COST_VARIANTS:
                    batch_metrics.update(critic_metrics)
                for k, v in batch_metrics.items():
                    if v != v:      # NaN
                        continue
                    accum[k] = accum.get(k, 0.0) + v
                    accum_n[k] = accum_n.get(k, 0) + 1
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

        avg = {k: v / max(accum_n.get(k, n_minibatches), 1)
               for k, v in accum.items()}

        # Slide beta once per update, using the last minibatch's
        # V_cost. No-op unless the model was built with
        # beta_gain_target. Logged either way so a run's head is
        # reconstructable from its CSV.
        if TRANSFORMER_VARIANT in COST_VARIANTS and last_v_cost is not None:
            avg["softplus_beta"] = model.adapt_beta(last_v_cost)

        # Slide alpha once per update, from the last minibatch's
        # feasible-alpha distribution. Applied AFTER the loop, so this
        # update's reported violation rates were all computed at one
        # alpha; the new value takes effect next update.
        if last_alpha_feasible is not None:
            avg["cert_alpha_next"] = adapt_alpha(*last_alpha_feasible)

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

        # Certification line, identical across all three arms so the
        # runs can be diffed directly. a*: the largest decay rate the
        # 5th/50th-percentile sample supports -- read a violation rate
        # of 5% off the first and 50% off the second. Compare a*_q05
        # against whatever alpha is being claimed; if the claim
        # exceeds it, the violation rate is measuring the claim, not
        # the controller.
        print(
            f"  cert | viol {avg.get('cert_violation', float('nan')):.3f}"
            f" | dV {avg.get('cert_mean_dV', float('nan')):+.6f}"
            f" | V {avg.get('cert_mean_V', float('nan')):.4f}"
            f" | inball {avg.get('cert_in_ball_rate', float('nan')):.3f}"
            f" | a*_q05 {avg.get('cert_alpha_q05', float('nan')):.5f}"
            f" | a*_q50 {avg.get('cert_alpha_q50', float('nan')):.5f}"
            f" | dV>=0 {avg.get('cert_alpha_nonpos_frac', float('nan')):.3f}"
            f" | n={CERT_NSTEP}"
            f" | a_used {avg.get('cert_alpha_used', float('nan')):.5f}"
            f" -> {avg.get('cert_alpha_next', float('nan')):.5f}"
        )
        if TRANSFORMER_VARIANT in COST_VARIANTS:
            print(
                f"  crit | n=1 | viol {avg.get('v_critic_violation', float('nan')):.3f}"
                f" | V {avg.get('v_critic_mean', float('nan')):.4f}"
                f" | Vmin {avg.get('v_critic_min', float('nan')):.4f}"
                f" | agree {avg.get('v_agreement', float('nan')):+.3f}"
                f" | a*_q05 {avg.get('v_critic_alpha_q05', float('nan')):.5f}"
                f" | a*_q50 {avg.get('v_critic_alpha_q50', float('nan')):.5f}"
                f" | dV>=0 {avg.get('v_critic_alpha_nonpos_frac', float('nan')):.3f}"
                f" | beta {avg.get('softplus_beta', float(model.softplus_beta)):.4f}"
            )
            # Secondary. The direct difference is noise-dominated at
            # the measured critic error -- printed so the gap stays
            # visible, NOT to be read as a failure rate. |d| is the
            # Bellman residual, the error term in the conditional
            # certificate; q95 is what the claim should be stated at.
            print(
                f"  bell | direct viol {avg.get('v_critic_violation_direct', float('nan')):.3f}"
                f" | direct dV>=0 {avg.get('v_critic_nonpos_direct', float('nan')):.3f}"
                f" | d_mean {avg.get('v_bellman_delta_mean', float('nan')):+.5f}"
                f" | d_std {avg.get('v_bellman_delta_std', float('nan')):.5f}"
                f" | d_q95 {avg.get('v_bellman_delta_q95', float('nan')):.5f}"
            )
        if metrics_writer is not None:
            row = {"episode": ep, "minibatches": n_minibatches,
                   "minibatches_possible": PPO_EPOCHS * math.ceil(n / MINIBATCH_SIZE),
                   "first_mb_kl": 0.0}
            row.update(avg)
            metrics_writer.writerow(row)

        # Left None deliberately. The convergence block gated on
        # `diag_lyap is not None` also sums diag_barrier and
        # diag_mean_V, which have no meaning without auxiliary heads --
        # populating only this one would crash on the first check.
        # The cost variant's certification is written to
        # training_metrics.csv and read by certify_stability.py.
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

            # "initial_lr" tags the groups the cosine decay applies to.
            # The auxiliary group is left untagged and so runs at a
            # constant rate -- see LR_DECAY_FINAL.
            opt = optim.AdamW([{"params": transformer_params, "lr": 1e-3, "weight_decay":1e-5, "initial_lr": 1e-3},
                               {"params": actor_params,       "lr": 3e-4, "weight_decay":1e-5, "initial_lr": 3e-4},
                               {"params": critic_params,      "lr": 3e-4, "weight_decay":1e-5, "initial_lr": 3e-4},
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
                               {"params": log_std_params,     "lr": 3e-4, "weight_decay":1e-5, "initial_lr": 3e-4},],
                              eps=1e-5,)
        elif TRANSFORMER_VARIANT in COST_VARIANTS:
            from cost_transformer import CostTransformerActorCritic
            if TRANSFORMER_VARIANT == "cost":
                model = CostTransformerActorCritic(
                    VIEW_DISTANCE, scalar_dim=SCALAR_DIM,
                    sequence_length=SEQUENCE_LENGTH,
                    softplus_beta=COST_BETA_INIT,
                    beta_gain_target=COST_BETA_GAIN_TARGET).to(device)
                print(f"[cost] critic head: beta*softplus, spectral, "
                      f"beta0 = {COST_BETA_INIT}, "
                      f"gain target = {COST_BETA_GAIN_TARGET}"
                      + (" (fixed beta)" if COST_BETA_GAIN_TARGET is None else ""))
            else:
                # Ablation arms. Same reward, same trunk, same RNG
                # consumption order -- only the critic construction
                # differs. cost_linear keeps the Lipschitz bound and
                # drops the sign constraint; cost_plain drops both.
                from ablation_transformer import AblationTransformerActorCritic
                use_spectral = (TRANSFORMER_VARIANT == "cost_linear")
                model = AblationTransformerActorCritic(
                    VIEW_DISTANCE, scalar_dim=SCALAR_DIM,
                    sequence_length=SEQUENCE_LENGTH,
                    spectral_critic=use_spectral).to(device)
                print(f"[{TRANSFORMER_VARIANT}] critic head: unconstrained linear, "
                      f"spectral_norm = {use_spectral}. V_cost >= 0 is NOT "
                      f"structural in this arm -- watch v_critic_min.")
            # Same per-group rates and decay tags as the other two
            # arms, so the comparison isolates the formulation.
            opt = optim.AdamW([
                {"params": (list(model.input_projection.parameters())
                            + list(model.encoder.parameters())
                            + list(model.attention_pool.parameters())
                            + [model.position_embedding]),
                 "lr": 1e-3, "weight_decay": 1e-5, "initial_lr": 1e-3},
                {"params": list(model.actor.parameters()),
                 "lr": 3e-4, "weight_decay": 1e-5, "initial_lr": 3e-4},
                # weight_decay = 0 on the critic, unlike the other
                # arms. Its Linear layers are spectral-normalized, so
                # the learnable tensor is `original` and the EFFECTIVE
                # weight is original / sigma_max(original). Decay
                # shrinks numerator and denominator together and so
                # has no effect on the weight the network actually
                # uses -- it only drives ||original|| toward zero,
                # which degrades the power-iteration estimate of
                # sigma_max over a long run. Regularizing here is all
                # cost and no benefit; spectral norm already bounds
                # the critic.
                # ...which means the justification only holds where
                # spectral norm is actually registered. cost_plain has
                # a bare critic, so it takes the same 1e-5 as every
                # other unnormalized parameter group in every arm.
                {"params": list(model.critic_body.parameters()),
                 "lr": 3e-4,
                 "weight_decay": (1e-5 if TRANSFORMER_VARIANT == "cost_plain"
                                  else 0.0),
                 "initial_lr": 3e-4},
                {"params": [model.log_std_param],
                 "lr": 3e-4, "weight_decay": 1e-5, "initial_lr": 3e-4},
            ],
                # Matches the lyapunov and normal arms. Omitting it
                # left this arm on AdamW's default 1e-8, which is a
                # difference in the optimizer rather than in the
                # formulation the comparison is supposed to isolate.
                eps=1e-5)

            # No auxiliary heads, so the per-group clip is equivalent
            # to a single global one.
            control_params = list(model.parameters())
            auxiliary_param_list = []
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
                 "lr": 1e-3, "weight_decay": 1e-5, "initial_lr": 1e-3},
                {"params": list(model.actor.parameters()),
                 "lr": 3e-4, "weight_decay": 1e-5, "initial_lr": 3e-4},
                {"params": list(model.critic.parameters()),
                 "lr": 3e-4, "weight_decay": 1e-5, "initial_lr": 3e-4},
                {"params": [model.log_std_param],
                 "lr": 3e-4, "weight_decay": 1e-5, "initial_lr": 3e-4},
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
        "lyap_v_rmse","grad_norm_control","grad_norm_aux","lr_scale",
        "certifiable_frac",
        "cert_violation","cert_mean_dV","cert_mean_V","cert_worst_slack",
        "cert_in_ball_rate","cert_certifiable_frac",
        "cert_alpha_q05","cert_alpha_q25","cert_alpha_q50",
        "cert_alpha_used","cert_alpha_next","cert_alpha_nonpos_frac",
        "cert_nstep","cert_n_samples",
        "v_critic_violation","v_critic_mean","v_critic_min","v_agreement",
        "v_critic_alpha_q05","v_critic_alpha_q25","v_critic_alpha_q50",
        "v_critic_alpha_nonpos_frac","v_critic_violation_direct",
        "v_critic_nonpos_direct","v_bellman_delta_mean",
        "v_bellman_delta_std","v_bellman_delta_q95",
        "softplus_beta",
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

            if TRANSFORMER_VARIANT in COST_VARIANTS:
                rew_total, rew_directional, rew_battery, rew_movement = reward_fn_cost(
                    after, tel, aft_batt / 100.0)
            else:
                rew_total, rew_directional, rew_battery, rew_movement = reward_fn(
                    after, tel, aft_batt-b_batt, current_action)
            rew = rew_total - EFFECTIVE_ACTION_SMOOTHNESS*smoothness

            total+=rew; previous_action=current_action
            total_directional+=rew_directional; total_battery_reward+=rew_battery; total_movement_penalty+=rew_movement
            r["states"].append(np.asarray(h)); r["actions"].append(stored_action[0].cpu().numpy())
            r["logps"].append(lp.item()); r["values"].append(v.item()); r["rewards"].append(rew)
            x,y,yaw=env.ch.get_position(); h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            r["next_states"].append(np.asarray(h))
            if aft_batt<=0:
                # Also charge it to the LOGGED total. reward_history
                # feeds REWARD_CONVERGENCE_THRESHOLD, best-model
                # tracking and the degradation guard; without this a
                # run that learned to die early would read as its own
                # best checkpoint.
                if TRANSFORMER_VARIANT in COST_VARIANTS:
                    total -= death_cost_logged(step)
                break

        rollout_elapsed = time.perf_counter() - rollout_start
        total_rollout_time += rollout_elapsed

        # GAE needs a bootstrap value for whatever comes after the last
        # recorded step.
        #
        # If the battery died, what comes after is the absorbing
        # failure state. For the REWARD arms its value is 0: no more
        # positive reward is collectable, which is the penalty. For the
        # COST arms 0 is the best value in the state space and would
        # make dying optimal -- they get -death_cost_to_go() instead.
        # See COST_DEATH_COST.
        #
        # If the loop only ended because MAX_STEPS_PER_EPISODE was
        # reached, the environment did not terminate at all;
        # bootstrapping with 0 would tell the value function "nothing
        # more happens here" when that isn't true. Use the critic's own
        # estimate of the real next state instead.
        if POLICY_TYPE == "transformer":
            if aft_batt <= 0:
                bootstrap_value = (-death_cost_to_go(step)
                                   if TRANSFORMER_VARIANT in COST_VARIANTS
                                   else 0.0)
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
                        # Sign-agnostic. The old form guarded on
                        # `best_avg_reward > 0` and fell through to
                        # regression = 0.0 otherwise, which silently
                        # DISABLED the guard for any run with negative
                        # returns -- i.e. the entire cost variant, and
                        # any reward arm that happened to be scoring
                        # below zero. That is the one mechanism
                        # protecting a run from late collapse.
                        #
                        # The scale floor keeps the ratio finite when
                        # best_avg_reward sits near zero.
                        reward_scale = max(abs(best_avg_reward),
                                           DEGRADATION_SCALE_FLOOR)
                        regression = (best_avg_reward - avg_reward) / reward_scale
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

            if TRANSFORMER_VARIANT in COST_VARIANTS:
                rew, rew_directional, rew_battery, rew_movement = reward_fn_cost(
                    aft, tel, aft_batt / 100.0)
            else:
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
            if aft_batt<=0:
                if TRANSFORMER_VARIANT in COST_VARIANTS:
                    val_total -= death_cost_logged(step)
                break

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
