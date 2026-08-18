"""
Particle swarm optimisation as an ONLINE PLANNER.

WHAT CHANGED FROM THE PREVIOUS VERSION, AND WHY

The earlier PSOPolicy used the swarm to search NETWORK WEIGHTS: one
particle was an 8,000-parameter vector defining a 2-layer MLP, held
fixed for a whole episode, scored by that episode's return. That is
neuroevolution, and it is the wrong instrument here on two counts.

First, dimensionality. PSO carries no gradient information; it is a
population method and it works where the search space is small --
tens to low hundreds of dimensions. At 8,000 a 30-particle swarm is
sampling a space it cannot cover, and each fitness evaluation costs a
full 720-step episode. A hundred swarm iterations is 3,000 episodes
per seed, roughly 36 hours, to search a space gradient descent
crosses in 400.

Second, it is not what PSO is for in a control setting. The standard
practical use is model-predictive control: at every step, search the
ACTION space against a model of what each action would do, commit the
best action, discard the plan, repeat. The search space is then the
action dimension -- 2 here -- which is exactly the regime PSO is good
at, and there is no training phase at all.

So this is a planner, not a policy that learns. It has no parameters,
consumes no gradient, and its performance is whatever the model and
the fitness function are worth. That makes it a genuine classical
baseline: the question it answers is how much of the task is solved
by a known model plus online optimisation, without learning anything.

THE MODEL IT PLANS AGAINST

Only what the observation already contains, so the comparison is
fair -- the planner sees no more than the learned arms do.

    patch    (2*VIEW_DISTANCE+1)^2 values of `potential` = 1 - obfuscation,
             a local map of solar potential centred on the vehicle.
             This is the whole model: it says where the sun is.
    scalars  x, y (normalised), sin/cos yaw, SOC, azimuth,
             elevation_norm, sun_usable.

An action (dx, dy) in [-1, 1]^2 is a target offset of
(dx, dy) * MAX_MOVE_PER_STEP cells, which lands inside the patch
because MAX_MOVE_PER_STEP (20) equals VIEW_DISTANCE (20). That
coincidence is what makes a one-step horizon the natural one: the
planner can see exactly as far as it can move in a step, and no
further, so a longer horizon would require predicting patches it has
never observed. Reported as a limit of the baseline rather than
hidden.

FITNESS

The negated one-step cost, built from the same terms as the cost MDP
so the planner is optimising the same thing the cost arms are trained
on:

    c_shade    obfuscation at the target cell        = 1 - potential
    c_move     distance / MAX_MOVE_PER_STEP
    c_deficit  (relu(SOC_TARGET - soc_next)/SOC_TARGET)^2, with
               soc_next projected from the harvest the target cell
               would give against a fixed drain

Weights mirror COST_SHADE_W / COST_MOVE_W / COST_DEFICIT_W so the
objective matches. Vectorised over the whole swarm: one numpy pass
per PSO iteration, no Python loop over particles.

A NOTE ON PSO IN TWO DIMENSIONS

Two dimensions is small enough to solve exhaustively -- ExactPolicy
below does exactly that, ~20x faster than the converged swarm. PSO is
not the efficient choice here; it is the requested one, implemented
faithfully and run to convergence so the comparison measures the
approach rather than a truncated search.

The pair is the point. ExactPolicy shares this class's fitness,
model, observation decoding and horizon, and differs only in how the
action is found, so:

    swarm vs exact          isolates search quality
    exact vs learned arms   isolates the planner's FORMULATION --
                            one step, greedy, no lookahead
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn

SWARM_SIZE = int(os.environ.get("LTAC_PSO_SWARM", "24"))
SWARM_ITERS = int(os.environ.get("LTAC_PSO_ITERS", "64"))
#
# 64, not 16. Measured against the exact solver over 400 random states:
#
#     iters   matched exact   mean cost gap   s/episode
#        16        1%            0.0101          1.15
#        32       12%            0.0033          2.25
#        64       86%            0.0013          4.45
#       128       90%            0.0013          8.33
#
# At 16 the swarm was leaving ~0.010 of cost on the table on almost
# every step, so a comparison against the learned arms would have been
# partly a measure of an under-converged optimiser. 64 gets the gap to
# 0.0013 and the curve flattens after that -- 128 buys nothing for
# twice the time. Against ~35 s of environment per episode, 4.45 s is
# about 13%, which is the price of making the baseline honest.
PSO_INERTIA = float(os.environ.get("LTAC_PSO_W", "0.7"))
PSO_C1 = float(os.environ.get("LTAC_PSO_C1", "1.5"))
PSO_C2 = float(os.environ.get("LTAC_PSO_C2", "1.5"))

# Mirrors of main.py, kept as explicit constants so this file can be
# read on its own. If they drift there, they must be changed here.
MAX_MOVE_PER_STEP = 20.0
SOC_TARGET = 0.90
W_SHADE = 1.00
W_DEFICIT = 2.00
W_MOVE = 0.50

# One-step SOC change, as fractions of capacity. Crude on purpose: the
# planner is not given a battery model the learned arms do not have,
# only the sign and rough magnitude of harvest versus drain.
DRAIN_PER_STEP = float(os.environ.get("LTAC_PSO_DRAIN", "0.0016"))
HARVEST_PER_STEP = float(os.environ.get("LTAC_PSO_HARVEST", "0.0030"))


class PSOPolicy(nn.Module):
    """
    Online PSO planner with the same call signature as the learned
    arms, so main.py's rollout does not need a separate code path.

    `act` returns (action, raw_action, log_prob, value). log_prob is
    zero and value is zero: there is no distribution and no critic.
    Anything downstream that consumes them -- PPO ratios, GAE,
    entropy -- is meaningless for this arm and its update step should
    be skipped, which is what `opt = None` in main.py signals.
    """

    def __init__(self, input_dim, output_dim=2, view_distance=20,
                 scalar_dim=8, **_ignored):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.view_distance = int(view_distance)
        self.scalar_dim = int(scalar_dim)
        self.size = 2 * self.view_distance + 1
        self.patch_dim = self.size * self.size

        if self.patch_dim + self.scalar_dim != input_dim:
            raise ValueError(
                f"input_dim {input_dim} != patch {self.patch_dim} + "
                f"scalars {self.scalar_dim}; check VIEW_DISTANCE")

        # A parameter so .to(device), .parameters() and state_dict()
        # behave; never read, never trained.
        self.register_buffer("_unused", torch.zeros(1))
        self.rng = np.random.default_rng(0)
        self.last_fitness = float("nan")

    # ---------------------------------------------------------------
    # model
    # ---------------------------------------------------------------

    def _patch_lookup(self, patch, dx, dy):
        """
        Solar potential at the target cell for each candidate action.

        Nearest-cell rather than bilinear: the patch is already a
        coarse view and interpolating would imply a spatial resolution
        the observation does not have. Out-of-patch targets cannot
        occur while MAX_MOVE_PER_STEP <= VIEW_DISTANCE, but the clip
        makes that assumption explicit rather than silent.
        """
        cx = np.clip(np.rint(dx * MAX_MOVE_PER_STEP).astype(int),
                     -self.view_distance, self.view_distance)
        cy = np.clip(np.rint(dy * MAX_MOVE_PER_STEP).astype(int),
                     -self.view_distance, self.view_distance)
        rows = cy + self.view_distance
        cols = cx + self.view_distance
        return patch.reshape(self.size, self.size)[rows, cols]

    def _cost(self, patch, soc, sun_usable, elevation, actions):
        """One-step cost for a [k, 2] batch of candidate actions."""
        dx, dy = actions[:, 0], actions[:, 1]
        potential = self._patch_lookup(patch, dx, dy)

        c_shade = 1.0 - potential
        dist = np.hypot(dx, dy) * MAX_MOVE_PER_STEP
        c_move = dist / MAX_MOVE_PER_STEP

        harvest = HARVEST_PER_STEP * potential * sun_usable * elevation
        soc_next = np.clip(soc + harvest - DRAIN_PER_STEP, 0.0, 1.0)
        c_deficit = (np.maximum(SOC_TARGET - soc_next, 0.0) / SOC_TARGET) ** 2

        return W_SHADE * c_shade + W_DEFICIT * c_deficit + W_MOVE * c_move

    # ---------------------------------------------------------------
    # search
    # ---------------------------------------------------------------

    def plan(self, patch, soc, sun_usable, elevation):
        """Run one swarm to convergence and return the best action."""
        pos = self.rng.uniform(-1.0, 1.0, size=(SWARM_SIZE, 2))
        vel = self.rng.uniform(-0.2, 0.2, size=(SWARM_SIZE, 2))

        fit = self._cost(patch, soc, sun_usable, elevation, pos)
        pbest_pos, pbest_fit = pos.copy(), fit.copy()
        g = int(np.argmin(pbest_fit))
        gbest_pos, gbest_fit = pbest_pos[g].copy(), float(pbest_fit[g])

        for _ in range(SWARM_ITERS):
            r1 = self.rng.random((SWARM_SIZE, 2))
            r2 = self.rng.random((SWARM_SIZE, 2))
            vel = (PSO_INERTIA * vel
                   + PSO_C1 * r1 * (pbest_pos - pos)
                   + PSO_C2 * r2 * (gbest_pos[None, :] - pos))
            # Velocity clamp. Without it the swarm oscillates out of
            # the box and every particle spends its iterations pinned
            # to a corner by the position clip.
            vel = np.clip(vel, -0.5, 0.5)
            pos = np.clip(pos + vel, -1.0, 1.0)

            fit = self._cost(patch, soc, sun_usable, elevation, pos)
            better = fit < pbest_fit
            pbest_pos[better] = pos[better]
            pbest_fit[better] = fit[better]
            g = int(np.argmin(pbest_fit))
            if pbest_fit[g] < gbest_fit:
                gbest_fit = float(pbest_fit[g])
                gbest_pos = pbest_pos[g].copy()

        self.last_fitness = gbest_fit
        return gbest_pos

    # ---------------------------------------------------------------
    # interface expected by main.py's rollout
    # ---------------------------------------------------------------

    def act(self, sequence, deterministic=False):
        """
        `sequence` is [batch, time, features]; only the last timestep
        is used. The planner is memoryless by construction -- it
        replans from the current observation every step, which is what
        makes it model-predictive rather than a policy.
        """
        last = sequence[:, -1, :].detach().cpu().numpy()
        out = np.empty((last.shape[0], 2), dtype=np.float32)
        for i in range(last.shape[0]):
            row = last[i]
            patch = row[:self.patch_dim]
            scalars = row[self.patch_dim:]
            soc = float(scalars[4])
            elevation = float(scalars[6])
            sun_usable = float(scalars[7])
            out[i] = self.plan(patch, soc, sun_usable, elevation)

        action = torch.as_tensor(out, device=sequence.device)
        zeros = torch.zeros(last.shape[0], device=sequence.device)
        return action, action, zeros, zeros

    def forward(self, sequence):
        return self.act(sequence)[0]


# Half-cell nudge. rint uses banker's rounding, so an action at exactly
# cx - 0.5 can land in cell cx-1 and be scored against the WRONG cell's
# potential. Without this the "exact" solver was returning costs the
# swarm could beat, which is a contradiction in terms and was the tell
# that the enumeration was off by half a cell.
_CELL_EPS = 1e-4


class ExactPolicy(PSOPolicy):
    """
    The same planner with the search replaced by exhaustive
    enumeration.

    Subclassed rather than copied so the fitness function, the model,
    the observation decoding and the horizon are shared by
    construction. The ONLY difference is how the action is found,
    which is what makes the pair a clean isolation of search quality:
    any gap between this arm and the swarm is the optimiser, and any
    gap between this arm and the learned arms is the planner's
    formulation -- one-step, greedy, no lookahead.

    WHY THIS IS GENUINELY EXACT

    The patch lookup rounds the action to an integer cell, so the
    action space is effectively the (2*VIEW_DISTANCE+1)^2 reachable
    cells. Within a cell the potential is constant and only the move
    cost varies, and the move cost is monotone in |dx| and |dy|
    separately -- so the best action reaching a given cell is the one
    with the smallest magnitude that still rounds into it. Enumerating
    one action per cell therefore covers the true optimum, not an
    approximation of it. 1681 candidates in one vectorised pass, ~242
    microseconds, roughly 20x faster than the converged swarm.

    A NOTE ON THE EXPECTED RESULT

    A one-step greedy planner has no lookahead: it cannot position
    itself now for sun that arrives in an hour, or spend energy early
    to avoid a deficit later. On a 720-step episode with a moving sun
    and a depleting battery, that should cost it against a policy
    trained on discounted return -- which is the interesting
    hypothesis this arm tests. But it is a hypothesis. If the exact
    solver matches or beats the learned arms, the honest reading is
    that the task is myopically solvable and the learned arms' margin
    over a known model is smaller than claimed. Nothing here is tuned
    to produce either outcome; the fitness function is the cost MDP's
    own terms and the search is exhaustive.
    """

    def plan(self, patch, soc, sun_usable, elevation):
        vd = self.view_distance
        cx, cy = np.meshgrid(np.arange(-vd, vd + 1), np.arange(-vd, vd + 1))
        cx = cx.ravel()
        cy = cy.ravel()
        dx = np.sign(cx) * np.maximum(np.abs(cx) - 0.5 + _CELL_EPS, 0.0) / MAX_MOVE_PER_STEP
        dy = np.sign(cy) * np.maximum(np.abs(cy) - 0.5 + _CELL_EPS, 0.0) / MAX_MOVE_PER_STEP
        cand = np.stack([dx, dy], axis=1)
        fit = self._cost(patch, soc, sun_usable, elevation, cand)
        i = int(np.argmin(fit))
        self.last_fitness = float(fit[i])
        return cand[i]
