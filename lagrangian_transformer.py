#!/usr/bin/env python
"""
Lagrangian PPO: the standard safe-RL comparator.

WHY THIS IS THE RIGHT BASELINE

The paper's claim is that the usual remedies enforce their constraint
on something other than the value the policy maximizes. A Lagrangian
method is the cleanest instance: it carries a SECOND value function,
regressing a separate cost-return, and couples it to the policy only
through a scalar dual variable. Nothing requires the two critics to
agree, and neither is a certificate of anything on its own.

So this arm is not a strawman to beat -- it is the construction the
merged critic is being contrasted against, and it should be tuned to
actually satisfy its own constraint. Two diagnostics decide whether it
was: the achieved J_c (did it meet the budget?) and lambda at
convergence (did the constraint stay active, or go slack and leave
plain PPO behind?). Both are logged.

THE CONSTRAINT

    J_c = E[ sum_t gamma^t 1(b_t < b_floor) ]   subject to   J_c <= d

with b_floor = 0.20 and d = 1.25 by default. The floor is chosen from
the data rather than picked round: episodes start at SoC 0.30-0.35 and
dip before harvesting, so gamma = 0.99 weights exactly the window
where charge is lowest, and the discounted J_c of the existing
converged arms is

    floor  0.10 -> 0.07    slack; lambda decays and this becomes PPO
    floor  0.15 -> 0.73    barely active
    floor  0.20 -> 2.50    active, with real spread across seeds
    floor  0.25 -> 9.69    binds so hard the task objective is drowned

0.20 is where the constraint is genuinely active without dominating.
The budget d = 1.25 is half the mean the existing arms achieve there:
any budget above 2.50 is met by changing nothing.

THE DUAL UPDATE

    lambda <- max(0, lambda + eta (J_c_hat - d))

on ascent, and the policy objective becomes

    (A_reward - lambda * A_cost) / (1 + lambda)

The normalization by (1 + lambda) keeps the gradient scale stable as
lambda grows; without it the effective learning rate drifts with the
multiplier and the run diverges once the constraint binds hard.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TransformerActorCritic

# Charge below which a step counts against the budget.
COST_FLOOR = float(os.environ.get("LTAC_COST_FLOOR", "0.20"))
# Discounted budget d.
COST_BUDGET = float(os.environ.get("LTAC_COST_BUDGET", "1.25"))
# Dual ascent step size.
LAMBDA_LR = float(os.environ.get("LTAC_LAMBDA_LR", "0.02"))
# Ceiling on the multiplier, and the number of updates before the dual
# is allowed to move at all.
#
# BOTH DEFAULTS ARE SET FROM A FAILED RUN, NOT FROM TASTE.
#
# At LAMBDA_MAX = 50 and no warm-up, seed 1 saturated the multiplier
# by update 48. The objective (A_r - lam A_c)/(1 + lam) then weights
# the reward at 1/51 = 2%, so the policy stopped learning to navigate:
# 2,579 episodes at a 70-83% death rate with no trend, and 1/10
# surviving at evaluation. J_c stayed near 20 against a budget of
# 1.25, which drove lambda higher still.
#
# That is a deadlock rather than slow convergence. J_c is high because
# the vehicle dies, it dies because it never learned to find sun, it
# never learned because lambda suppressed the reward signal, and
# lambda is high because J_c is high.
#
# The warm-up breaks the loop at its only entry point: the policy gets
# WARMUP updates of undisturbed task signal before the constraint
# begins to bind. The lower ceiling bounds the damage if the dual runs
# away anyway -- at LAMBDA_MAX = 5 the reward still carries 1/6 = 17%
# of the objective, which is enough to keep learning.
LAMBDA_MAX = float(os.environ.get("LTAC_LAMBDA_MAX", "5.0"))
# Counted in EPISODES: main.py passes the episode index,
# there being no update counter in scope at the call site.
LAMBDA_WARMUP = int(os.environ.get("LTAC_LAMBDA_WARMUP", "300"))


class LagrangianTransformerActorCritic(TransformerActorCritic):
    """PPO with a separate cost critic and a learned dual variable.

    Subclasses the REWARD baseline, not the cost one. That is not a
    detail: CostTransformerActorCritic puts a negated softplus on the
    value head, forcing V^pi <= 0, which is correct for a cost MDP and
    wrong here. This arm maximizes reward under a constraint, so its
    reward critic must be free to take either sign -- clamping it
    non-positive would leave the value regression fighting the
    architecture from the first update.

    Building on the reward baseline also isolates what the comparison
    is for: this arm and "normal" differ by exactly the cost critic and
    the multiplier, so any gap between them is the constraint
    mechanism and nothing else.
    """

    def __init__(self, *args, scalar_dim=8, d_model=128, **kwargs):
        super().__init__(*args, scalar_dim=scalar_dim, d_model=d_model,
                         **kwargs)
        self.scalar_dim = int(scalar_dim)

        # The second critic. Deliberately a separate head with its own
        # regression target: that separation IS the baseline. It reads
        # the shared encoder's latent without training it (see
        # cost_value below), so the encoder is shaped by the policy and
        # reward critic exactly as in Standard PPO.
        L1 = max(64, d_model // 2)
        self.cost_critic = nn.Sequential(
            nn.Linear(d_model, L1), nn.GELU(),
            nn.Linear(L1, L1), nn.GELU(),
            nn.Linear(L1, 1),
        )

        # log_lambda rather than lambda, so the projection onto
        # non-negatives is free and the ascent is multiplicative near
        # zero -- a plain clamp at 0 stalls whenever the constraint is
        # satisfied, because the gradient there points below the
        # boundary and is discarded.
        self.register_buffer("log_lambda",
                             torch.tensor(float(torch.log(torch.tensor(0.1)))))

    # -- the multiplier -------------------------------------------------
    @property
    def lam(self):
        return float(torch.exp(self.log_lambda).clamp(max=LAMBDA_MAX))

    @torch.no_grad()
    def update_lambda(self, j_c, update_idx=None):
        """
        Dual ascent on the constraint violation.

        Done in log space: lambda <- lambda * exp(eta (J_c - d)). Near
        lambda = 0 this moves proportionally rather than additively, so
        a satisfied constraint decays the multiplier smoothly instead
        of pinning it at the boundary.

        Frozen for the first LAMBDA_WARMUP episodes. A constraint that
        binds before the policy can satisfy it drives the multiplier to
        its ceiling and takes the task signal with it; the dual has
        nothing useful to chase until J_c reflects a policy that has
        started to work.
        """
        if update_idx is not None and update_idx < LAMBDA_WARMUP:
            return self.lam
        viol = float(j_c) - COST_BUDGET
        self.log_lambda.add_(LAMBDA_LR * viol)
        self.log_lambda.clamp_(min=-11.5, max=float(
            torch.log(torch.tensor(LAMBDA_MAX))))
        return self.lam

    # -- the cost value -------------------------------------------------
    #
    # The cost critic READS the shared latent but does not TRAIN it.
    #
    # Measured on seed 1: with the cost regression back-propagating into
    # the encoder, the arm sat at 62-65% deaths through a 300-episode
    # warm-up with lambda frozen at 0.1 -- i.e. while it was within ~10%
    # of Standard PPO, which reaches <=10% deaths by episode ~300 with
    # the same encoder. The constraint signal was reshaping the
    # representation the policy depends on before the constraint had
    # any say in the policy objective.
    #
    # Detaching keeps the arm plain Lagrangian PPO: separate reward and
    # cost value functions are the standard arrangement (Safety Gym's
    # PPO-Lagrangian uses separate networks outright), and with the
    # encoder trained only by the policy and reward critic, this arm
    # differs from Standard PPO by the cost critic's advantage and the
    # multiplier alone -- which is the comparison it exists for.
    def cost_value(self, latent):
        return self.cost_critic(latent.detach()).squeeze(-1)

    def cost_value_only(self, sequence):
        # No graph through the encoder is needed: the latent is detached
        # anyway, so building one would only cost memory.
        with torch.no_grad():
            latent = self.encode(sequence)
        return self.cost_value(latent)

    @staticmethod
    def cost_signal(soc):
        """Per-step cost: 1 while the charge is below the floor."""
        return (soc < COST_FLOOR).astype("float32")

    # -- the objective ---------------------------------------------------
    @staticmethod
    def combine_advantages(adv_r, adv_c, lam):
        """
        (A_reward - lambda A_cost) / (1 + lambda).

        The denominator is not cosmetic: without it the policy-gradient
        magnitude scales with lambda, so the effective learning rate
        changes whenever the constraint tightens and the run destabilises
        exactly when safety starts to matter.
        """
        return (adv_r - lam * adv_c) / (1.0 + lam)

    def forward(self, sequence):
        """
        The parent's 4-tuple, unchanged.

        act(), distribution() and evaluate_actions() all unpack exactly
        four values, and main.py's rollout unpacks four. Appending the
        cost value here would break every one of them; it is reached
        through cost_value_only() instead, which is what the rollout
        and the value regression call.
        """
        return super().forward(sequence)
