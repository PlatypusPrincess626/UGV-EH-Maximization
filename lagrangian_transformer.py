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

from cost_transformer import CostTransformerActorCritic

# Charge below which a step counts against the budget.
COST_FLOOR = float(os.environ.get("LTAC_COST_FLOOR", "0.20"))
# Discounted budget d.
COST_BUDGET = float(os.environ.get("LTAC_COST_BUDGET", "1.25"))
# Dual ascent step size.
LAMBDA_LR = float(os.environ.get("LTAC_LAMBDA_LR", "0.02"))
# Ceiling on the multiplier. A runaway lambda silently turns the run
# into pure constraint satisfaction with no task signal, which looks
# like convergence failure rather than what it is.
LAMBDA_MAX = float(os.environ.get("LTAC_LAMBDA_MAX", "50.0"))


class LagrangianTransformerActorCritic(CostTransformerActorCritic):
    """PPO with a separate cost critic and a learned dual variable."""

    def __init__(self, *args, scalar_dim=8, d_model=128, **kwargs):
        super().__init__(*args, scalar_dim=scalar_dim, d_model=d_model,
                         **kwargs)
        self.scalar_dim = int(scalar_dim)

        # The second critic. Deliberately a separate head with its own
        # regression target: that separation IS the baseline. It shares
        # the encoder, exactly as the auxiliary-certificate baseline
        # does, so the only structural difference from the merged arm
        # is which object the constraint is stated on.
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
    def update_lambda(self, j_c):
        """
        Dual ascent on the constraint violation.

        Done in log space: lambda <- lambda * exp(eta (J_c - d)). Near
        lambda = 0 this moves proportionally rather than additively, so
        a satisfied constraint decays the multiplier smoothly instead
        of pinning it at the boundary.
        """
        viol = float(j_c) - COST_BUDGET
        self.log_lambda.add_(LAMBDA_LR * viol)
        self.log_lambda.clamp_(min=-11.5, max=float(
            torch.log(torch.tensor(LAMBDA_MAX))))
        return self.lam

    # -- the cost value -------------------------------------------------
    def cost_value(self, latent):
        return self.cost_critic(latent).squeeze(-1)

    def cost_value_only(self, sequence):
        return self.cost_value(self.encode(sequence))

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
        """Parent tuple, with the cost value appended."""
        latent = self.encode(sequence)
        raw_mean = self.actor(latent)
        raw_log_std = self.log_std_param.unsqueeze(0).expand(
            raw_mean.shape[0], -1)
        return (raw_mean, raw_log_std, self.critic(latent), latent,
                self.cost_value(latent))
