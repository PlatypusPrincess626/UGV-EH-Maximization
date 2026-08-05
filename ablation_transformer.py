"""
Ablation head for the cost arm: the cost reward with the Lyapunov
construction removed.

WHAT QUESTION THIS ANSWERS

The cost arm changed two things at once relative to the reward arms:
the reward formulation (a cost MDP) and the critic construction (a
sign-constrained, spectral-normalized head that doubles as the
Lyapunov function). Its results -- task parity, a critic fitting ~1.5x
better relative to its own scale, v_agreement rising 0.335 -> 0.50,
and an analytic certificate going from unsatisfiable to alpha_hat ~
0.0012 -- are consistent with the construction doing the work AND with
a well-shaped cost reward doing it. Nothing in the sweep separates
them.

This subclass holds the reward fixed and removes the construction, so
the difference is attributable.

WHAT VARIES, AND WHAT DOES NOT

Subclassed rather than copied, deliberately. Every part of the trunk,
the actor, the log_std parameterization, the layer sizes, the
initialization and -- critically -- the ORDER in which they consume
the RNG is inherited unchanged, so a given seed produces a
bit-identical encoder in every arm. A copied file would drift.

Two knobs, giving three arms:

    variant          head                spectral norm   isolates
    ---------------  ------------------  --------------  -----------
    cost             beta*softplus       yes             (reference)
    cost_linear      unconstrained       yes             the head
    cost_plain       unconstrained       no              both

`cost_plain` alone answers the headline question: does the Lyapunov
construction contribute, or is it the reward? `cost_linear` splits
that into the sign constraint versus the Lipschitz bound, which
matters because the spectral norm is the thing compressing the
critic's dynamic range and is NOT used by the analytic certificate.

WHAT THE ABLATED CRITIC GIVES UP

    V_cost >= 0          no longer structural. Under a cost MDP every
                         return is <= 0, so a well-fit critic should
                         land there anyway -- but nothing enforces it,
                         and nothing enforces it at states the rollout
                         never visited. v_critic_min going negative is
                         the signal, and it is already logged.

    V_cost = 0 at goal   no longer structural.

    L <= 1               gone entirely when spectral_critic is False.
                         Every region claim built on the Lipschitz
                         bound goes with it.

That is the point: if these are load-bearing, removing them should
degrade v_agreement and the certified alpha_hat while leaving task
performance roughly intact. If task and certificate both hold up, the
construction was decoration and the reward was doing the work.

INITIALIZATION

The parent starts the critic at V_cost ~ +1.0, near the measured
early-training value scale, by offsetting the final bias. It does that
in PRE-nonlinearity units, where beta*softplus(+1.0) ~ +1.0 so the
bias is +1.0. Here value = raw directly, so the same V_cost needs a
bias of -1.0. Matching the initial V_cost rather than the raw bias
keeps the arms comparable -- otherwise the ablation would start 2.2
units of value away from the reference and the first updates would
differ for a reason that has nothing to do with the construction.
"""

import torch
import torch.nn as nn

from cost_transformer import CostTransformerActorCritic

# V_cost at initialization, shared with the parent's CRITIC_BIAS_INIT.
# Sized near the early-training value scale (~1.23 measured at the
# first updates of cost seed 1).
CRITIC_INIT_VCOST = 1.0


class AblationTransformerActorCritic(CostTransformerActorCritic):
    """
    Cost-MDP actor-critic with an UNCONSTRAINED critic head.

    value = critic_body(latent), with no nonlinearity forcing a sign.
    V_cost = -value is still the quantity certified, so every
    certification metric in main.py applies unchanged -- it is just no
    longer guaranteed non-negative, which is exactly what is being
    tested.
    """

    def __init__(self, *args, spectral_critic=False, **kwargs):
        # softplus_beta is inherited but unused; beta_gain_target is
        # forced off so the parent's adaptive-beta machinery cannot
        # introduce a difference that has nothing to do with the head.
        kwargs.pop("beta_gain_target", None)
        super().__init__(*args, spectral_critic=spectral_critic,
                         beta_gain_target=None, **kwargs)

        self.spectral_critic = bool(spectral_critic)

        # Re-point the bias so the ablation starts at the same V_cost
        # as the reference arm. See INITIALIZATION above.
        critic_out = [m for m in self.critic_body
                      if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            if critic_out.bias is not None:
                critic_out.bias.fill_(-CRITIC_INIT_VCOST)

    def critic(self, latent):
        """
        Unconstrained. value may take either sign; V_cost = -value may
        go negative, and that is the observation, not a failure to
        handle.
        """
        return self.critic_body(latent).squeeze(-1)

    @torch.no_grad()
    def adapt_beta(self, v_cost, quantile=0.10):
        """
        No-op. beta shapes a Softplus knee that this head does not
        have. Overridden rather than left inherited so main.py can
        call it uniformly across arms without branching.
        """
        return float(self.softplus_beta)
