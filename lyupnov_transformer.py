import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Soft bounds for the state-dependent log_std head (see distribution()).
# Same numeric range as the earlier hard clamp (std in ~[0.135, 1.65]),
# but enforced with a smooth tanh squash instead of a hard clamp so
# there's no dead-gradient zone at the edges.
# LOG_STD_MIN lowered -2.0 -> -4.0.
#
# At -2.0, sigma could never fall below exp(-2) = 0.135, which puts a
# FLOOR of mean |a| = 0.168 on the squashed action -- 3.35 cells
# commanded every step, forever. Holding position needs |a| < 0.05.
# The policy was structurally incapable of parking regardless of what
# the reward said, which is consistent with the evaluation: |a| < 0.05
# occurred 0 times in 720 steps and the minimum observed was 0.0715.
#
#   LOG_STD_MIN   sigma_min   min mean |a|   cells commanded
#      -2.0         0.135        0.168            3.35
#      -3.0         0.050        0.062            1.25
#      -4.0         0.018        0.023            0.46
#
# -4.0 gives room to park with margin. It does not FORCE a small
# sigma -- the entropy bonus and RAW_LOG_STD_TARGET still set the
# equilibrium; this only stops the clamp from ruling parking out.
LOG_STD_MIN = -4.0
LOG_STD_MAX = 0.5

# Threshold for the raw_log_std regularizer (see evaluate_actions()'s
# raw_log_std_reg). Below this, the reg term applies zero force,
# handing off final convergence to the entropy bonus -- the entropy
# bonus's own calibrated target (see target_entropy below) works out
# to an equilibrium near raw_log_std=0, so this just needs to get
# comfortably close, not all the way to exactly 0.
RAW_LOG_STD_TARGET = 0.3


class LyapunovTransformerActorCritic(nn.Module):
    """
    Lyapunov Transformer Actor-Critic (LTAC)

    Outputs
    -------
    Actor:
        Continuous Gaussian policy

    Critic:
        PPO value estimate

    Lyapunov:
        Positive-definite energy function V(s)

    Dynamics:
        Predicts next latent representation

    Energy:
        Compresses transformer latent into an energy space
        for smoother Lyapunov estimation.
    """

    def __init__(
            self,
            view_dist,
            scalar_dim=7,
            action_dim=2,
            sequence_length=12,
            d_model=128,  # 128
            energy_dim=64,  # 64
            nhead=4,
            num_layers=2,
            dim_feedforward=256,  # 256
            # Dropout disabled.
            #
            # Rollout log-probs are recorded under model.eval()
            # (dropout off) while update() runs under model.train()
            # (dropout on), so with dropout > 0 the importance ratio
            # is never 1 even on the FIRST minibatch, where lp and
            # oldlp come from identical parameters and it must be.
            #
            # That asymmetry made approx_kl average 2.07 against
            # TARGET_KL=0.015, so the early stop fired after the first
            # minibatch in 499 of 500 updates -- silently discarding
            # the entire PPO epoch/minibatch budget and reverting to
            # roughly one gradient step per rollout.
            #
            # The effect is amplified by tanh saturation: the squash
            # correction carries a log(1 - tanh(z)^2) term whose
            # derivative diverges as |tanh| -> 1, so at the boundary
            # any perturbation produces enormous log-prob swings.
            #
            # With a 1440-sample batch from two correlated episodes,
            # dropout was not buying meaningful regularization anyway.
            dropout=0.0,
    ):
        super().__init__()

        ############################################################
        # Observation dimensions
        ############################################################

        self.patch_dim = (2 * int(view_dist) + 1) ** 2
        self.input_dim = self.patch_dim + scalar_dim

        self.d_model = d_model
        self.sequence_length = sequence_length
        self.energy_dim = energy_dim
        self.action_dim = action_dim

        ############################################################
        # Input Projection
        ############################################################

        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        ############################################################
        # Learnable Positional Encoding
        ############################################################

        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )

        ############################################################
        # Transformer Encoder
        ############################################################

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            # norm_first=True (Pre-LN) only normalizes each sublayer's
            # input -- the residual stream itself is never renormalized
            # unless a closing norm is supplied here. Without it, the
            # scale of the encoder's output is whatever falls out of
            # accumulated residual additions at initialization, not
            # something the architecture controls -- and that
            # unnormalized latent feeds straight into the actor head.
            norm=nn.LayerNorm(d_model),
            # The nested-tensor fast path only supports norm_first=False,
            # so it can never actually be used here -- disabling the
            # attempt just silences the "enable_nested_tensor is True,
            # but..." warning without changing any computation.
            enable_nested_tensor=False,
        )

        ############################################################
        # Attention Pooling
        ############################################################

        self.attention_pool = nn.Linear(d_model, 1)

        ############################################################
        # Energy Encoder
        #
        # Compresses the transformer latent into an energy manifold.
        ############################################################

        self.energy_encoder = nn.Sequential(

            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),

            nn.Linear(d_model, energy_dim),
            nn.LayerNorm(energy_dim),
            nn.GELU(),
        )

        ############################################################
        # Actor Head
        ############################################################

        self.actor = nn.Sequential(

            nn.Linear(d_model, d_model),
            nn.GELU(),

            nn.Linear(d_model, action_dim),
        )

        ############################################################
        # Critic Head
        ############################################################
        L1 = int(2 * d_model)
        L3 = int(d_model / 2)
        self.critic = nn.Sequential(
            nn.Linear(d_model, L1),
            nn.GELU(),

            nn.Linear(L1, d_model),
            nn.GELU(),

            nn.Linear(d_model, L3),
            nn.GELU(),

            nn.Linear(L3, 1),
        )

        ############################################################
        # Lyapunov Head
        #
        # Softplus guarantees V(s) >= 0.
        ############################################################

        self.lyapunov = nn.Sequential(

            nn.Linear(energy_dim, energy_dim),
            nn.GELU(),

            nn.Linear(energy_dim, 1),

            nn.Softplus(),
        )

        ############################################################
        # Latent Dynamics Model
        #
        # Predicts z(t+1)
        ############################################################

        self.dynamics = nn.Sequential(

            nn.Linear(d_model, d_model),
            nn.GELU(),

            nn.Linear(d_model, d_model),
        )

        ############################################################
        # Barrier Head
        #
        # Vector-valued Control Barrier Function
        ############################################################

        self.barrier = nn.Sequential(

            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),

            nn.Linear(d_model, 64),
            nn.GELU(),

            # Battery
            # Boundary
            # Vegetation
            # Velocity
            # Communication
            nn.Linear(64, 5),
        )

        ############################################################
        # Gaussian exploration
        #
        # Fixed, non-state-dependent log_std -- not a function of
        # latent. A state-dependent version (via a log_std_head
        # reading latent) was tried first, on the theory that the
        # policy could stay exploratory in unresolved states and
        # sharpen up in resolved ones. In practice this created a
        # persistent coupling problem: latent keeps getting reshaped
        # every update by policy/value/Lyapunov/barrier/dynamics
        # losses, and whatever read from it kept getting perturbed by
        # that reshaping as a side effect, regardless of what the
        # entropy bonus and hinge regularizer (raw_log_std_reg, see
        # evaluate_actions()) were actually trying to do to it.
        # Detaching latent's gradient only cut one direction of that
        # coupling (log_std_head's own gradient no longer reshaped
        # the encoder) -- it did nothing about the forward-pass
        # direction (log_std_head still read a constantly-moving
        # target), and a full run confirmed the oscillation persisted
        # regardless. Removing latent from the picture entirely
        # removes the coupling by construction: there's no forward-
        # pass dependency left to be a moving target, and the entropy
        # bonus vs. hinge regularizer tug-of-war (which was already
        # comfortably favorable in isolation, 2-20x in various
        # measurements) can now actually play out the way that math
        # predicted. Also cheaper than the Linear layer it replaces.
        ############################################################

        self.log_std_param = nn.Parameter(torch.zeros(action_dim))

        # Entropy temperature (alpha) is NOT learned here anymore --
        # see ALPHA_START/ALPHA_END/ALPHA_DECAY_EPISODES in main.py.
        # A learned (SAC-style) alpha was tried first, but its
        # convergence rate is inherently sensitive to gradient noise,
        # and it never had a strong guarantee of reaching a low
        # enough value within this run's fixed ~500-update budget
        # (SAC, where this technique comes from, typically runs with
        # far more updates via a replay buffer). Repeated coefficient
        # increases on the competing regularizer bought a better
        # starting position each time but the same deceleration kept
        # reappearing closer to the target -- a symptom of fighting a
        # convergence-dependent process rather than removing the
        # dependency on convergence at all. A schedule tied directly
        # to episode number has no such dependency: alpha's value at
        # any point in training is known in advance, regardless of
        # gradient noise or how the shared encoder's representation
        # happens to be evolving underneath it.

        ############################################################
        # Weight Initialization
        ############################################################

        self.apply(self._initialize_weights)

        ############################################################
        # Policy head: small-gain output layer
        ############################################################
        # Standard Xavier on the actor's final Linear(d_model, 2)
        # gives weight std ~0.124, so with unit-scale activations
        # raw_mean is drawn with std ~1.40 -- the policy is SATURATED
        # AT INITIALIZATION:
        #
        #     mean |tanh(raw_mean)|          0.654
        #     fraction |tanh| > 0.95         19.2%
        #     fraction |tanh| > 0.99          5.9%
        #     squash correction log(1-t^2)   mean -1.27, min -11.8
        #
        # (Measured mean_abs_action at episode 2 of the previous run
        # was 0.686, matching this almost exactly.)
        #
        # Three consequences, all observed:
        #   1. The log-prob is dominated by the squash correction,
        #      whose derivative diverges as |tanh| -> 1, so log-probs
        #      swing wildly for small parameter changes.
        #   2. MEAN_SATURATION_COEF then applies a large, consistent
        #      gradient pulling the mean back toward zero -- a real
        #      policy change that the KL check reads as divergence.
        #   3. One Adam step therefore moves raw_mean far enough to
        #      exceed the emergency KL threshold on minibatch 2, so
        #      updates logged MB 2/24.
        #
        # Scaling the final policy layer down by 0.01 (standard PPO
        # practice) starts raw_mean at std ~0.014, i.e. tanh ~ 0 with
        # the full action range reachable via the sampling noise
        # rather than via a pre-committed saturated mean.
        POLICY_OUTPUT_GAIN = 0.01
        policy_out = [m for m in self.actor if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            policy_out.weight.mul_(POLICY_OUTPUT_GAIN)
            if policy_out.bias is not None:
                policy_out.bias.zero_()

    ############################################################
    # Weight Initialization
    ############################################################

    def _initialize_weights(self, module):
        """
        Xavier initialization for all Linear layers.
        LayerNorm defaults are preserved.
        """

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        ############################################################
        # Shared Transformer Encoder
        ############################################################

    def encode(self, sequence):
        """
        Converts an observation sequence into a latent vector.

        Parameters
        ----------
        sequence : Tensor
            [batch, sequence_length, input_dim]

        Returns
        -------
        latent : Tensor
            [batch, d_model]
        """

        sequence = sequence.float()

        x = self.input_projection(sequence)

        # Add learnable positional encoding
        x = x + self.position_embedding[:, :x.size(1)]

        # Transformer encoding
        x = self.encoder(x)

        attention_scores = self.attention_pool(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        latent = torch.sum(
            attention_weights * x,
            dim=1
        )

        return latent

    ############################################################
    # Energy Projection
    ############################################################

    def energy_representation(self, latent):
        """
        Maps the latent state into an energy manifold.

        This generally produces a smoother Lyapunov function
        than using the raw transformer latent.
        """

        return self.energy_encoder(latent)

    ############################################################
    # Lyapunov Function
    ############################################################

    def lyapunov_value(self, latent):
        """
        Computes V(s).

        Returns
        -------
        Tensor
            Positive Lyapunov value.
        """

        energy = self.energy_representation(latent)

        V = self.lyapunov(energy)

        return V.squeeze(-1)

    ############################################################
    # Barrier Function
    ############################################################

    def barrier_value(self, latent):
        """
        Returns

        [battery,
         boundary,
         vegetation,
         velocity,
         communication]
        """

        return self.barrier(latent)

    ############################################################
    # Latent Dynamics
    ############################################################

    def predict_next_latent(self, latent):
        """
        Predicts z(t+1).

        Used later for latent dynamics regularization.
        """

        return self.dynamics(latent)

    ############################################################
    # Convenience Method
    ############################################################

    def encode_state(self, sequence):
        """
        Computes everything shared by every network head.

        Returns
        -------
        latent
        energy
        lyapunov
        """

        latent = self.encode(sequence)

        # The auxiliary heads (energy/Lyapunov, barrier, latent
        # dynamics) read a DETACHED latent, so their losses train
        # only their own parameters and never reshape the shared
        # encoder.
        #
        # This was the original intent -- `latent_aux` was already
        # being computed here and then not used, so every auxiliary
        # gradient was flowing back into the trunk. Two concrete
        # problems that caused:
        #
        #   1. dynamics_loss has a trivial collapse solution. The
        #      target is detached, so the only gradient path is
        #      through dynamics(latent) -> encoder, and the cheapest
        #      way to make the latent predictable is to make it
        #      CONSTANT. A constant latent means a state-independent
        #      actor, which is exactly what the trained policy turned
        #      out to be.
        #   2. barrier_loss and lyapunov_penalty were both an order
        #      of magnitude larger than policy_loss, so the trunk was
        #      being shaped by constraint satisfaction rather than by
        #      reward.
        #
        # With the detach in place the trunk is trained by the policy
        # and value losses only; the auxiliary heads still learn to
        # describe the representation, they just no longer get to
        # dictate it.
        latent_aux = latent.detach()

        energy = self.energy_representation(latent_aux)
        V = self.lyapunov(energy).squeeze(-1)
        barrier = self.barrier(latent_aux)
        next_latent = self.dynamics(latent_aux)
        return latent, energy, V, barrier, next_latent

    ############################################################
    # Forward Pass
    ############################################################

    def forward(self, sequence):
        """
        Forward pass through the LTAC network.

        Parameters
        ----------
        sequence : Tensor
            [batch, sequence_length, input_dim]

        Returns
        -------
        raw_mean : Tensor
            Unbounded location parameter of the pre-squash Gaussian.
            NOT itself the action -- see `act()`/`fast_act()`, which
            apply `tanh` once, to the sample drawn from this Gaussian,
            so the bound to (-1, 1) is enforced exactly once and
            training/execution always see the identical value.

        raw_log_std : Tensor
            Unbounded, per-state log_std output. Squashed into
            [LOG_STD_MIN, LOG_STD_MAX] in `distribution()`, not here.

        critic : Tensor
            PPO value estimate

        lyapunov : Tensor
            Positive Lyapunov function

        latent : Tensor
            Transformer latent representation

        next_latent : Tensor
            Predicted next latent representation
        """

        latent, energy, V, barrier, next_latent = self.encode_state(sequence)

        ##########################################
        # Actor
        #
        # No tanh here -- squashing happens once,
        # at the sampled point, in act()/fast_act().
        ##########################################

        raw_mean = self.actor(latent)

        # Fixed parameter, not read from latent at all -- no forward-
        # pass dependency on the encoder, so nothing about its
        # ongoing reshaping (from policy/value/Lyapunov/barrier/
        # dynamics losses) can perturb this. Expanded to match
        # raw_mean's batch dimension so everything downstream
        # (distribution(), the smooth tanh bound, the hinge
        # regularizer) sees the same per-sample shape as before and
        # needs no further changes.
        raw_log_std = self.log_std_param.unsqueeze(0).expand(raw_mean.shape[0], -1)

        ##########################################
        # Critic
        ##########################################

        critic = self.critic(latent).squeeze(-1)

        return (
            raw_mean,
            raw_log_std,
            critic,
            V,
            barrier,
            latent,
            next_latent,
        )

    ############################################################
    # Gaussian Distribution
    ############################################################

    def distribution(self, sequence):
        """
        Returns the pre-squash policy distribution together with all
        auxiliary outputs. This Normal lives in unbounded space --
        actions are obtained by sampling from it and then applying
        `tanh` (see `_squash`), never by clamping.
        """

        (
            raw_mean,
            raw_log_std,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
        ) = self(sequence)

        # Smooth bound instead of a hard clamp: tanh saturates
        # gracefully and always has a nonzero gradient pointing back
        # toward the interior, so raw_log_std can never get stuck at
        # a dead boundary the way a plain .clamp() can.
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(raw_log_std) + 1.0
        )

        dist = Normal(
            raw_mean,
            log_std.exp(),
        )

        return (
            dist,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
            raw_log_std,
        )

    ############################################################
    # Tanh Squashing
    #
    # Applied once, to a sample z from the pre-squash Gaussian,
    # never to the mean and never via a hard clamp. Returns the
    # bounded action together with the log-det-Jacobian correction
    # needed to turn log N(z) into a correct log-density on
    # tanh(z). Using this consistently in act()/fast_act() and
    # evaluate_actions() means the value scored for training and
    # the value sent to the environment are always literally the
    # same tensor -- there is nothing else to keep in sync.
    ############################################################

    _TANH_EPS = 1e-6

    def _squash(self, z):
        action = torch.tanh(z)
        correction = torch.log(1.0 - action.pow(2) + self._TANH_EPS)
        return action, correction

    ############################################################
    # Action Selection
    ############################################################
    def fast_act(self,
                 sequence
                 ):
        latent = self.encode(sequence)

        raw_mean = self.actor(latent)

        # Deterministic action: squash the mean directly. tanh
        # guarantees this lands in (-1, 1) on its own -- no clamp
        # step, so there's no separate "training value" vs
        # "execution value" to reconcile.
        action, _ = self._squash(raw_mean)

        return (
            action,
            raw_mean,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def act(
            self,
            sequence,
            deterministic=False,
    ):
        """
        Samples an action.

        Returns
        -------
        action : Tensor
            Bounded to (-1, 1) by construction (tanh), no clamp.
        raw_z : Tensor
            The pre-squash Gaussian sample. Store this in the
            rollout buffer (not `action`) -- `evaluate_actions`
            needs it to recompute an exact, correctly-corrected
            log_prob for this same physical action under updated
            policy parameters.
        log_prob
        critic
        lyapunov
        """

        (
            dist,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
            _,
        ) = self.distribution(sequence)

        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample()

        action, correction = self._squash(z)

        log_prob = (
            (dist.log_prob(z) - correction)
            .sum(dim=-1)
        )

        return (
            action,
            z,
            log_prob,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
        )

    ############################################################
    # PPO Evaluation
    ############################################################

    def evaluate_actions(
            self,
            sequence,
            actions,
    ):
        """
        Evaluates previously sampled actions.

        Used during PPO optimization.

        `actions` here must be `z`, the pre-squash Gaussian sample
        returned by `act()` -- not `tanh(z)`. The physical action
        actually executed was `tanh(z)`; passing `z` back in lets us
        recompute `log N(z; new_mean, new_std) - correction(z)`,
        which is the exact, Jacobian-corrected log-probability of
        that same physical action under the updated policy. There is
        no clamp anywhere in this path, so nothing here needs to be
        kept in sync with a separately-clipped execution value.

        Entropy is reported for the pre-squash Gaussian, which is the
        usual practical proxy for the (harder to compute in closed
        form) entropy of the squashed distribution -- it still serves
        its purpose as an exploration bonus.

        `mean_std` and `mean_raw_log_std` are included purely for
        logging (exploration level during training) -- neither plays
        any role in any loss. `mean_raw_log_std` in particular is the
        pre-tanh log_std_param value, logged directly rather than
        inferred backward from Std, so saturation of the smooth bound
        can be confirmed (or ruled out) from a real number instead of
        a guess.

        `raw_log_std_reg` (NOT detached) is added to the loss in
        main.py. Its gradient reaches log_std_param without passing
        through tanh's saturating derivative at all -- unlike the
        entropy bonus or the reward-driven policy gradient, it can't
        be drowned out by noise in either of those pathways.

        This is a hinge penalty (relu(|raw_log_std| - RAW_LOG_STD_TARGET)),
        not a plain L2 penalty on raw_log_std^2. That distinction
        matters: an L2 penalty's gradient is proportional to
        raw_log_std itself, so it weakens exactly as raw_log_std
        approaches the target -- precisely when the entropy bonus's
        own gradient (proportional to 1-tanh(x)^2, which *grows* as
        x shrinks) has the most relative leverage. In practice this
        showed up as repeated deceleration: doubling the old L2
        coefficient bought a better starting position but the same
        slowdown re-emerged closer to the target, just as the two
        forces converged again. A hinge penalty instead applies
        *constant* force everywhere above the target and zero below
        it -- it doesn't taper as raw_log_std approaches the target,
        and it hands off cleanly (zero added force, no fighting) once
        below it, leaving the entropy bonus's own calibrated
        equilibrium (near raw_log_std=0) to handle final convergence.
        """

        (
            dist,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
            raw_log_std,
        ) = self.distribution(sequence)

        # The pre-squash mean. Returned so main.py can apply a
        # saturation penalty to it: nothing else in this system
        # constrains |raw_mean|, and once tanh(raw_mean) reaches
        # +-0.99 the environment can no longer distinguish one
        # raw_mean from another, so the reward gradient vanishes
        # while the log-prob term keeps pushing the mean further
        # out. That runaway is what collapsed exploration in the
        # previous run (actions pinned at dx=-0.99 for all 720
        # steps) even though the logged pre-squash Std looked
        # perfectly healthy at ~0.47.
        raw_mean = dist.mean

        z = actions
        _, correction = self._squash(z)

        log_probs = (
            (dist.log_prob(z) - correction)
            .sum(dim=-1)
        )

        entropy = (
            dist.entropy()
            .sum(dim=-1)
        )

        mean_std = dist.stddev.mean().detach()
        mean_raw_log_std = raw_log_std.mean().detach()
        raw_log_std_reg = F.relu(raw_log_std.abs() - RAW_LOG_STD_TARGET).mean()

        return (
            log_probs,
            entropy,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
            mean_std,
            mean_raw_log_std,
            raw_log_std_reg,
            raw_mean,
        )

    ############################################################
    # Next-State Evaluation
    ############################################################

    def evaluate_next(self, next_sequence):
        """
        Computes only the quantities needed for the NEXT state:
        V(s') and the actual next latent.

        `evaluate_transition` below runs the encoder twice (once on
        the current sequence, once on the next), but `evaluate_actions`
        has already computed everything for the current sequence --
        V(s) and predicted_next_latent are both in its return tuple.
        Calling both therefore encodes the current sequence twice per
        optimizer step for identical results.

        That was tolerable at one gradient step per rollout. With PPO
        epochs and minibatches it is not: the waste is multiplied by
        epochs * minibatches. Use this together with `evaluate_actions`
        instead -- two encoder passes per minibatch (states, next_states)
        rather than three.
        """

        (
            _,
            _,
            _,
            V_next,
            _,
            next_latent,
            _,
        ) = self(next_sequence)

        return V_next, next_latent

    ############################################################
    # Transition Evaluation
    ############################################################

    def evaluate_transition(
            self,
            current_sequence,
            next_sequence,
    ):
        """
        Computes Lyapunov quantities for a transition.

        Returns
        -------
        V(s)

        V(s')

        ΔV

        predicted_next_latent

        actual_next_latent
        """

        (
            _,
            _,
            _,
            V_current,
            _,
            latent,
            predicted_next,
        ) = self(current_sequence)

        (
            _,
            _,
            _,
            V_next,
            _,
            next_latent,
            _,
        ) = self(next_sequence)

        delta_V = V_next - V_current

        return (
            V_current,
            V_next,
            delta_V,
            predicted_next,
            next_latent.detach(),
        )

    ############################################################
    # Barrier Transition
    ############################################################

    def evaluate_barrier(
            self,
            current_sequence,
            next_sequence,
    ):
        """
        Returns

        h(s)

        h(s')

        barrier difference
        """

        (
            _,
            _,
            _,
            _,
            barrier_now,
            _,
            _,
        ) = self(current_sequence)

        (
            _,
            _,
            _,
            _,
            barrier_next,
            _,
            _,
        ) = self(next_sequence)

        return (

            barrier_now,

            barrier_next,

            barrier_next - barrier_now,
        )