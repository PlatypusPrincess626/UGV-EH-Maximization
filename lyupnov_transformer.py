import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Soft bounds for the state-dependent log_std head (see distribution()).
# Same numeric range as the earlier hard clamp (std in ~[0.135, 1.65]),
# but enforced with a smooth tanh squash instead of a hard clamp so
# there's no dead-gradient zone at the edges.
LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.5


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
            dropout=0.1,
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
        # log_std is now a function of the latent (like the mean),
        # not a single global scalar -- the policy can stay
        # exploratory in states it hasn't resolved yet and sharpen
        # up in states it has, rather than PPO being forced to pick
        # one exploration level for every state at once.
        ############################################################

        self.log_std_head = nn.Linear(d_model, action_dim)

        ############################################################
        # Automatic Entropy Temperature (SAC-style)
        #
        # Replaces a fixed ENTROPY_COEF with a learned multiplier
        # that targets a specific entropy level directly: alpha
        # grows when entropy drifts below target_entropy, shrinks
        # when it's comfortably above. See update() in main.py for
        # the alpha_loss that trains this.
        ############################################################

        self.target_entropy = -float(action_dim)
        self.log_alpha = nn.Parameter(torch.zeros(1))

        ############################################################
        # Weight Initialization
        ############################################################

        self.apply(self._initialize_weights)

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

        latent_aux = latent.detach()
        energy = self.energy_representation(latent)
        V = self.lyapunov(energy).squeeze(-1)
        barrier = self.barrier(latent)
        next_latent = self.dynamics(latent)
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
        raw_log_std = self.log_std_head(latent)

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

        `mean_std` is included purely for logging (e.g. printing
        exploration level during training) -- it plays no role in
        any loss.
        """

        (
            dist,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
        ) = self.distribution(sequence)

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

        return (
            log_probs,
            entropy,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
            mean_std,
        )

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