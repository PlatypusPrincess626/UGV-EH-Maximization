import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


def chebyshev_sequence(length, degree=4, seed=0.723456, discard=200):
    """
    Generate a bounded chaotic sequence using the Chebyshev map.

        x_(n+1) = cos(degree * arccos(x_n))

    Output is normalized to [-1,1].
    """

    x = np.clip(seed, -0.999999, 0.999999)

    seq = []

    for _ in range(discard + length):
        x = math.cos(degree * math.acos(x))
        x = np.clip(x, -0.999999, 0.999999)
        seq.append(x)

    seq = np.asarray(seq[discard:], dtype=np.float32)

    # Numerical safety
    seq = np.nan_to_num(seq)

    return seq


def chaotic_xavier_init(tensor, order=4, seed=None):
    """
        Xavier initialization whose ordering comes from a Chebyshev chaotic map.
    """
    if seed is None:
        seed = np.random.uniform(-0.95, 0.95)

    if tensor.ndim < 2:
        return

    fan_out, fan_in = tensor.shape[:2]

    limit = math.sqrt(6.0 / (fan_in + fan_out))

    seq = chebyshev_sequence(
        tensor.numel(),
        degree=order,
        seed=seed
    )

    weights = torch.from_numpy(seq).reshape(tensor.shape)

    with torch.no_grad():
        tensor.copy_(weights.to(tensor.device, tensor.dtype) * limit)


class ChebyshevLyapunovTransformerActorCritic(nn.Module):
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
        d_model=128,
        energy_dim=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
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
        )

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

        self.critic = nn.Sequential(

            nn.Linear(d_model, d_model),
            nn.GELU(),

            nn.Linear(d_model, 1),
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
        ############################################################

        self.log_std = nn.Parameter(
            torch.full((action_dim,), -0.5)
        )

        ############################################################
        # Weight Initialization
        ############################################################

        self.initialize_weights()

    ############################################################
    # Weight Initialization
    ############################################################

    def initialize_weights(self):
        """
        Xavier initialization for all Linear layers.
        LayerNorm defaults are preserved.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                chaotic_xavier_init(module.weight)

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

        # Last token summarizes recent history
        latent = x[:, -1]

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

        energy = self.energy_representation(latent)

        V = self.lyapunov(energy).squeeze(-1)

        barrier = self.barrier(latent)

        return latent, energy, V, barrier

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
        mean : Tensor
            Mean action in [-1,1]

        critic : Tensor
            PPO value estimate

        lyapunov : Tensor
            Positive Lyapunov function

        latent : Tensor
            Transformer latent representation

        next_latent : Tensor
            Predicted next latent representation
        """

        latent, energy, V, barrier = self.encode_state(sequence)

        ##########################################
        # Actor
        ##########################################

        mean = torch.tanh(
            self.actor(latent)
        )

        ##########################################
        # Critic
        ##########################################

        critic = self.critic(latent).squeeze(-1)

        ##########################################
        # Dynamics
        ##########################################

        next_latent = self.predict_next_latent(latent)

        return (
            mean,
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
        Returns the policy distribution together with all
        auxiliary outputs.
        """

        (
            mean,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
        ) = self(sequence)

        dist = Normal(
            mean,
            self.log_std.exp().expand_as(mean),
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
    # Action Selection
    ############################################################
    def fast_act(self,
                 sequence
                 ):
        sequence = sequence.float()

        x = self.input_projection(sequence)

        # Add learnable positional encoding
        x = x + self.position_embedding[:, :x.size(1)]

        # Transformer encoding
        x = self.encoder(x)

        # Last token summarizes recent history
        latent = x[:, -1]

        mean = torch.tanh(
            self.actor(latent)
        )

        dist = Normal(
            mean,
            self.log_std.exp().expand_as(mean),
        )

        raw_action = dist.mean

        action = raw_action.clamp(
            -0.999,
            0.999,
        )

        return (
            action,
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
        action
        log_probability
        critic
        lyapunov
        latent
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
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        action = raw_action.clamp(
            -0.999,
            0.999,
        )

        log_prob = (
            dist.log_prob(action)
            .sum(dim=-1)
        )

        return (
            action,
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
        """

        (
            dist,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
        ) = self.distribution(sequence)

        log_probs = (
            dist.log_prob(actions)
            .sum(dim=-1)
        )

        entropy = (
            dist.entropy()
            .sum(dim=-1)
        )

        return (
            log_probs,
            entropy,
            critic,
            lyapunov,
            barrier,
            latent,
            next_latent,
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
            V_current,
            _,
            latent,
            predicted_next,
        ) = self(current_sequence)

        (
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
            barrier_now,
            _,
            _,
        ) = self(current_sequence)

        (
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