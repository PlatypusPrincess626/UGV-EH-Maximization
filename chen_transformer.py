import math
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def chen_sequence(length, a=35.0, b=3.0, c=28.0, dt=0.005, discard=500):
    """
    Generates a chaotic sequence using the Chen attractor.

    dx/dt = a(y-x)
    dy/dt = (c-a)x - xz + cy
    dz/dt = xy - bz
    """

    x = 0.11
    y = 0.17
    z = 0.23

    seq = []

    total = discard + length

    for _ in range(total):
        dx = a * (y - x)
        dy = (c - a) * x - x * z + c * y
        dz = x * y - b * z

        x += dx * dt
        y += dy * dt
        z += dz * dt

        seq.append(x)

    seq = np.asarray(seq[discard:])

    # Normalize to [-1,1]
    seq = (seq - seq.min()) / (seq.max() - seq.min())
    seq = seq * 2.0 - 1.0

    return seq


def chen_init_tensor(tensor):

    fan_in = tensor.size(1)
    fan_out = tensor.size(0)

    limit = math.sqrt(6.0/(fan_in+fan_out))

    seq = chen_sequence(tensor.numel())

    weights = torch.tensor(
        seq,
        dtype=tensor.dtype,
        device=tensor.device
    ).reshape(tensor.shape)

    with torch.no_grad():
        tensor.copy_(weights * limit)


class ChenTransformer(nn.Module):
    """Temporal actor-critic. Input: [batch, sequence, flattened_patch + scalar_features]."""
    def __init__(self, view_dist, scalar_dim=7, action_dim=2, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.patch_dim = (2 * int(view_dist) + 1) ** 2
        self.input_dim = self.patch_dim + scalar_dim
        self.input_projection = nn.Sequential(nn.Linear(self.input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.actor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, action_dim))
        self.critic = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.initialize_weights()

    def initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                chen_init_tensor(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):

                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sequence):
        sequence = sequence.to(torch.float32)
        x = self.encoder(self.input_projection(sequence))[:, -1]
        # bounded mean: action is normalized relative displacement in [-1, 1]
        return torch.tanh(self.actor(x)), self.critic(x).squeeze(-1)

    def distribution(self, sequence):
        mean, value = self(sequence)
        return Normal(mean, self.log_std.exp().expand_as(mean)), value

    def act(self, sequence, deterministic=False):
        dist, value = self.distribution(sequence)
        raw = dist.mean if deterministic else dist.rsample()
        action = raw.clamp(-0.999, 0.999)  # execution and log-prob use same bounded action
        return action, dist.log_prob(action).sum(-1), value

    def evaluate_actions(self, sequence, actions):
        dist, values = self.distribution(sequence)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), values
