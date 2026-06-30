import math
import torch
import torch.nn as nn
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


class ChebyshevTransformer(nn.Module):
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

        for m in self.modules():

            if isinstance(m, nn.Linear):

                chaotic_xavier_init(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):

                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, sequence):
        sequence = sequence.to(torch.float32)
        x = self.encoder(self.input_projection(sequence))[:, -1]
        # bounded mean: action is normalized relative displacement in [-1, 1]
        x = torch.clamp(x, -10.0, 10.0)
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
