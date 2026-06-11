import numpy as np
import torch
import torch.nn as nn


class PSOPolicy(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=2,
        swarm_size=30,
        inertia=0.7,
        c1=1.5,
        c2=1.5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_weights = (
            input_dim * 64 +
            64 +
            64 * output_dim +
            output_dim
        )

        self.swarm_size = swarm_size
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

        self.positions = np.random.randn(
            swarm_size,
            self.num_weights
        ) * 0.1

        self.velocities = np.zeros_like(self.positions)

        self.personal_best_pos = self.positions.copy()
        self.personal_best_fit = np.full(
            swarm_size,
            -np.inf
        )

        self.global_best_pos = self.positions[0].copy()
        self.global_best_fit = -np.inf

        self.current_particle = 0

    def _decode(self, particle):

        idx = 0

        w1_size = self.input_dim * 64
        W1 = particle[idx:idx+w1_size].reshape(
            self.input_dim,
            64
        )
        idx += w1_size

        b1 = particle[idx:idx+64]
        idx += 64

        w2_size = 64 * self.output_dim
        W2 = particle[idx:idx+w2_size].reshape(
            64,
            self.output_dim
        )
        idx += w2_size

        b2 = particle[idx:idx+self.output_dim]

        return W1, b1, W2, b2

    def forward(self, x):

        particle = self.positions[self.current_particle]
        W1, b1, W2, b2 = self._decode(particle)

        x_np = x.detach().cpu().numpy()

        h = np.tanh(x_np @ W1 + b1)
        y = h @ W2 + b2

        return torch.tensor(
            y,
            dtype=torch.float32,
            device=x.device
        )

    def evaluate_particle(self, particle_idx, reward):

        if reward > self.personal_best_fit[particle_idx]:

            self.personal_best_fit[particle_idx] = reward
            self.personal_best_pos[particle_idx] = (
                self.positions[particle_idx].copy()
            )

        if reward > self.global_best_fit:

            self.global_best_fit = reward
            self.global_best_pos = (
                self.positions[particle_idx].copy()
            )

    def update_swarm(self):

        r1 = np.random.rand(*self.positions.shape)
        r2 = np.random.rand(*self.positions.shape)

        self.velocities = (
            self.inertia * self.velocities
            + self.c1 * r1 *
            (self.personal_best_pos - self.positions)
            + self.c2 * r2 *
            (self.global_best_pos - self.positions)
        )

        self.positions += self.velocities