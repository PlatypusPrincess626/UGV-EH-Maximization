import math
import torch
import torch.nn as nn
from torch.distributions import Normal

class TransformerActorCritic(nn.Module):
    """Temporal actor-critic. Input: [batch, sequence, flattened_patch + scalar_features]."""
    # dropout defaults to 0.0, not 0.1.
    #
    # main.py records rollout log-probs under model.eval() and runs
    # update() under model.train(). With dropout > 0 the importance
    # ratio is not 1 even on the first minibatch, where it must be --
    # on the lyapunov model this drove approx_kl to 2.07 against a
    # TARGET_KL of 0.03, so the KL early stop fired after one minibatch
    # and discarded the entire PPO epoch budget (round 2 / round 4).
    # The baseline shares that training loop and would hit the same
    # failure, which would look like an architectural result rather
    # than the bug it is.
    def __init__(self, view_dist, scalar_dim=7, action_dim=2, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.patch_dim = (2 * int(view_dist) + 1) ** 2
        self.input_dim = self.patch_dim + scalar_dim
        self.input_projection = nn.Sequential(nn.Linear(self.input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.actor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, action_dim))
        self.critic = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Small-gain init on the policy output layer, matching the
        # lyapunov model (round 10). Plain init gives the actor's final
        # Linear a weight std around 0.124, so with unit-scale
        # activations the pre-tanh mean is drawn with std ~1.4 and the
        # policy is SATURATED AT INITIALIZATION -- mean |tanh| 0.654,
        # 19% of samples past 0.95. That is a general property of the
        # architecture, not of the Lyapunov heads, and leaving it here
        # would handicap the baseline for a reason unrelated to the
        # comparison.
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)
            if self.actor[-1].bias is not None:
                self.actor[-1].bias.zero_()

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
