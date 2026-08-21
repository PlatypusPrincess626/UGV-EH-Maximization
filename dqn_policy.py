"""
DQN arm: value-based control, with the Lyapunov construction moved onto
the Q head.

WHY THIS IS A BETTER FIT FOR THE THESIS THAN IT LOOKS

In the PPO arms the certified object is V(s) = critic(latent), and the
policy is a separate actor trained against advantages derived from it.
A reviewer can reasonably ask what the certificate constrains: the
critic is certified, the actor is not, and the connection between them
runs through GAE.

With DQN there is no actor. The policy IS

    a*(s) = argmin_a  V_cost(s, a),      V_cost(s, a) = -Q(s, a)

so the object that is certified and the object that selects actions
are the same function. The construction also survives composition
intact:

    sign      Softplus on the Q head gives Q <= 0, hence
              V_cost = -Q >= 0 structurally, exactly as in the cost arm.
    Lipschitz max_a over a FINITE action set is 1-Lipschitz, so
              L(state -> max_a Q) <= L(state -> Q). The spectral bound
              on the Q body carries through the argmax unchanged.
    zero      V_cost = 0 at the goal remains structural.

So the merge the cost arm attempts -- critic and Lyapunov function as
one object -- is stricter here, not weaker.

DISCRETISATION

DQN needs a finite action set. The environment is already effectively
discrete: an action is scaled by MAX_MOVE_PER_STEP and the patch
lookup rounds to an integer cell, so the true action space is the
(2*VIEW_DISTANCE+1)^2 = 1681 reachable cells that ExactPolicy
enumerates.

1681 Q-values is a legal output layer but a poor learning target --
most cells are never visited and their Q-values never get a gradient.
The default instead is a polar set: N_DIRECTIONS headings x
N_MAGNITUDES step lengths, plus an explicit STAY action.

    8 x 3 + 1 = 25 actions

STAY is separate rather than emerging from a small magnitude because
staying still is the behaviour the cost MDP most needs to express --
c_move is exactly zero only at rest, and a policy that can only choose
"move a little" pays movement cost forever.

Set LTAC_DQN_ACTIONS=full to enumerate all reachable cells instead.

WHAT IS AND IS NOT IMPLEMENTED HERE

Implemented and self-contained: the network, the action set, the
epsilon schedule, the replay buffer, the target network, and a Double
DQN update step.

NOT implemented: the wiring into main.py's training loop. That loop is
PPO-shaped end to end -- compute_batch builds GAE advantages,
update() computes ratios, clipping and an approximate KL, and the
certification block reads V from the critic. None of that applies to
an off-policy value method, and quietly reusing it would produce a run
that looks like DQN and is not. See INTEGRATION at the bottom for the
specific call sites.
"""

import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cost_transformer import CostTransformerActorCritic

# Action set geometry.
N_DIRECTIONS = int(os.environ.get("LTAC_DQN_DIRS", "8"))
N_MAGNITUDES = int(os.environ.get("LTAC_DQN_MAGS", "3"))
ACTION_MODE = os.environ.get("LTAC_DQN_ACTIONS", "polar").strip().lower()

# Epsilon-greedy schedule, in environment steps.
EPS_START = float(os.environ.get("LTAC_DQN_EPS_START", "1.0"))
EPS_END = float(os.environ.get("LTAC_DQN_EPS_END", "0.05"))
EPS_DECAY_STEPS = int(os.environ.get("LTAC_DQN_EPS_DECAY", "150000"))

REPLAY_CAPACITY = int(os.environ.get("LTAC_DQN_REPLAY", "200000"))
BATCH_SIZE = int(os.environ.get("LTAC_DQN_BATCH", "256"))
TARGET_SYNC = int(os.environ.get("LTAC_DQN_TARGET_SYNC", "2000"))
LEARN_START = int(os.environ.get("LTAC_DQN_LEARN_START", "10000"))


def build_action_set(view_distance, max_move):
    """
    [n_actions, 2] array of (dx, dy) in [-1, 1].

    Index 0 is always STAY, so a policy that has learned nothing still
    has "do nothing" available at a known index -- useful when reading
    an action histogram.
    """
    if ACTION_MODE == "full":
        vd = int(view_distance)
        cx, cy = np.meshgrid(np.arange(-vd, vd + 1), np.arange(-vd, vd + 1))
        acts = np.stack([cx.ravel(), cy.ravel()], axis=1) / max_move
        # Move STAY to the front.
        stay = np.argmin(np.abs(acts).sum(axis=1))
        order = [stay] + [i for i in range(len(acts)) if i != stay]
        return acts[order].astype(np.float32)

    acts = [(0.0, 0.0)]
    for m in range(1, N_MAGNITUDES + 1):
        r = m / N_MAGNITUDES
        for d in range(N_DIRECTIONS):
            th = 2.0 * math.pi * d / N_DIRECTIONS
            acts.append((r * math.cos(th), r * math.sin(th)))
    return np.array(acts, dtype=np.float32)


class DQNPolicy(CostTransformerActorCritic):
    """
    Shares the cost arm's trunk exactly -- subclassed, not copied, so a
    given seed produces a bit-identical encoder and the DQN arm is
    comparable to the PPO arms on everything except the learning rule.

    The inherited actor and log_std_param are left in place but receive
    no gradient. They cost a little memory and keep the RNG
    consumption order identical, which is worth more than removing
    them.
    """

    def __init__(self, *args, view_distance=20, max_move=20.0,
                 d_model=128, **kwargs):
        super().__init__(*args, d_model=d_model, **kwargs)

        self.register_buffer(
            "action_set",
            torch.as_tensor(build_action_set(view_distance, max_move)))
        self.n_actions = self.action_set.shape[0]

        # Q head. Mirrors critic_body's shape so the two arms have
        # comparable capacity, but outputs one value per action.
        L1, L3 = int(2 * d_model), int(d_model / 2)
        self.q_body = nn.Sequential(
            nn.Linear(d_model, L1), nn.GELU(),
            nn.Linear(L1, d_model), nn.GELU(),
            nn.Linear(d_model, L3), nn.GELU(),
            nn.Linear(L3, self.n_actions),
        )
        self.apply_spectral_to_q()

    def apply_spectral_to_q(self):
        """
        Spectral-normalise the Q body if the parent did so for its
        critic, so the certified Lipschitz bound applies to the same
        object in both arms.
        """
        if not getattr(self, "spectral_critic", True):
            return
        from cost_transformer import SPECTRAL_C, _LipschitzScale
        try:
            from torch.nn.utils.parametrizations import spectral_norm
            from torch.nn.utils import parametrize
        except ImportError:                                # pragma: no cover
            return
        for layer in self.q_body:
            if isinstance(layer, nn.Linear):
                spectral_norm(layer, n_power_iterations=5)
                if SPECTRAL_C != 1.0:
                    parametrize.register_parametrization(
                        layer, "weight", _LipschitzScale(SPECTRAL_C))

    # ---------------------------------------------------------------
    # value
    # ---------------------------------------------------------------

    def q_values(self, sequence):
        """
        [batch, n_actions]. Softplus-negated so Q <= 0, matching the
        cost MDP where every return is non-positive, and making
        V_cost = -Q >= 0 structural rather than learned.
        """
        raw = self.q_body(self.encode(sequence))
        return -F.softplus(raw, beta=float(self.softplus_beta))

    def value_only(self, sequence):
        """
        V(s) = max_a Q(s, a) -- the value of acting greedily.

        Keeps the signature the other arms use, so diagnostics.py
        scores this arm with no branching. V_cost = -value >= 0 as
        before, and because max over a finite set is 1-Lipschitz the
        spectral bound on q_body carries through unchanged.
        """
        return self.q_values(sequence).max(dim=-1).values

    # ---------------------------------------------------------------
    # acting
    # ---------------------------------------------------------------

    def epsilon(self, step):
        frac = min(max(step / max(EPS_DECAY_STEPS, 1), 0.0), 1.0)
        return EPS_START + frac * (EPS_END - EPS_START)

    def act(self, sequence, deterministic=False, step=0):
        """
        Returns (action, action_index, log_prob, value) so main.py's
        rollout can call this exactly as it calls the PPO arms.

        log_prob is returned as zeros: there is no behaviour policy
        density here, and anything downstream that uses it -- PPO
        ratios above all -- is meaningless for this arm and must not
        be run. The second slot carries the ACTION INDEX rather than a
        pre-squash z, because that is what the replay buffer and the
        TD target need.
        """
        q = self.q_values(sequence)
        idx = q.argmax(dim=-1)

        if not deterministic:
            eps = self.epsilon(step)
            explore = torch.rand(idx.shape, device=idx.device) < eps
            if explore.any():
                rand_idx = torch.randint(
                    self.n_actions, idx.shape, device=idx.device)
                idx = torch.where(explore, rand_idx, idx)

        action = self.action_set[idx]
        value = q.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        zeros = torch.zeros_like(value)
        return action, idx, zeros, value


class EpisodeReplay:
    """
    Replay over whole episodes, sampling (episode, t) index pairs.

    WHY NOT ONE WINDOW PER TRANSITION

    The obvious buffer stores the [sequence_length, features] window
    the encoder consumes, per transition. At 32 x 1689 floats that is
    216 KB each, so a 200k-capacity buffer needs 43 GB. Consecutive
    windows overlap in 31 of their 32 rows, so almost all of it is
    duplicated.

    Storing each episode's observation stream ONCE and reconstructing
    windows by slicing costs 6.8 KB per step instead:

        per-transition windows   200k -> 43.2 GB
        per-episode streams      200k ->  1.35 GB

    The stream includes the leading history the first window needs, so
    obs[t : t+L] is the state at step t and obs[t+1 : t+L+1] is the
    next state, with no special case at the episode boundary.
    """

    def __init__(self, capacity=REPLAY_CAPACITY, seq_len=32):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = deque()
        self.n_steps = 0

    def push_episode(self, obs_stream, action_idx, rewards, dones):
        """
        obs_stream : [T + seq_len, features]; the first seq_len rows
                     are the history preceding step 0.
        """
        obs_stream = np.asarray(obs_stream, dtype=np.float32)
        T = len(rewards)
        if T == 0 or obs_stream.shape[0] < T + self.seq_len:
            return
        self.episodes.append((obs_stream,
                              np.asarray(action_idx, dtype=np.int64),
                              np.asarray(rewards, dtype=np.float32),
                              np.asarray(dones, dtype=np.float32)))
        self.n_steps += T
        while self.n_steps > self.capacity and len(self.episodes) > 1:
            old = self.episodes.popleft()
            self.n_steps -= len(old[2])

    def sample(self, batch_size, device):
        lengths = [len(e[2]) for e in self.episodes]
        total = sum(lengths)
        picks = np.random.randint(0, total, size=batch_size)
        # Map a flat index to (episode, t) without materialising the
        # index list -- the buffer holds hundreds of thousands of steps
        # and rebuilding that array every sample would dominate.
        bounds = np.cumsum(lengths)
        ep_ids = np.searchsorted(bounds, picks, side="right")
        starts = np.concatenate([[0], bounds[:-1]])
        ts = picks - starts[ep_ids]

        L = self.seq_len
        s, s2, a, r, d = [], [], [], [], []
        for e, t in zip(ep_ids, ts):
            obs, ai, rw, dn = self.episodes[e]
            s.append(obs[t:t + L])
            s2.append(obs[t + 1:t + 1 + L])
            a.append(ai[t]); r.append(rw[t]); d.append(dn[t])

        return (torch.as_tensor(np.stack(s), device=device),
                torch.as_tensor(np.asarray(a), dtype=torch.long, device=device),
                torch.as_tensor(np.asarray(r), dtype=torch.float32, device=device),
                torch.as_tensor(np.stack(s2), device=device),
                torch.as_tensor(np.asarray(d), dtype=torch.float32, device=device))

    def __len__(self):
        return self.n_steps


class DQNTrainer:
    """
    Double DQN update.

    DOUBLE, not vanilla: the online network selects the next action and
    the target network evaluates it. Vanilla DQN's max over a
    target network systematically overestimates, and here that bias
    has a specific cost -- V_cost = -max_a Q is the certified quantity,
    so an overestimated max is a certificate that claims more margin
    than exists. The overestimation would land directly on the number
    the paper reports.
    """

    def __init__(self, model, optimizer, gamma, device):
        self.model = model
        self.opt = optimizer
        self.gamma = gamma
        self.device = device
        self.target = None
        self.steps = 0
        self.sync_target()

    def sync_target(self):
        import copy
        self.target = copy.deepcopy(self.model).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    def learn(self, replay, batch_size=BATCH_SIZE):
        if len(replay) < max(LEARN_START, batch_size):
            return None

        s, a, r, s2, done = replay.sample(batch_size, self.device)

        q = self.model.q_values(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online = self.model.q_values(s2)
            best = next_online.argmax(dim=1, keepdim=True)
            next_q = self.target.q_values(s2).gather(1, best).squeeze(1)
            target = r + self.gamma * (1.0 - done) * next_q

        # Huber rather than MSE: the cost MDP's death charge is an
        # order of magnitude larger than any ordinary step cost, and
        # under MSE those transitions dominate the gradient of every
        # minibatch they appear in.
        loss = F.smooth_l1_loss(q, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 8.0)
        self.opt.step()

        self.steps += 1
        if self.steps % TARGET_SYNC == 0:
            self.sync_target()

        return {"dqn_loss": float(loss.item()),
                "dqn_q_mean": float(q.mean().item()),
                "dqn_target_mean": float(target.mean().item())}


# =====================================================================
# INTEGRATION -- what main.py needs, and why it is not done here.
#
# main.py's training loop is PPO-shaped end to end. Reusing it would
# produce a run labelled DQN that is not DQN, which is the failure mode
# worth avoiding above all others.
#
# The specific call sites:
#
#   1. model.act(s) at the rollout (~line 2888) returns a 4-tuple. This
#      class matches that signature, but act() also needs the global
#      step for the epsilon schedule -- pass it, or epsilon stays at
#      EPS_START forever and the arm never stops exploring.
#
#   2. The rollout stores raw_a as a pre-squash z. Here slot 2 is an
#      ACTION INDEX. Anything that treats it as a continuous action --
#      evaluate_actions, the tanh correction -- must not run.
#
#   3. compute_batch() builds GAE advantages and update() computes
#      ratios, clipping and approx_kl. None of it applies. The DQN
#      branch should call DQNTrainer.learn() per environment step (or
#      every k steps) instead of calling update() per episode batch.
#
#   4. The certification block reads V from the critic. It works
#      unchanged if pointed at value_only(), which returns
#      max_a Q(s, a) -- but note the semantics shift: V is now the
#      value of acting GREEDILY, not of following the behaviour policy.
#      Under epsilon-greedy those differ, and the certificate is about
#      the greedy policy. Worth stating explicitly rather than leaving
#      implicit.
#
#   5. diagnostics.py works as-is: it only calls value_only().
#
#   6. EpisodeReplay stores each episode's observation stream once
#      and slices windows at sample time: ~1.35 GB at 200k steps,
#      against 43 GB for per-transition windows.
# =====================================================================
