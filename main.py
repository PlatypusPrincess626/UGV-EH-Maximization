# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import math
import matplotlib.pyplot as plt
from pvlib import solarposition

# Import the refactored simulation modules
from environment import sim_env
from transformer import TransformerEncoder

# =====================================================================
# 1. HYPERPARAMETERS & CONFIGURATION (For easy tuning)
# =====================================================================
# Simulation Settings
SCENE = "test"  # Scenario setup name inside the environment matrix
NUM_SENSORS = 20  # Count of static ground nodes distributed on map
MAX_STEPS_PER_EPISODE = 720  # Upper frame ceiling limit per iteration
VIEW_DISTANCE = 20

# Hyperparameters
TOTAL_EPISODES = 1_000        # RL takes longer to converge than simple MSE
TRAIN_EVERY_X_EPISODES = 5
LEARNING_RATE = 5e-4
GAMMA = 0.99                # Discount factor for future battery rewards

# Transformer Model Specifications
D_MODEL = 64  # Internal feature width sizing
NHEAD = 4  # Multihead partition factor
NUM_LAYERS = 2  # Stack sequence sizing for encoder tracking
DIM_FEEDFORWARD = 256  # Dense inner width for position-wise networks
DROPOUT = 0.1  # Regulation dropout probability fraction

# Ensure hardware acceleration is leveraged if accessible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device execution target: {device}")

env = sim_env(SCENE, NUM_SENSORS, MAX_STEPS_PER_EPISODE)
env.set_view_dist(VIEW_DISTANCE)
model = TransformerEncoder(VIEW_DISTANCE, d_model=D_MODEL, num_layers=NUM_LAYERS,
                           dim_feedforward=DIM_FEEDFORWARD).to(device)

# Initialize log_std as a trainable parameter for 2D continuous actions (x, y)
log_std = torch.zeros(2, requires_grad=True, device=device)

# Define the optimizer to update both model parameters AND log_std
optimizer = optim.Adam(
    list(model.parameters()) + [log_std],
    lr=LEARNING_RATE
)

episode_step_counts = []
episode_rewards = []
all_batch_log_probs = []
all_batch_rewards = []

# =====================================================================
# 2. MAIN SIMULATION TRAINING LOOP
# =====================================================================
for episode in range(1, TOTAL_EPISODES + 1):
    # Reset environment components back to baseline parameters
    env.place_devices()
    env.ch.reset()

    ep_log_probs = []
    ep_rewards = []
    steps_taken = 0
    total_ep_reward = 0.0

    # Extract initial solar azimuth and zenith vectors for the initial state calculation
    ugv_x, ugv_y, _ = env.ch.get_position()
    lat_offset = ugv_x * env.stp
    long_offset = ugv_y * env.stp
    solpos = solarposition.get_solarposition(env.times[0], env.lat_center + lat_offset, env.long_center + long_offset)

    next_obs = env.get_obfuscation(ugv_x, ugv_y, 0, solpos['azimuth'].iloc[0], solpos['apparent_zenith'].iloc[0])

    for step in range(MAX_STEPS_PER_EPISODE):
        # B. Evaluate state and get stochastic action selection from Normal distribution
        model.eval()
        flat_obs = torch.tensor(next_obs.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_mean = model.forward(flat_obs).squeeze(0)

        std = torch.exp(log_std)
        dist_distribution = Normal(action_mean, std)

        # Sample continuous target coordinates and calculate its mathematical log probability
        action = dist_distribution.sample()

        # Re-enable gradient tracking context explicitly for the step log-prob evaluation
        model.train()
        action_mean_grad = model.forward(flat_obs).squeeze(0)
        dist_distribution_grad = Normal(action_mean_grad, std)
        log_prob = dist_distribution_grad.log_prob(action).sum()

        # Clip generated coordinates safely to maintain placement bounds on the grid map
        target_x = max(0.0, min(float(action[0].item()), float(env.dim - 1)))
        target_y = max(0.0, min(float(action[1].item()), float(env.dim - 1)))

        # C. Capture battery metrics baseline prior to advancing physics environment
        battery_before = env.ch.get_battery()

        # Advance simulation execution (updates location, draws physical current, samples solar irradiance)
        telemetry, next_obs = env.step_simulation(step, target_x, target_y)
        steps_taken += 1

        battery_after = env.ch.get_battery()

        # D. Reward Equation Calculation: Net capacity delta scaled for stable gradients
        reward = (battery_after - battery_before) * 100.0

        ep_log_probs.append(log_prob)
        ep_rewards.append(reward)
        total_ep_reward += reward

        # Break rollout if UGV runs completely out of battery
        if battery_after <= 0.0:
            break

    # Register metrics history
    episode_step_counts.append(steps_taken)
    episode_rewards.append(total_ep_reward)

    all_batch_log_probs.append(ep_log_probs)
    all_batch_rewards.append(ep_rewards)

    print(
        f"Episode {episode:03d}/{TOTAL_EPISODES:03d} | Steps Survived: {steps_taken:03d} | Final Battery: "
        f"{telemetry['battery_after']:.2f}% | Cumulative Reward: {total_ep_reward:.2f}")

    # =====================================================================
    # 3. POLICY GRADIENT UPDATE STEP (Triggered every X Episodes)
    # =====================================================================
    if episode % TRAIN_EVERY_X_EPISODES == 0:
        model.train()
        policy_loss = []

        for ep_lp, ep_r in zip(all_batch_log_probs, all_batch_rewards):
            # Compute discounted future returns tracking back from timeline limits
            discounted_rewards = []
            G = 0
            for r in reversed(ep_r):
                G = r + GAMMA * G
                discounted_rewards.insert(0, G)

            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)

            # Standardize array vectors to normalize variance scales across batches
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                            discounted_rewards.std() + 1e-6)

            for lp, return_val in zip(ep_lp, discounted_rewards):
                policy_loss.append(-lp * return_val)

        if len(policy_loss) > 0:
            optimizer.zero_grad()
            total_loss = torch.stack(policy_loss).sum()
            total_loss.backward()

            # Apply standard gradient clipping to protect stability across long horizons (720 steps max)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print(
                f" >>> [POLICY UPDATE] Complete. Gradients optimized for step slice batch. Total Loss: {total_loss.item():.4f}")

        # Evict processed rollout records from buffer memory structures
        all_batch_log_probs.clear()
        all_batch_rewards.clear()

# =====================================================================
# 4. POST-RUN EVALUATION GRAPHING
# =====================================================================
print("\nTraining complete. Generating evaluation analytics plot...")

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Steps per Episode', color=color)
ax1.plot(range(1, TOTAL_EPISODES + 1), episode_step_counts, color=color, alpha=0.6, label='Steps Limit')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Total Cumulative Reward', color=color)
ax2.plot(range(1, TOTAL_EPISODES + 1), episode_rewards, color=color, linestyle='--', alpha=0.6, label='Rewards Index')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('UGV Battery Maximization Training Metrics (Transformer Policy Gradient)')
fig.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

if __name__ == "__main__":
    run_rl_training()