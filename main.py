import csv, math, random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pvlib import solarposition
from environment import sim_env
from transformer import TransformerActorCritic
from chebyshev_transformer import ChebyshevTransformer
from lyupnov_transformer import LyapunovTransformerActorCritic
from chaotic_lyupnov_transformer import ChebyshevLyapunovTransformerActorCritic
from pso_policy import PSOPolicy
import datetime
import time

# ============================================================
# Set POLICY_TYPE = "transformer" or "pso"
POLICY_TYPE = "transformer"
if POLICY_TYPE == "transformer":
    # Set TRANSFORMER_VARIANT = "normal" or "chaotic" or "lyapunov"
    TRANSFORMER_VARIANT = "lyapunov"
    # Set TRANSFORMER_INIT = "normal" or "chaotic"
    if TRANSFORMER_VARIANT == "lyapunov":
        TRANSFORMER_INIT = "normal"
else:
    TRANSFORMER_VARIANT = "normal"
# ============================================================

TOTAL_EPISODES=1000; MAX_STEPS_PER_EPISODE=720; VIEW_DISTANCE=20
SEQUENCE_LENGTH=32; UPDATE_EVERY_EPISODES=2; GAMMA=.99; GAE_LAMBDA=.95
LR=3e-4; MAX_MOVE_PER_STEP=20.0; ENTROPY_COEF=.01; VALUE_COEF=.5

###############################################################
# Lyapunov Hyperparameters
###############################################################
LYAPUNOV_COEF = 0.25; DYNAMICS_COEF = 0.10; LATENT_COEF = 0.05
BARRIER_COEF = 0.20; ACTION_SMOOTHNESS = 0.01; LYAPUNOV_MARGIN = 0.01
BATTERY_MARGIN = 0.10; BOUNDARY_MARGIN = 0.10; VEGETATION_MARGIN = 0.05
VELOCITY_MARGIN = 0.05; COMM_MARGIN = 0.10; CLIP_EPS =0.20; LYAPUNOV_ALPHA = 0.02

###############################################################
# Convergence Monitoring / Checkpointing
#
# Rather than guessing an episode count up front: track moving
# averages of reward and of the Lyapunov/barrier penalties, save
# the best-so-far model whenever the reward average improves, and
# flag (and optionally stop on) convergence once the stability
# constraints are comfortably satisfied AND reward has stopped
# improving for a while. Convergence is only ever checked once the
# full-weight regime (ep>=300) is active -- checking earlier would
# just be measuring the pre-curriculum warmup, not real convergence.
###############################################################
REWARD_WINDOW = 50              # episodes averaged for the reward moving average
STABILITY_WINDOW = 20           # update() calls averaged for Lyapunov/barrier stability
CONVERGENCE_PATIENCE = 10       # consecutive stability checks with no new best reward
LYAPUNOV_STABLE_THRESHOLD = 2 * LYAPUNOV_MARGIN
BARRIER_STABLE_THRESHOLD = 0.05
CHECKPOINT_EVERY = 100          # periodic safety-net checkpoint, regardless of performance
AUTO_STOP_ON_CONVERGENCE = True # set False to just log/checkpoint without ending the run

# Format as YYYY-MM-DD_HH-MM-SS
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT=Path("rl_csv_"+timestamp); OUT.mkdir(exist_ok=True)

size = 2 * VIEW_DISTANCE + 1
y, x = np.mgrid[-VIEW_DISTANCE:VIEW_DISTANCE+1,
                -VIEW_DISTANCE:VIEW_DISTANCE+1]
sigma = 6.0
kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
kernel /= kernel.sum()
GAUSSIAN_KERNEL = kernel.flatten()

def log_status(ep, total_episodes, steps, avg_reward, final_batt, loss, is_eval=False):
    prefix = "[EVALUATION]" if is_eval else f"[Episode {ep}/{total_episodes}]"
    loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else loss
    print(f"{prefix} Steps: {steps:3} | Reward: {avg_reward:7.2f} | Battery: {final_batt:6.2f}% | Loss: {loss_str}")

def obs(env, x, y, yaw, step):
    sol=solarposition.get_solarposition(env.times[min(step,len(env.times)-1)], env.lat_center+y*env.stp, env.long_center+x*env.stp)
    patch=env.get_obfuscation(x,y,min(step,len(env.times)-1),sol.azimuth.iloc[0],sol.apparent_zenith.iloc[0]).flatten()
    potential = 1.0 - patch
    scalars=np.array([x/(env.dim-1),y/(env.dim-1),math.sin(yaw),math.cos(yaw),
                      env.ch.get_battery()/100, sol.azimuth.iloc[0]/360, sol.apparent_zenith.iloc[0]/90],np.float32)
    return np.concatenate([potential.astype(np.float32),scalars])

def seq_tensor(history, device):
    return torch.tensor(np.asarray(history),dtype=torch.float32,device=device).unsqueeze(0)

def reward_fn(before, after, telemetry, delta_batt):
    # Dense, scaled reward: energy gain, movement cost, survival, and boundary discouragement.
    potential_before = 1.0 - before
    potential_after = 1.0 - after

    score_before = float(np.dot(GAUSSIAN_KERNEL, potential_before))
    score_after = float(np.dot(GAUSSIAN_KERNEL, potential_after))

    directional_reward = 10 * (score_after - score_before)

    px, py, _ = telemetry["previous_position"]
    nx, ny, _ = telemetry["new_position"]

    distance = math.hypot(nx - px, ny - py)

    movement_penalty = 0.002 * distance

    battery_reward = 2.0 * delta_batt

    return float(directional_reward + battery_reward - movement_penalty)

def update(model,opt,rollouts,device, ep):
    # GAE advantages are normalized once across the complete rollout batch, not per episode.
    states=[]; next_states=[]; actions=[]; oldlp=[]; adv=[]; returns=[]
    for r in rollouts:
        gae=0.0; ret=0.0
        for i in reversed(range(len(r["rewards"]))):
            ret=r["rewards"][i]+GAMMA*ret
            nxt=r.get("bootstrap_value",0.0) if i==len(r["rewards"])-1 else r["values"][i+1]
            gae=r["rewards"][i]+GAMMA*nxt-r["values"][i]+GAMMA*GAE_LAMBDA*gae
            adv.insert(0,gae); returns.insert(0,ret)
        states+=r["states"]; next_states+=r["next_states"]; actions+=r["actions"]; oldlp+=r["logps"]
    adv = torch.tensor(adv,device=device,dtype=torch.float32); adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    returns=torch.tensor(returns,device=device,dtype=torch.float32); # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    actions=torch.tensor(np.asarray(actions),dtype=torch.float32,device=device)
    oldlp=torch.tensor(oldlp,device=device); states=torch.tensor(np.asarray(states),dtype=torch.float32,device=device)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device)
    if TRANSFORMER_VARIANT == "lyapunov":
        if ep < 100:
            lyap_weight = 0.0
            barrier_weight = 0.0
            dynamics_weight = 0.0
        elif ep < 300:
            lyap_weight = 0.10
            barrier_weight = 0.05
            dynamics_weight = 0.05
        else:
            lyap_weight = LYAPUNOV_COEF
            barrier_weight = BARRIER_COEF
            dynamics_weight = DYNAMICS_COEF

        (lp, entropy, values, lyapunov, barrier, latent, predicted_next, mean_std) = model.evaluate_actions(states,actions)
        (V_now, V_next, delta_V, predicted_latent, actual_latent) = model.evaluate_transition(states,next_states)

        ratio = torch.exp(lp - oldlp)
        approx_kl = (oldlp - lp).mean()
        clip_fraction = ((torch.abs(ratio - 1.0) > CLIP_EPS).float().mean())
        surr1 = ratio * adv
        surr2 = torch.clamp(
            ratio,
            1.0 - CLIP_EPS,
            1.0 + CLIP_EPS,
        ) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        lyapunov_penalty = F.relu(delta_V
                                  + LYAPUNOV_ALPHA * V_now
                                  + LYAPUNOV_MARGIN).mean()
        value_loss = F.mse_loss(values, returns.detach())
        dynamics_loss = F.mse_loss(
            predicted_latent,
            actual_latent.detach(),
        )
        battery_loss = torch.relu(BATTERY_MARGIN - barrier[:, 0]).mean()
        boundary_loss = torch.relu(BOUNDARY_MARGIN - barrier[:, 1]).mean()
        vegetation_loss = torch.relu(VEGETATION_MARGIN - barrier[:, 2]).mean()
        velocity_loss = torch.relu(VELOCITY_MARGIN - barrier[:, 3]).mean()
        communication_loss = torch.relu(COMM_MARGIN - barrier[:, 4]).mean()
        barrier_loss = (1.0 * battery_loss + 0.5 * boundary_loss + 0.25 * vegetation_loss
                        + 0.25 * velocity_loss + 0.75 * communication_loss)

        # Automatic entropy temperature (SAC-style): alpha adjusts
        # itself to hold entropy near model.target_entropy, instead
        # of relying on a fixed ENTROPY_COEF guess. alpha is detached
        # in the policy bonus (it should only shape the actor, not
        # get a gradient from the policy loss); entropy is detached
        # in alpha_loss (alpha should only be adjusted based on the
        # current entropy level, not try to change it directly).
        alpha = model.log_alpha.exp()
        alpha_loss = -(model.log_alpha * (model.target_entropy - entropy.detach())).mean()

        loss = (policy_loss
                + VALUE_COEF * value_loss
                - alpha.detach() * entropy.mean()
                + alpha_loss
                + lyap_weight * lyapunov_penalty
                + dynamics_weight * dynamics_loss
                + barrier_weight * barrier_loss)
        print(
            f"Policy {policy_loss:.4f} | "
            f"Value {value_loss:.4f} | " # 925016.2500
            f"Lyap {lyapunov_penalty:.4f} | "
            f"Dynamics {dynamics_loss:.4f} | "
            f"Barrier {barrier_loss:.4f} | "
            f"KL {approx_kl:.5f} | "
            f"CF {clip_fraction:.4f} | "
            f"Std {mean_std:.4f} | "
            f"Alpha {alpha.item():.4f}"
        )
        diag_lyap = float(lyapunov_penalty.item())
        diag_barrier = float(barrier_loss.item())
    else:
        lp,entropy,values=model.evaluate_actions(states,actions)
        # Single efficient batched actor-critic update; no duplicate forward pass per step.
        ratio = torch.exp(lp - oldlp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio,
                            1.0 - CLIP_EPS,
                            1.0 + CLIP_EPS) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        loss = (policy_loss
                + VALUE_COEF * F.mse_loss(values, returns)
                - ENTROPY_COEF * entropy.mean())
        diag_lyap = None
        diag_barrier = None
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return float(loss.item()), diag_lyap, diag_barrier

def run():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=sim_env("test",20,MAX_STEPS_PER_EPISODE); env.set_view_dist(VIEW_DISTANCE)

    # 1. Initialization Logic
    if POLICY_TYPE == "transformer":
        if TRANSFORMER_VARIANT == "lyapunov":
            if TRANSFORMER_INIT == "chaotic":
                model = ChebyshevLyapunovTransformerActorCritic(
                    view_dist=VIEW_DISTANCE,
                    sequence_length=SEQUENCE_LENGTH,
                ).to(device)
            else:
                model = LyapunovTransformerActorCritic(
                    view_dist=VIEW_DISTANCE,
                    sequence_length=SEQUENCE_LENGTH,
                ).to(device)
            actor_params = (list(model.actor.parameters()) + list(model.log_std_head.parameters()))

            critic_params = list(model.critic.parameters())

            alpha_params = [model.log_alpha]

            transformer_params = (
                    list(model.input_projection.parameters()) +
                    list(model.encoder.parameters()) +
                    list(model.attention_pool.parameters()) +
                    [model.position_embedding]
            )

            auxiliary_params = (
                    list(model.energy_encoder.parameters()) +
                    list(model.lyapunov.parameters()) +
                    list(model.barrier.parameters()) +
                    list(model.dynamics.parameters())
            )

            opt = optim.AdamW([{"params": transformer_params, "lr": 1e-3, "weight_decay":1e-5},
                               {"params": actor_params,       "lr": 3e-4, "weight_decay":1e-5},
                               {"params": critic_params,      "lr": 3e-4, "weight_decay":1e-5},
                               {"params": auxiliary_params,   "lr": 5e-5, "weight_decay":1e-5},
                               {"params": alpha_params,       "lr": 3e-4, "weight_decay":0.0},],
                              eps=1e-5,)
        elif TRANSFORMER_VARIANT == "chaotic":
            model = ChebyshevTransformer(VIEW_DISTANCE).to(device)
            opt = optim.Adam(model.parameters(), lr=LR)
        else:
            model = TransformerActorCritic(VIEW_DISTANCE).to(device)
            opt = optim.Adam(model.parameters(), lr=LR)
    else:
        # Assuming observation dimension is flattened patch size + scalars
        # Calculate based on your obs function (e.g., 20x20 patch + 7 scalars)
        input_dim = (VIEW_DISTANCE * 2 + 1) ** 2 + 7
        model = PSOPolicy(input_dim=input_dim).to(device)
        opt = None  # PSO does not use torch.optim

    total_inference_time = 0.0
    total_inference_steps = 0
    epfile=open(OUT/"episode_metrics.csv","w",newline="")
    epw=csv.DictWriter(epfile,fieldnames=["episode","steps","final_battery","total_reward","loss"])
    epw.writeheader()
    rollouts=[]

    # Convergence monitoring / checkpointing state
    ckpt_dir = OUT / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    reward_history = deque(maxlen=REWARD_WINDOW)
    lyap_history = deque(maxlen=STABILITY_WINDOW)
    barrier_history = deque(maxlen=STABILITY_WINDOW)
    best_avg_reward = -float("inf")
    checks_since_best = 0
    converged = False

    for ep in range(1,TOTAL_EPISODES+1):
        env.place_devices(); env.reset(); x,y,yaw=env.ch.get_position()
        h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH);total=0;
        r={"states":[],"next_states":[],"actions":[],"logps":[],"values":[],"rewards":[],"lyapunov":[], "barrier":[]};
        previous_action = None; smoothness = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            s=seq_tensor(h,device)

            start = time.perf_counter()
            # 2. Action Selection Logic
            if POLICY_TYPE == "transformer":
                with torch.no_grad():
                    if TRANSFORMER_VARIANT == "lyapunov":
                        (a, raw_a, lp, v, lyapunov, barrier, latent, predicted_next) = model.act(s)
                        current_action = a[0].detach().cpu().numpy()
                        if previous_action is None:
                            smoothness = 0.0
                        else:
                            smoothness = np.sum(((current_action - previous_action) / MAX_MOVE_PER_STEP) ** 2)
                        r["lyapunov"].append(lyapunov.cpu().numpy()); r["barrier"].append(barrier.cpu().numpy())
                    else:
                        a, lp, v = model.act(s)
                        lyapunov = None
                        barrier = None
                        current_action = a[0].detach().cpu().numpy()
            else:
                # Set the current particle for the forward pass
                model.current_particle = (ep - 1) % model.swarm_size
                with torch.no_grad():
                    a = model(s)
                    lp, v = torch.tensor(0.0), torch.tensor(0.0)  # Dummy values
                    current_action = a[0].detach().cpu().numpy()

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            total_inference_time += elapsed
            total_inference_steps += 1

            # normalized action -> local, physically scaled target; no global-coordinate clipping mismatch
            dx,dy=a[0].cpu().numpy()*MAX_MOVE_PER_STEP
            tx=float(np.clip(x+dx,0,env.dim-1)); ty=float(np.clip(y+dy,0,env.dim-1))

            sol = solarposition.get_solarposition(env.times[min(step, len(env.times) - 1)],
                                                  env.lat_center + y * env.stp, env.long_center + x * env.stp)
            before = env.get_obfuscation(x,y,min(step,len(env.times)-1),sol.azimuth.iloc[0],
                                         sol.apparent_zenith.iloc[0]).flatten()
            b_batt = env.ch.get_battery()

            tel, nxt= env.step_simulation(step,tx,ty)

            x_new, y_new, yaw_new = env.ch.get_position()
            sol = solarposition.get_solarposition(env.times[min(step, len(env.times) - 1)],
                                                  env.lat_center + y_new * env.stp, env.long_center + x_new * env.stp)
            after = env.get_obfuscation(x_new, y_new, min(step, len(env.times) - 1), sol.azimuth.iloc[0],
                                         sol.apparent_zenith.iloc[0]).flatten()
            aft_batt = env.ch.get_battery()

            rew=reward_fn(before, after, tel, aft_batt-b_batt) - ACTION_SMOOTHNESS*smoothness
            total+=rew; previous_action=current_action
            r["states"].append(np.asarray(h)); r["actions"].append(raw_a[0].cpu().numpy())
            r["logps"].append(lp.item()); r["values"].append(v.item()); r["rewards"].append(rew)
            x,y,yaw=env.ch.get_position(); h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            r["next_states"].append(np.asarray(h))
            if aft_batt<=0: break

        # GAE needs a bootstrap value for whatever comes after the last
        # recorded step. If the battery genuinely died (aft_batt<=0),
        # there truly is no future reward -- 0 is correct. If the loop
        # only ended because MAX_STEPS_PER_EPISODE was reached, the
        # environment did not actually terminate; bootstrapping with 0
        # would tell the value function "no more reward is possible
        # here" when that isn't true. Use the critic's own estimate of
        # the real next state in that case instead.
        if POLICY_TYPE == "transformer" and TRANSFORMER_VARIANT == "lyapunov":
            if aft_batt <= 0:
                bootstrap_value = 0.0
            else:
                with torch.no_grad():
                    (_, bootstrap_value_t, _, _, _, _) = model.distribution(seq_tensor(h, device))
                bootstrap_value = bootstrap_value_t.item()
            r["bootstrap_value"] = bootstrap_value

        rollouts.append(r); loss=""
        reward_history.append(total)

        start = time.perf_counter()
        # 3. Training Update Logic
        if POLICY_TYPE == "transformer":
            if ep % UPDATE_EVERY_EPISODES == 0:
                loss, diag_lyap, diag_barrier = update(model, opt, rollouts, device, ep)
                rollouts = []

                if diag_lyap is not None:
                    lyap_history.append(diag_lyap)
                    barrier_history.append(diag_barrier)

                    ready = (
                        ep >= 300
                        and len(lyap_history) == STABILITY_WINDOW
                        and len(reward_history) == REWARD_WINDOW
                    )
                    if ready:
                        avg_lyap = sum(lyap_history) / len(lyap_history)
                        avg_barrier = sum(barrier_history) / len(barrier_history)
                        avg_reward = sum(reward_history) / len(reward_history)
                        stable = (
                            avg_lyap <= LYAPUNOV_STABLE_THRESHOLD
                            and avg_barrier <= BARRIER_STABLE_THRESHOLD
                        )

                        if avg_reward > best_avg_reward:
                            best_avg_reward = avg_reward
                            checks_since_best = 0
                            torch.save(model.state_dict(), ckpt_dir / "best.pt")
                        else:
                            checks_since_best += 1

                        print(
                            f"[Convergence check] ep {ep} | "
                            f"avg_reward(last {REWARD_WINDOW}) {avg_reward:.2f} "
                            f"(best {best_avg_reward:.2f}) | "
                            f"avg_lyap(last {STABILITY_WINDOW}) {avg_lyap:.4f} | "
                            f"avg_barrier(last {STABILITY_WINDOW}) {avg_barrier:.4f} | "
                            f"stable={stable} | "
                            f"checks_since_best={checks_since_best}/{CONVERGENCE_PATIENCE}"
                        )

                        if stable and checks_since_best >= CONVERGENCE_PATIENCE:
                            print(
                                f"\n[Convergence] Lyapunov/barrier stable and reward "
                                f"plateaued at episode {ep} -- "
                                f"avg_reward={avg_reward:.2f}, avg_lyap={avg_lyap:.4f}, "
                                f"avg_barrier={avg_barrier:.4f}."
                            )
                            torch.save(model.state_dict(), ckpt_dir / "converged.pt")
                            converged = True
        else:
            # PSO Update: Evaluate current particle and update swarm
            model.evaluate_particle(model.current_particle, total)
            if ep % model.swarm_size == 0:
                model.update_swarm()
            loss = "N/A (PSO)"

        if ep % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), ckpt_dir / f"episode_{ep}.pt")

        elapsed = time.perf_counter() - start
        total_inference_time += elapsed
        total_inference_steps += 1

        steps_taken = len(r["rewards"]); log_status(ep, TOTAL_EPISODES, steps_taken, total, aft_batt, loss)
        epw.writerow(dict(episode=ep,steps=len(r["rewards"]),final_battery=aft_batt,total_reward=total,loss=loss)); epfile.flush()

        if converged and AUTO_STOP_ON_CONVERGENCE:
            print(f"[Convergence] Stopping training early at episode {ep}/{TOTAL_EPISODES}.")
            break

    # final deterministic evaluation, step-level telemetry CSV
    env.place_devices(); env.reset(); x,y,yaw=env.ch.get_position(); h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH)
    with open(OUT/"final_evaluation_steps.csv","w",newline="") as f:
        fields=["step","x_before","y_before","target_x","target_y","x_after","y_after","battery_before",
                "battery_after","battery_delta","reward","action_dx_norm","action_dy_norm"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()

        final_inference_time = 0.0
        final_inference_steps = 0
        for step in range(MAX_STEPS_PER_EPISODE):
            start = time.perf_counter()
            if POLICY_TYPE == "transformer":
                with torch.no_grad():
                    if TRANSFORMER_VARIANT == "lyapunov":
                        (a, raw_a, lp, v, lyapunov, barrier, latent, predicted_next) = model.fast_act(seq_tensor(h,device))
                    else:
                        a, lp, v = model.act(seq_tensor(h,device),True)
            else:
                # Set the current particle for the forward pass
                model.current_particle = (TOTAL_EPISODES - 1) % model.swarm_size
                with torch.no_grad():
                    a = model(seq_tensor(h,device))
                    lp, v = torch.tensor(0.0), torch.tensor(0.0)  # Dummy values

            elapsed = time.perf_counter() - start
            total_inference_time += elapsed
            total_inference_steps += 1
            final_inference_time += elapsed
            final_inference_steps += 1

            dx,dy=a[0].cpu().numpy()*MAX_MOVE_PER_STEP; tx=float(np.clip(x+dx,0,env.dim-1)); ty=float(np.clip(y+dy,0,env.dim-1))

            sol = solarposition.get_solarposition(env.times[min(step, len(env.times) - 1)],
                                                  env.lat_center + y * env.stp, env.long_center + x * env.stp)
            b = env.get_obfuscation(x, y, min(step, len(env.times) - 1), sol.azimuth.iloc[0],
                                         sol.apparent_zenith.iloc[0]).flatten()
            b_batt = env.ch.get_battery()

            tel,_=env.step_simulation(step,tx,ty)

            nx, ny, nyaw = env.ch.get_position()
            sol = solarposition.get_solarposition(env.times[min(step, len(env.times) - 1)],
                                                  env.lat_center + ny * env.stp, env.long_center + nx * env.stp)
            aft = env.get_obfuscation(nx, ny, min(step, len(env.times) - 1), sol.azimuth.iloc[0],
                                        sol.apparent_zenith.iloc[0]).flatten()
            aft_batt = env.ch.get_battery()

            rew=reward_fn(b, aft, tel, aft_batt-b_batt)

            w.writerow(dict(step=step,x_before=x,y_before=y,target_x=tx,target_y=ty,x_after=nx,y_after=ny,
                            battery_before=b_batt,battery_after=aft_batt,battery_delta=aft_batt-b_batt,reward=rew,
                            action_dx_norm=a[0,0].item(),action_dy_norm=a[0,1].item()))
            x,y,yaw=nx,ny,nyaw; h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            if aft_batt<=0: break
    epfile.close()
    # ADD THIS SECTION:
    print("\n" + "=" * 30)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 30)
    import pandas as pd
    df_eval = pd.read_csv(OUT / "final_evaluation_steps.csv")
    total_steps = len(df_eval)
    print("\n" + "=" * 30)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 30)
    print(f"Total Steps Performed: {total_steps}")
    print(f"Final Battery Level: {df_eval['battery_after'].iloc[-1]:.2f}%")
    avg_inference = total_inference_time / total_inference_steps
    avg_final_inference = final_inference_time / final_inference_steps
    print(f"Total inference calls : {total_inference_steps}")
    print(f"Total inference time  : {total_inference_time:.6f} s")
    print(f"Average inference     : {avg_inference * 1000:.3f} ms/step")
    print(f"Final inference calls : {final_inference_steps}")
    print(f"Final inference time  : {final_inference_time:.6f} s")
    print(f"Final Avg inference   : {avg_final_inference * 1000:.3f} ms/step")
if __name__=="__main__": run()