
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

# ============================================================
# PSO SUPPORT ADDED
# Set POLICY_TYPE = "transformer" or "pso"
# NOTE: This is a scaffold showing where to integrate PSO while
# preserving transformer logic. Replace model initialization and
# action selection as described.
POLICY_TYPE = "pso"
if POLICY_TYPE == "transformer":
    # Set TRANSFORMER_VARIANT = "normal" or "chaotic" or "lyapunov"
    TRANSFORMER_VARIANT = "chaotic"
    # Set TRANSFORMER_INIT = "normal" or "chaotic"
    if TRANSFORMER_VARIANT == "lyapunov":
        TRANSFORMER_INIT = "chaotic"
else:
    TRANSFORMER_VARIANT = "normal"
# ============================================================

TOTAL_EPISODES=1000; MAX_STEPS_PER_EPISODE=720; VIEW_DISTANCE=20
SEQUENCE_LENGTH=12; UPDATE_EVERY_EPISODES=5; GAMMA=.99; GAE_LAMBDA=.95
LR=3e-4; MAX_MOVE_PER_STEP=20.0; ENTROPY_COEF=.01; VALUE_COEF=.5

###############################################################
# Lyapunov Hyperparameters
###############################################################
LYAPUNOV_COEF = 0.25; DYNAMICS_COEF = 0.10; LATENT_COEF = 0.05
BARRIER_COEF = 0.20; ACTION_SMOOTHNESS = 0.01; LYAPUNOV_MARGIN = 0.01
BATTERY_MARGIN = 0.10; BOUNDARY_MARGIN = 0.10; VEGETATION_MARGIN = 0.05
VELOCITY_MARGIN = 0.05; COMM_MARGIN = 0.10; CLIP_EPS =0.20; LYAPUNOV_ALPHA = 0.02

# Format as YYYY-MM-DD_HH-MM-SS
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT=Path("rl_csv_"+timestamp); OUT.mkdir(exist_ok=True)

def log_status(ep, total_episodes, steps, avg_reward, final_batt, loss, is_eval=False):
    prefix = "[EVALUATION]" if is_eval else f"[Episode {ep}/{total_episodes}]"
    loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else loss
    print(f"{prefix} Steps: {steps:3} | Reward: {avg_reward:7.2f} | Battery: {final_batt:6.2f}% | Loss: {loss_str}")

def obs(env, x, y, yaw, step):
    sol=solarposition.get_solarposition(env.times[min(step,len(env.times)-1)], env.lat_center+x*env.stp, env.long_center+y*env.stp)
    patch=env.get_obfuscation(x,y,min(step,len(env.times)-1),sol.azimuth.iloc[0],sol.apparent_zenith.iloc[0]).flatten()
    scalars=np.array([x/(env.dim-1),y/(env.dim-1),math.sin(yaw),math.cos(yaw),
                      env.ch.get_battery()/100, sol.azimuth.iloc[0]/360, sol.apparent_zenith.iloc[0]/90],np.float32)
    return np.concatenate([patch.astype(np.float32),scalars])

def seq_tensor(history, device):
    return torch.tensor(np.asarray(history),dtype=torch.float32,device=device).unsqueeze(0)

def reward_fn(before, after, telemetry, env):
    # Dense, scaled reward: energy gain, movement cost, survival, and boundary discouragement.
    delta=after-before
    px,py,_=telemetry["previous_position"]; nx,ny,_=telemetry["new_position"]
    distance=math.hypot(nx-px,ny-py)
    boundary=min(nx,ny,env.dim-nx,env.dim-ny)/env.dim
    return 2.0*delta - .015*distance + .20*(after/100) + .10*boundary

def update(model,opt,rollouts,device, ep):
    # GAE advantages are normalized once across the complete rollout batch, not per episode.
    states=[]; next_states=[]; actions=[]; oldlp=[]; adv=[]; returns=[]
    for r in rollouts:
        gae=0.0; ret=0.0
        for i in reversed(range(len(r["rewards"]))):
            ret=r["rewards"][i]+GAMMA*ret
            nxt=0 if i==len(r["rewards"])-1 else r["values"][i+1]
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

        (lp, entropy, values, lyapunov, barrier, latent, predicted_next) = model.evaluate_actions(states,actions)
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

        loss = (policy_loss
                + VALUE_COEF * value_loss
                - ENTROPY_COEF * entropy.mean()
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
            f"CF {clip_fraction:.4f}"
        )
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
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return float(loss.item())

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
        elif TRANSFORMER_VARIANT == "chaotic":
            model = ChebyshevTransformer(VIEW_DISTANCE).to(device)
        else:
            model = TransformerActorCritic(VIEW_DISTANCE).to(device)
        opt = optim.Adam(model.parameters(), lr=LR)
    else:
        # Assuming observation dimension is flattened patch size + scalars
        # Calculate based on your obs function (e.g., 20x20 patch + 7 scalars)
        input_dim = (VIEW_DISTANCE * 2 + 1) ** 2 + 7
        model = PSOPolicy(input_dim=input_dim).to(device)
        opt = None  # PSO does not use torch.optim

    epfile=open(OUT/"episode_metrics.csv","w",newline=""); epw=csv.DictWriter(epfile,fieldnames=["episode","steps","final_battery","total_reward","loss"]); epw.writeheader()
    rollouts=[]
    for ep in range(1,TOTAL_EPISODES+1):
        env.place_devices(); env.reset(); x,y,yaw=env.ch.get_position()
        h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH);total=0;
        r={"states":[],"next_states":[],"actions":[],"logps":[],"values":[],"rewards":[],"lyapunov":[], "barrier":[]};
        previous_action = None; smoothness = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            s=seq_tensor(h,device)

            # 2. Action Selection Logic
            if POLICY_TYPE == "transformer":
                with torch.no_grad():
                    if TRANSFORMER_VARIANT == "lyapunov":
                        (a, lp, v, lyapunov, barrier, latent, predicted_next) = model.act(s)
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

            # normalized action -> local, physically scaled target; no global-coordinate clipping mismatch
            dx,dy=a[0].cpu().numpy()*MAX_MOVE_PER_STEP
            tx=float(np.clip(x+dx,0,env.dim-1)); ty=float(np.clip(y+dy,0,env.dim-1))
            before=env.ch.get_battery(); tel,nxt=env.step_simulation(step,tx,ty); after=env.ch.get_battery()
            rew=reward_fn(before,after,tel,env)-ACTION_SMOOTHNESS*smoothness;
            total+=rew; previous_action=current_action
            r["states"].append(np.asarray(h)); r["actions"].append(a[0].cpu().numpy())
            r["logps"].append(lp.item()); r["values"].append(v.item()); r["rewards"].append(rew)
            x,y,yaw=env.ch.get_position(); h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            r["next_states"].append(np.asarray(h))
            if after<=0: break
        rollouts.append(r); loss=""

        # 3. Training Update Logic
        if POLICY_TYPE == "transformer":
            if ep % UPDATE_EVERY_EPISODES == 0:
                loss = update(model, opt, rollouts, device, ep)
                rollouts = []
        else:
            # PSO Update: Evaluate current particle and update swarm
            model.evaluate_particle(model.current_particle, total)
            if ep % model.swarm_size == 0:
                model.update_swarm()
            loss = "N/A (PSO)"

        steps_taken = len(r["rewards"]); log_status(ep, TOTAL_EPISODES, steps_taken, total, after, loss)
        epw.writerow(dict(episode=ep,steps=len(r["rewards"]),final_battery=after,total_reward=total,loss=loss)); epfile.flush()
    # final deterministic evaluation, step-level telemetry CSV
    env.place_devices(); env.reset(); x,y,yaw=env.ch.get_position(); h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH)
    with open(OUT/"final_evaluation_steps.csv","w",newline="") as f:
        fields=["step","x_before","y_before","target_x","target_y","x_after","y_after","battery_before","battery_after","battery_delta","reward","action_dx_norm","action_dy_norm"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for step in range(MAX_STEPS_PER_EPISODE):
            if POLICY_TYPE == "transformer":
                with torch.no_grad():
                    if TRANSFORMER_VARIANT == "lyapunov":
                        (a, lp, v, lyapunov, barrier, latent, predicted_next) = model.act(seq_tensor(h,device), True)
                    else:
                        a, lp, v = model.act(seq_tensor(h,device),True)
            else:
                # Set the current particle for the forward pass
                model.current_particle = (TOTAL_EPISODES - 1) % model.swarm_size
                with torch.no_grad():
                    a = model(seq_tensor(h,device))
                    lp, v = torch.tensor(0.0), torch.tensor(0.0)  # Dummy values
            dx,dy=a[0].cpu().numpy()*MAX_MOVE_PER_STEP; tx=float(np.clip(x+dx,0,env.dim-1)); ty=float(np.clip(y+dy,0,env.dim-1))
            b=env.ch.get_battery(); tel,_=env.step_simulation(step,tx,ty); aft=env.ch.get_battery(); nx,ny,nyaw=env.ch.get_position(); rew=reward_fn(b,aft,tel,env)
            w.writerow(dict(step=step,x_before=x,y_before=y,target_x=tx,target_y=ty,x_after=nx,y_after=ny,battery_before=b,battery_after=aft,battery_delta=aft-b,reward=rew,action_dx_norm=a[0,0].item(),action_dy_norm=a[0,1].item()))
            x,y,yaw=nx,ny,nyaw; h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            if aft<=0: break
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
if __name__=="__main__": run()
