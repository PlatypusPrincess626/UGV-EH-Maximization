
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
from chen_transformer import ChenTransformer
from pso_policy import PSOPolicy
import datetime

# ============================================================
# PSO SUPPORT ADDED
# Set POLICY_TYPE = "transformer" or "pso"
# NOTE: This is a scaffold showing where to integrate PSO while
# preserving transformer logic. Replace model initialization and
# action selection as described.
POLICY_TYPE = "transformer"
if POLICY_TYPE == "transformer":
    # Set TRANSFORMER_VARIANT = "normal" or "chen"
    TRANSFORMER_VARIANT = "chen"
# ============================================================

TOTAL_EPISODES=1000; MAX_STEPS_PER_EPISODE=720; VIEW_DISTANCE=20
SEQUENCE_LENGTH=12; UPDATE_EVERY_EPISODES=5; GAMMA=.99; GAE_LAMBDA=.95
LR=3e-4; MAX_MOVE_PER_STEP=20.0; ENTROPY_COEF=.01; VALUE_COEF=.5
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

def reward_fn(before, after, telemetry, action):
    # Dense, scaled reward: energy gain, movement cost, survival, and boundary discouragement.
    delta=after-before
    px,py,_=telemetry["previous_position"]; nx,ny,_=telemetry["new_position"]
    distance=math.hypot(nx-px,ny-py)
    boundary=min(nx,ny,800-nx,800-ny)/800
    return 25.0*delta - .015*distance + .20*(after/100) + .10*boundary

def update(model,opt,rollouts,device):
    # GAE advantages are normalized once across the complete rollout batch, not per episode.
    states=[]; actions=[]; oldlp=[]; adv=[]; returns=[]
    for r in rollouts:
        gae=0.0; ret=0.0
        for i in reversed(range(len(r["rewards"]))):
            ret=r["rewards"][i]+GAMMA*ret
            nxt=0 if i==len(r["rewards"])-1 else r["values"][i+1]
            gae=r["rewards"][i]+GAMMA*nxt-r["values"][i]+GAMMA*GAE_LAMBDA*gae
            adv.insert(0,gae); returns.insert(0,ret)
        states+=r["states"]; actions+=r["actions"]; oldlp+=r["logps"]
    adv = torch.tensor(adv,device=device,dtype=torch.float32); adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    returns=torch.tensor(returns,device=device,dtype=torch.float32); actions=torch.tensor(np.asarray(actions),dtype=torch.float32,device=device)
    oldlp=torch.tensor(oldlp,device=device); states=torch.tensor(np.asarray(states),dtype=torch.float32,device=device)
    lp,entropy,values=model.evaluate_actions(states,actions)
    # Single efficient batched actor-critic update; no duplicate forward pass per step.
    loss=-(lp*adv.detach()).mean()+VALUE_COEF*F.mse_loss(values,returns)+-ENTROPY_COEF*entropy.mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return float(loss.item())

def run():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=sim_env("test",20,MAX_STEPS_PER_EPISODE); env.set_view_dist(VIEW_DISTANCE)

    # 1. Initialization Logic
    if POLICY_TYPE == "transformer":
        if TRANSFORMER_VARIANT == "normal":
            model = TransformerActorCritic(VIEW_DISTANCE).to(device)
            opt = optim.Adam(model.parameters(), lr=LR)
        else:
            model = ChenTransformer(VIEW_DISTANCE).to(device)
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
        env.place_devices(); env.ch.reset(); x,y,yaw=env.ch.get_position()
        h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH); r={"states":[],"actions":[],"logps":[],"values":[],"rewards":[]}; total=0
        for step in range(MAX_STEPS_PER_EPISODE):
            s=seq_tensor(h,device)

            # 2. Action Selection Logic
            if POLICY_TYPE == "transformer":
                with torch.no_grad():
                    a, lp, v = model.act(s)
            else:
                # Set the current particle for the forward pass
                model.current_particle = (ep - 1) % model.swarm_size
                with torch.no_grad():
                    a = model(s)
                    lp, v = torch.tensor(0.0), torch.tensor(0.0)  # Dummy values

            # normalized action -> local, physically scaled target; no global-coordinate clipping mismatch
            dx,dy=a[0].cpu().numpy()*MAX_MOVE_PER_STEP
            tx=float(np.clip(x+dx,0,env.dim-1)); ty=float(np.clip(y+dy,0,env.dim-1))
            before=env.ch.get_battery(); tel,nxt=env.step_simulation(step,tx,ty); after=env.ch.get_battery()
            rew=reward_fn(before,after,tel,a[0]); total+=rew
            r["states"].append(np.asarray(h)); r["actions"].append(a[0].cpu().numpy()); r["logps"].append(lp.item()); r["values"].append(v.item()); r["rewards"].append(rew)
            x,y,yaw=env.ch.get_position(); h.append(obs(env,x,y,yaw,min(step+1,MAX_STEPS_PER_EPISODE-1)))
            if after<=0: break
        rollouts.append(r); loss=""

        # 3. Training Update Logic
        if POLICY_TYPE == "transformer":
            if ep % UPDATE_EVERY_EPISODES == 0:
                loss = update(model, opt, rollouts, device)
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
    env.place_devices(); env.ch.reset(); x,y,yaw=env.ch.get_position(); h=deque([obs(env,x,y,yaw,0)]*SEQUENCE_LENGTH,maxlen=SEQUENCE_LENGTH)
    with open(OUT/"final_evaluation_steps.csv","w",newline="") as f:
        fields=["step","x_before","y_before","target_x","target_y","x_after","y_after","battery_before","battery_after","battery_delta","reward","action_dx_norm","action_dy_norm"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for step in range(MAX_STEPS_PER_EPISODE):
            if POLICY_TYPE == "transformer":
                with torch.no_grad():
                    a, lp, v = model.act(seq_tensor(h,device),True)
            else:
                # Set the current particle for the forward pass
                model.current_particle = (TOTAL_EPISODES - 1) % model.swarm_size
                with torch.no_grad():
                    a = model(seq_tensor(h,device))
                    lp, v = torch.tensor(0.0), torch.tensor(0.0)  # Dummy values
            dx,dy=a[0].cpu().numpy()*MAX_MOVE_PER_STEP; tx=float(np.clip(x+dx,0,env.dim-1)); ty=float(np.clip(y+dy,0,env.dim-1))
            b=env.ch.get_battery(); tel,_=env.step_simulation(step,tx,ty); aft=env.ch.get_battery(); nx,ny,nyaw=env.ch.get_position(); rew=reward_fn(b,aft,tel,a[0])
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
