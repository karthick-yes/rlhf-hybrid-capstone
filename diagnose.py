import torch
import numpy as np
import gymnasium as gym
from src.models.reward_ensemble import RewardEnsemble
from src.agents.sac import SAC
from src.envs.wrappers import make_env
import matplotlib.pyplot as plt
import os

def diagnose(checkpoint_path):
    print(f"\n🔍 DIAGNOSING: {checkpoint_path}")
    
    # 1. Load Checkpoint
    try:
        ckpt = torch.load(checkpoint_path)
        config = ckpt['config']
        print("   ✓ Checkpoint loaded")
    except FileNotFoundError:
        print(f"   ❌ File not found: {checkpoint_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_name = config.get('env_name', 'CartPole-v1')
    env = make_env(env_name)
    
    # 2. Reconstruct Reward Model
    print("\n📊 REWARD MODEL DIAGNOSIS")
    ensemble = RewardEnsemble(env.state_dim, env.agent_action_dim, config, device)
    # Load weights into the first model of the ensemble (representative)
    ensemble.models[0].load_state_dict(ckpt['ensemble_state'][0])
    
    # Test: Balanced vs Crashing
    # State: [CartPos, CartVel, PoleAngle, PoleVel]
    balanced_state = torch.FloatTensor([0, 0, 0, 0]).to(device).unsqueeze(0) # Perfect balance
    crashed_state  = torch.FloatTensor([0, 0, 0.2, 0.5]).to(device).unsqueeze(0) # Tilted/Falling
    action = torch.FloatTensor([[0]]).to(device) # Action irrelevant for state-value intuition check
    
    with torch.no_grad():
        r_bal, _ = ensemble.predict_step(balanced_state, action)
        r_crash, _ = ensemble.predict_step(crashed_state, action)
    
    print(f"   Predicted Reward (Balanced): {r_bal.item():.4f}")
    print(f"   Predicted Reward (Crashing): {r_crash.item():.4f}")
    
    if r_bal > r_crash:
        print("   ✓ Reward Model correctly prefers Balancing.")
    else:
        print("   ❌ Reward Model prefers CRASHING! (Misaligned)")

    # 3. Reconstruct SAC Agent
    print("\n🤖 SAC AGENT DIAGNOSIS")
    sac = SAC(env.state_dim, env.agent_action_dim, is_discrete=env.is_discrete, device=device)
    sac.actor.load_state_dict(ckpt['sac_actor'])
    sac.critic.load_state_dict(ckpt['sac_critic'])
    
    # Test Policy Output
    with torch.no_grad():
        # Check action probabilities for Balanced State
        if env.is_discrete:
            logits = sac.actor(balanced_state)
            probs = torch.softmax(logits, dim=1)
            print(f"   Policy at Balanced: {probs.cpu().numpy()} (Should be ~50/50 or confident counter-balance)")
            
            # Check Q-Values
            q1, q2 = sac.critic(balanced_state)
            print(f"   Q-Values at Balanced: {q1.cpu().numpy()}")
        else:
            mean, _ = sac.actor(balanced_state)
            print(f"   Policy Mean at Balanced: {mean.item():.4f}")

    # 4. Run a Test Episode
    print("\n🎮 LIVE TEST EPISODE")
    s, _ = env.reset()
    total_r = 0
    steps = 0
    done = False
    
    print("   Step | Action | Pole Angle | Predicted Reward")
    while not done and steps < 20:
        a = sac.select_action(s, deterministic=True)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(device)
        
        # Handle discrete/continuous for reward pred
        if env.is_discrete:
            a_t = torch.FloatTensor([[a]]).to(device)
        else:
            a_t = torch.FloatTensor(a).unsqueeze(0).to(device)
            
        r_pred, _ = ensemble.predict_step(s_t, a_t)
        
        print(f"   {steps:4d} | {a:6.2f} | {s[2]:10.4f} | {r_pred.item():.4f}")
        
        s, r, term, trunc, _ = env.step(a)
        total_r += r
        steps += 1
        done = term or trunc
        
    print(f"   Final Length: {steps}")
    if steps < 15:
        print("   ❌ Agent dies immediately.")
    else:
        print("   ✓ Agent survives.")

if __name__ == "__main__":
    # Check the checkpoints folder for the latest file
    import glob
    files = glob.glob("checkpoints/checkpoint_round*_sac_done.pt")
    if files:
        latest_file = max(files, key=os.path.getctime)
        diagnose(latest_file)
    else:
        print("No checkpoint found!")