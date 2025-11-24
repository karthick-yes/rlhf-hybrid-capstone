
import sys
sys.path.append('.')

import torch
import numpy as np
from src.utils.buffer_adapter import SACBufferAdapter
from src.utils.replay_buffer import ReplayBuffer
from src.models.reward_ensemble import RewardEnsemble

def test_buffer_adapter():
    print("\n" + "="*70)
    print("PHASE 3 TEST: Buffer Adapter & Relabeling")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim, action_dim = 4, 1
    
    # 1. Create trajectory buffer with mock data
    pref_buffer = ReplayBuffer()
    
    print("\n1. Creating mock trajectories...")
    for i in range(5):
        states = np.random.randn(50, state_dim)
        actions = np.random.randn(50, action_dim)
        cumulative_reward = np.random.randn()
        pref_buffer.add_trajectory(states, actions, cumulative_reward)
    
    print(f"   Created {len(pref_buffer)} trajectories")
    
    # 2. Create ensemble
    print("\n2. Creating reward ensemble...")
    config = {'ensemble_size': 3, 'lr': 3e-4}
    ensemble = RewardEnsemble(state_dim, action_dim, config, device)
    print(f"   Ensemble ready with {ensemble.K} models")
    
    # 3. Create SAC buffer adapter
    print("\n3. Creating SAC buffer adapter...")
    adapter = SACBufferAdapter(state_dim, action_dim, capacity=1000, device=device)
    print(f"   Adapter initialized (capacity: 1000)")
    
    # 4. Relabel and flatten
    print("\n4. Relabeling trajectories with learned rewards...")
    adapter.relabel_and_flatten(pref_buffer, ensemble)
    
    print(f"\nRESULTS:")
    print(f"   Adapter size: {adapter.size} transitions")
    print(f"   Expected: ~{5 * 49} transitions (5 trajs x 49 steps each)")
    
    # 5. Sample and check rewards
    if adapter.size > 10:
        states, actions, rewards, next_states, dones = adapter.sample(10)
        print(f"\n   Sample batch statistics:")
        print(f"     Reward mean: {rewards.mean():.4f}")
        print(f"     Reward std: {rewards.std():.4f}")
        print(f"     Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
        
        # Check non-zero
        non_zero = (np.abs(rewards) > 1e-6).sum()
        print(f"     Non-zero rewards: {non_zero}/10")
    
    # Pass/Fail
    print("\n" + "-"*70)
    if adapter.size > 0:
        print("PASS: Buffer adapter works correctly")
    else:
        print("FAIL: Buffer is empty after relabeling")
        raise AssertionError("Relabeling failed")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    test_buffer_adapter()