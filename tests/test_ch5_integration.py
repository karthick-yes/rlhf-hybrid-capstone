"""
Checkpoint 5: Full Pipeline Integration Test

Runs the complete hybrid system on STRUCTURED toy data and verifies:
1. Graph grows correctly
2. Augmentation ratio > 2.0 (UCB/LCB + transitivity)
3. No crashes
4. Human queries are made (not stuck in toxic loop)

- Uses 5 queries (not 20) for speed
- Generates structured trajectories with clear reward hierarchy
- Runs in ~30 seconds
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from experiments.train import HybridTrainer

def generate_structured_toy_trajectories(pref_buffer, n_trajectories=30):
    """
    Generate toy trajectories where reward CORRELATES with state content.
    This makes the problem learnable and validates theoretical guarantees.
    Hierarchy: 5 "good" trajectories, 15 "medium", 10 "bad"
    """
    np.random.seed(42)
    
    # Trajectory quality tiers
    tiers = {
        'good': {'count': 5, 'reward_center': 10.0, 'reward_std': 1.0},
        'medium': {'count': 15, 'reward_center': 5.0, 'reward_std': 2.0},
        'bad': {'count': 10, 'reward_center': 0.0, 'reward_std': 1.5},
    }
    
    traj_id = 0
    for tier, config in tiers.items():
        for _ in range(config['count']):
            # Generate trajectory data - NOW CORRELATED WITH REWARD
            T = 50
            # Create correlation: Higher cumulative state sum → Higher reward
            base_offset = np.random.normal(config['reward_center'] * 0.5, 0.5)
            states = np.random.randn(T, 4) + base_offset  # States encode quality
            actions = np.random.randn(T, 1) * 0.3
            
            # Reward is a FUNCTION of states (learnable), not independent noise
            cumulative_reward = float(np.sum(states)) + np.random.normal(0, 0.5)
            cumulative_reward = np.clip(cumulative_reward, -10, 15)
            
            pref_buffer.add_trajectory(states, actions, cumulative_reward)
            traj_id += 1
    
    print(f"   Generated {traj_id} structured trajectories:")
    print(f"     Good (reward ~10): {tiers['good']['count']}")
    print(f"     Medium (reward ~5): {tiers['medium']['count']}")
    print(f"     Bad (reward ~0): {tiers['bad']['count']}")

def test_full_integration():
    print("\n" + "="*70)
    print("CHECKPOINT 5: FULL PIPELINE INTEGRATION (FAST MODE)")
    print("="*70)
    
    # MINIMAL config for SPEED
    config = {
        'state_dim': 4,
        'action_dim': 1,
        'env_name': 'CartPole-v1',  # For compatibility, but mocked
        'ensemble_size': 3,
        'lr': 1e-3,  # Faster convergence
        'hidden_dim': 64,  # Smaller network
        'beta': 2.0,  # More aggressive UCB/LCB for toy data
        'pool_size': 15,  # Smaller pool
        'update_freq': 5,  # Update every 5 queries
        'n_trajectories': 30,  # Mocked trajectories
        'n_queries': 5,  # ONLY 5 QUERIES FOR SPEED
        'n_bootstrap': 10,  # INCREASED for stable ensemble
        'buffer_capacity': 1000,
        'min_warmup_reward': 8.0,  # Lower threshold for toy data
        'max_defender_uncertainty': 2.0,  # More lenient
        'exploration_epsilon': 0.2,  # More exploration
        'label_smoothing': 0.05,
        'seed': 42
    }
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    trainer = HybridTrainer(config)
    
    # Generate STRUCTURED toy data (now learnable)
    print("\n🎲 Generating structured toy trajectories...")
    generate_structured_toy_trajectories(trainer.pref_buffer, n_trajectories=30)
    
    print("\n🔄 Running 5-query test (should take ~30 seconds)...")
    
    # Run SHORT active loop
    trainer.phase2_active_learning(n_queries=5, pool_size=15)
    
    # NEW: Validate ensemble calibration before final evaluation
    print("\n🔍 Validating ensemble calibration...")
    try:
        trainer.validate_ensemble_calibration()
        calibration_pass = True
    except AssertionError as e:
        print(f"❌ Calibration failed: {e}")
        calibration_pass = False
    
    # Evaluate results
    stats = trainer.graph.get_stats()
    direct = stats['direct']
    transitive = stats['transitive']
    total_edges = stats['total']
    
    # Calculate augmentation ratio
    augmentation_ratio = total_edges / direct if direct > 0 else 0
    
    print(f"\n📊 FINAL METRICS:")
    print(f"   Human queries:    {trainer.total_human_queries}")
    print(f"   Direct edges:     {direct}")
    print(f"   Transitive edges: {transitive}")
    print(f"   Total edges:      {total_edges}")
    print(f"   UCB/LCB auto:     {stats['auto']}")
    print(f"   Augmentation:     {augmentation_ratio:.2f}x")
    
    # --- PASS/FAIL CRITERIA ---
    print("\n" + "-"*70)
    
    success = True
    
    # 1. System must make human queries (not stuck)
    if trainer.total_human_queries >= 2:
        print(f"✅ PASS: Made {trainer.total_human_queries} human queries (not stuck)")
    else:
        print(f"❌ FAIL: Only {trainer.total_human_queries} human queries (toxic loop?)")
        success = False
    
    # 2. Augmentation ratio must be > 2.0 (NOW HARD FAILURE)
    if augmentation_ratio >= 2.0:
        print(f"✅ PASS: Augmentation ratio {augmentation_ratio:.2f}x >= 2.0x")
    else:
        print(f"❌ FAIL: Augmentation {augmentation_ratio:.2f}x < 2.0x")
        success = False  # Made this a hard failure
    
    # 3. Graph must grow
    if total_edges > direct:
        print(f"✅ PASS: Transitive closure working ({transitive} edges inferred)")
    else:
        print(f"❌ FAIL: No transitive augmentation")
        success = False
    
    # 4. Ensemble calibration check
    if calibration_pass:
        print(f"✅ PASS: Ensemble is well-calibrated")
    else:
        print(f"❌ FAIL: Ensemble calibration failed")
        success = False
    
    # 5. No crashes
    print(f"✅ PASS: System executed without crashes")
    
    print("="*70 + "\n")
    
    if not success:
        raise AssertionError("Integration test failed")

if __name__ == "__main__":
    test_full_integration()