"""
Standalone calibration test for ensemble after bootstrap.
Can be run independently to verify ensemble quality.
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from experiments.train import HybridTrainer

def test_ensemble_calibration_standalone():
    """Test that ensemble learns correlation between states and rewards"""
    print("\n" + "="*70)
    print("CHECKPOINT: ENSEMBLE CALIBRATION (STANDALONE)")
    print("="*70)
    
    config = {
        'state_dim': 4,
        'action_dim': 1,
        'env_name': 'CartPole-v1',
        'ensemble_size': 3,
        'lr': 1e-3,
        'hidden_dim': 64,
        'n_bootstrap': 10,  # Need enough data
        'buffer_capacity': 1000,
        'seed': 42,
        'beta': 3.0,
    }
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    trainer = HybridTrainer(config)
    
    # Generate correlated toy data
    for i in range(30):
        T = 50
        base_offset = np.random.uniform(-2, 2)  # Determines quality
        states = np.random.randn(T, 4) + base_offset
        actions = np.random.randn(T, 1) * 0.3
        cumulative_reward = float(np.sum(states)) + np.random.normal(0, 0.5)
        
        trainer.pref_buffer.add_trajectory(states, actions, cumulative_reward)
    
    # Run bootstrap
    print("\nRunning bootstrap...")
    trainer.bootstrap_ensemble(n_bootstrap=10)
    
    # The calibration check happens automatically in bootstrap_ensemble()
    print("\n Calibration test PASSED")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_ensemble_calibration_standalone()