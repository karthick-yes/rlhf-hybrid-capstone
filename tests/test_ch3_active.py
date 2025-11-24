"""
Checkpoint 3: UCB/LCB Correctness Test

Verifies that UCB/LCB pseudo-labeling achieves >95% accuracy.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from src.acquisition.ucb_lcb import UCBLCBFilter

def test_ucb_lcb_correctness():
    print("\n" + "="*70)
    print("CHECKPOINT 3: UCB/LCB PSEUDO-LABELING CORRECTNESS")
    print("="*70)
    
    np.random.seed(42)
    filter = UCBLCBFilter(beta=3.0)
    
    # Ground truth
    true_def_reward = 10.0
    
    # Generate 500 candidates
    n_candidates = 500
    true_cand_rewards = np.random.randn(n_candidates) * 5 + 8  # Mean=8, Std=5
    
    # Simulate ensemble predictions (add noise)
    prediction_noise = 2.0
    cand_mus = true_cand_rewards + np.random.randn(n_candidates) * prediction_noise
    cand_stds = np.random.rand(n_candidates) * 3 + 0.5  # [0.5, 3.5]
    
    def_mu = true_def_reward + np.random.randn() * prediction_noise
    def_std = 1.5
    
    # Apply filter
    cand_ids = list(range(n_candidates))
    auto_labels, uncertain_idx, dethroned, new_def = filter.filter_candidates(
        def_mu, def_std, cand_mus, cand_stds, cand_ids
    )
    
    # Check accuracy
    correct = 0
    total = len(auto_labels)
    
    for winner, loser in auto_labels:
        if winner == 'defender':
            if true_def_reward > true_cand_rewards[loser]:
                correct += 1
        else:
            if true_cand_rewards[winner] > true_def_reward:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\n📊 RESULTS:")
    print(f"   Total candidates:        {n_candidates}")
    print(f"   Auto-labeled:            {total} ({100*total/n_candidates:.1f}%)")
    print(f"   Uncertain (need human):  {len(uncertain_idx)} ({100*len(uncertain_idx)/n_candidates:.1f}%)")
    print(f"   Accuracy of auto-labels: {accuracy:.4f} ({correct}/{total})")
    
    # Visualize subset (first 30 candidates)
    subset_size = 30
    subset_mus = cand_mus[:subset_size]
    subset_stds = cand_stds[:subset_size]
    
    labels = []
    for i in range(subset_size):
        cand_ucb, cand_lcb = filter.compute_bounds(subset_mus[i], subset_stds[i])
        def_ucb, def_lcb = filter.compute_bounds(def_mu, def_std)
        
        if def_lcb > cand_ucb:
            labels.append('auto_def')
        elif cand_lcb > def_ucb:
            labels.append('auto_chal')
        else:
            labels.append('uncertain')
    
    filter.visualize_bounds(def_mu, def_std, subset_mus, subset_stds, labels)
    
    # Pass/Fail
    print("\n" + "-"*70)
    if accuracy > 0.95:
        print("✅ PASS: UCB/LCB generates correct pseudo-labels (accuracy > 95%)")
    else:
        print(f"❌ FAIL: Accuracy {accuracy:.4f} < 0.95")
        raise AssertionError(f"Accuracy {accuracy:.4f} < 0.95")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_ucb_lcb_correctness()