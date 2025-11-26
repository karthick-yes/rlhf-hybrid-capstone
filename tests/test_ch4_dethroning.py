"""
Checkpoint 4: Active Dethroning Sanity Check

Verifies that max-entropy acquisition selects pairs with P(Win) ≈ 0.5.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.acquisition.sampling import ActiveDethroning

def test_dethroning_sanity():
    print("\n" + "="*70)
    print("CHECKPOINT 4: ACTIVE DETHRONING SANITY CHECK")
    print("="*70)
    
    np.random.seed(42)
    acq = ActiveDethroning()
    
    # Defender
    def_mu = 10.0
    def_std = 2.0
    
    # Create 100 candidates with varying P(Win)
    n_candidates = 100
    target_p_wins = np.linspace(0.05, 0.95, n_candidates)
    
    # Reverse-engineer candidate stats to achieve target P(Win)
    # P(Win) = Φ(μ_Δ / σ_Δ) → μ_Δ = Φ^(-1)(P) * σ_Δ
    std_delta = 3.0
    mu_deltas = norm.ppf(target_p_wins) * std_delta
    
    cand_stds = np.full(n_candidates, 2.2)
    cand_mus = def_mu + mu_deltas
    cand_indices = list(range(n_candidates))
    
    # Compute actual P(Win) and scores
    actual_p_wins = []
    scores = []
    
    for i in range(n_candidates):
        p_win = acq.compute_win_probability(def_mu, def_std, cand_mus[i], cand_stds[i])
        score = acq.max_entropy_score(p_win)
        actual_p_wins.append(p_win)
        scores.append(score)
    
    actual_p_wins = np.array(actual_p_wins)
    scores = np.array(scores)
    
    # Select best
    best_idx, best_score, best_p_win, all_scores = acq.select_best_challenger(
        def_mu, def_std, cand_mus, cand_stds, cand_indices
    )
    
    print(f"\n📊 RESULTS:")
    print(f"   Best challenger ID:  {best_idx}")
    print(f"   P(Win):              {best_p_win:.4f} (target: 0.5000)")
    print(f"   Entropy score:       {best_score:.4f} (max: 1.0000)")
    print(f"   Distance from 0.5:   {abs(best_p_win - 0.5):.4f}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Entropy vs P(Win)
    ax1.plot(actual_p_wins, scores, linewidth=2.5, color='steelblue', label='Entropy Score')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Optimal (50-50)', alpha=0.7)
    ax1.scatter([best_p_win], [best_score], color='red', s=200, zorder=5, 
               marker='*', edgecolors='black', linewidths=2,
               label=f'Selected (P={best_p_win:.3f})')
    ax1.set_xlabel('P(Challenger Wins)', fontsize=13)
    ax1.set_ylabel('Entropy Score', fontsize=13)
    ax1.set_title('Max Entropy Acquisition Function', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Distribution of P(Win)
    ax2.hist(actual_p_wins, bins=20, alpha=0.6, edgecolor='black', color='skyblue')
    ax2.axvline(x=best_p_win, color='red', linestyle='--', linewidth=2.5, 
               label=f'Selected: {best_p_win:.3f}')
    ax2.axvline(x=0.5, color='green', linestyle=':', linewidth=2.5, label='Target: 0.500')
    ax2.set_xlabel('P(Challenger Wins)', fontsize=13)
    ax2.set_ylabel('Count', fontsize=13)
    ax2.set_title('Distribution of Win Probabilities', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, linestyle=':', axis='y')
    
    plt.tight_layout()
    plt.savefig('dethroning_sanity.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: dethroning_sanity.png")
    plt.close()
    
    # Pass/Fail
    print("\n" + "-"*70)
    if 0.4 <= best_p_win <= 0.6:
        print(" PASS: Dethroning selects near 50-50 pairs")
    else:
        print(f" FAIL: Selected P(Win)={best_p_win:.4f} not in [0.4, 0.6]")
        raise AssertionError(f"P(Win)={best_p_win:.4f} not in [0.4, 0.6]")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_dethroning_sanity()