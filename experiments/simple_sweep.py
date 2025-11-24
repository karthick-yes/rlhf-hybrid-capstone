"""
Hyperparameter Sweep for CartPole
Systematically tests different configurations to find optimal settings
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from experiments.train import HybridTrainer

# Base configuration
BASE_CONFIG = {
    'env_name': 'CartPole-v1',
    'state_dim': 4,
    'action_dim': 1,
    'segment_length': 50,
    'n_trajectories': 20,
    'n_bootstrap': 5,
    'n_queries': 20,
    'pool_size': 20,
    'update_freq': 10,
    'ensemble_size': 3,
    'hidden_dim': 256,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'sac_steps': 2000,
    'buffer_capacity': 10000,
    'seed': 42
}

def run_single_config(config, seed=42):
    """
    Run one configuration and return metrics.
    
    Returns:
        dict with:
        - final_augmentation: Final augmentation ratio
        - avg_auto_rate: Average auto-label rate
        - total_human_queries: Human queries consumed
        - convergence_iteration: When augmentation >3.0 (or -1)
    """
    config['seed'] = seed
    trainer = HybridTrainer(config)
    
    # Run phases
    trainer.phase1_warmup(n_episodes=config['n_trajectories'])
    trainer.phase2_active_learning(
        n_queries=config['n_queries'],
        pool_size=config['pool_size']
    )
    
    # Extract metrics
    stats = trainer.graph.get_stats()
    
    # Compute averages from log
    if len(trainer.query_log) > 0:
        avg_auto_rate = np.mean([log['auto_rate'] for log in trainer.query_log])
        
        # Find convergence point
        convergence = -1
        for log in trainer.query_log:
            if log['augmentation'] >= 3.0:
                convergence = log['iteration']
                break
    else:
        avg_auto_rate = 0.0
        convergence = -1
    
    return {
        'final_augmentation': stats['ratio'],
        'avg_auto_rate': avg_auto_rate,
        'total_human_queries': trainer.total_human_queries,
        'convergence_iteration': convergence,
        'final_edges': stats['total']
    }


def sweep_beta(beta_values=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], n_seeds=3):
    """
    Sweep over beta (UCB/LCB confidence width).
    
    Question: What beta balances precision (avoid wrong auto-labels) 
              and efficiency (maximize auto-labeling)?
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER SWEEP: BETA (UCB/LCB CONFIDENCE WIDTH)")
    print("="*70)
    
    results = []
    
    for beta in beta_values:
        print(f"\nTesting beta={beta}...")
        
        config = BASE_CONFIG.copy()
        config['beta'] = beta
        
        # Run multiple seeds
        metrics_list = []
        for seed in range(n_seeds):
            try:
                metrics = run_single_config(config, seed=seed + 42)
                metrics_list.append(metrics)
                print(f"  Seed {seed}: Aug={metrics['final_augmentation']:.2f}x, "
                      f"Auto={metrics['avg_auto_rate']:.1f}%, "
                      f"Human={metrics['total_human_queries']}")
            except Exception as e:
                print(f"  Seed {seed} FAILED: {e}")
        
        # Aggregate across seeds
        if metrics_list:
            results.append({
                'beta': beta,
                'augmentation_mean': np.mean([m['final_augmentation'] for m in metrics_list]),
                'augmentation_std': np.std([m['final_augmentation'] for m in metrics_list]),
                'auto_rate_mean': np.mean([m['avg_auto_rate'] for m in metrics_list]),
                'auto_rate_std': np.std([m['avg_auto_rate'] for m in metrics_list]),
                'human_queries_mean': np.mean([m['total_human_queries'] for m in metrics_list]),
                'convergence_mean': np.mean([m['convergence_iteration'] for m in metrics_list if m['convergence_iteration'] > 0])
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Augmentation ratio
    ax1.errorbar(df['beta'], df['augmentation_mean'], 
                yerr=df['augmentation_std'], marker='o', linewidth=2, capsize=5)
    ax1.axhline(y=3.0, color='red', linestyle='--', label='Target: 3.0x')
    ax1.set_xlabel('Beta (Confidence Width)')
    ax1.set_ylabel('Final Augmentation Ratio')
    ax1.set_title('Augmentation vs Beta')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Auto-label rate
    ax2.errorbar(df['beta'], df['auto_rate_mean'], 
                yerr=df['auto_rate_std'], marker='s', linewidth=2, capsize=5, color='green')
    ax2.set_xlabel('Beta (Confidence Width)')
    ax2.set_ylabel('Average Auto-label Rate (%)')
    ax2.set_title('Auto-labeling Efficiency vs Beta')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Human queries consumed
    ax3.bar(df['beta'], df['human_queries_mean'], alpha=0.7, color='orange')
    ax3.set_xlabel('Beta (Confidence Width)')
    ax3.set_ylabel('Human Queries Used')
    ax3.set_title('Query Efficiency vs Beta')
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Convergence iteration
    converged = df[df['convergence_mean'] > 0]
    if not converged.empty:
        ax4.plot(converged['beta'], converged['convergence_mean'], 
                marker='^', linewidth=2, markersize=10, color='purple')
        ax4.set_xlabel('Beta (Confidence Width)')
        ax4.set_ylabel('Convergence Iteration (Aug >3.0x)')
        ax4.set_title('Convergence Speed vs Beta')
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('beta_sweep_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved: beta_sweep_results.png")
    
    # Save CSV
    df.to_csv('beta_sweep_results.csv', index=False)
    print(f"Data saved: beta_sweep_results.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    best_aug = df.loc[df['augmentation_mean'].idxmax()]
    best_auto = df.loc[df['auto_rate_mean'].idxmax()]
    best_queries = df.loc[df['human_queries_mean'].idxmin()]
    
    print(f"Best Augmentation: beta={best_aug['beta']} ({best_aug['augmentation_mean']:.2f}x)")
    print(f"Best Auto-rate:    beta={best_auto['beta']} ({best_auto['auto_rate_mean']:.1f}%)")
    print(f"Fewest Queries:    beta={best_queries['beta']} ({best_queries['human_queries_mean']:.0f} queries)")
    
    # Recommend based on trade-offs
    if best_aug['beta'] == best_auto['beta']:
        print(f"\nRECOMMENDATION: beta={best_aug['beta']} (optimal for both metrics)")
    else:
        print(f"\nRECOMMENDATION: beta={best_aug['beta']} for augmentation, "
              f"beta={best_auto['beta']} for auto-labeling")
    
    return df


def sweep_ensemble_size(sizes=[1, 3, 5, 7], n_seeds=3):
    """
    Sweep over ensemble size.
    
    Question: How many models needed for good uncertainty estimates?
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER SWEEP: ENSEMBLE SIZE")
    print("="*70)
    
    results = []
    
    for K in sizes:
        print(f"\nTesting K={K} models...")
        
        config = BASE_CONFIG.copy()
        config['ensemble_size'] = K
        
        metrics_list = []
        for seed in range(n_seeds):
            try:
                metrics = run_single_config(config, seed=seed + 42)
                metrics_list.append(metrics)
                print(f"  Seed {seed}: Aug={metrics['final_augmentation']:.2f}x, "
                      f"Auto={metrics['avg_auto_rate']:.1f}%")
            except Exception as e:
                print(f"  Seed {seed} FAILED: {e}")
        
        if metrics_list:
            results.append({
                'ensemble_size': K,
                'augmentation_mean': np.mean([m['final_augmentation'] for m in metrics_list]),
                'auto_rate_mean': np.mean([m['avg_auto_rate'] for m in metrics_list]),
            })
    
    df = pd.DataFrame(results)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(df['ensemble_size'], df['augmentation_mean'], marker='o', linewidth=2)
    ax1.axhline(y=3.0, color='red', linestyle='--', label='Target: 3.0x')
    ax1.set_xlabel('Ensemble Size (K)')
    ax1.set_ylabel('Final Augmentation Ratio')
    ax1.set_title('Augmentation vs Ensemble Size')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(df['ensemble_size'], df['auto_rate_mean'], marker='s', linewidth=2, color='green')
    ax2.set_xlabel('Ensemble Size (K)')
    ax2.set_ylabel('Average Auto-label Rate (%)')
    ax2.set_title('Auto-labeling vs Ensemble Size')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ensemble_sweep_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved: ensemble_sweep_results.png")
    
    return df


def sweep_pool_size(sizes=[10, 20, 30, 40, 50], n_seeds=3):
    """
    Sweep over candidate pool size.
    
    Question: How many candidates to sample per query?
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER SWEEP: POOL SIZE")
    print("="*70)
    
    results = []
    
    for pool in sizes:
        print(f"\nTesting pool_size={pool}...")
        
        config = BASE_CONFIG.copy()
        config['pool_size'] = pool
        
        metrics_list = []
        for seed in range(n_seeds):
            try:
                metrics = run_single_config(config, seed=seed + 42)
                metrics_list.append(metrics)
                print(f"  Seed {seed}: Aug={metrics['final_augmentation']:.2f}x")
            except Exception as e:
                print(f"  Seed {seed} FAILED: {e}")
        
        if metrics_list:
            results.append({
                'pool_size': pool,
                'augmentation_mean': np.mean([m['final_augmentation'] for m in metrics_list]),
            })
    
    df = pd.DataFrame(results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['pool_size'], df['augmentation_mean'], marker='o', linewidth=2)
    plt.axhline(y=3.0, color='red', linestyle='--', label='Target: 3.0x')
    plt.xlabel('Pool Size (Candidates per Query)')
    plt.ylabel('Final Augmentation Ratio')
    plt.title('Augmentation vs Pool Size')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('pool_sweep_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved: pool_sweep_results.png")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, choices=['beta', 'ensemble', 'pool', 'all'],
                       default='beta', help='Which hyperparameter to sweep')
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds')
    args = parser.parse_args()
    
    if args.sweep == 'beta' or args.sweep == 'all':
        sweep_beta(n_seeds=args.seeds)
    
    if args.sweep == 'ensemble' or args.sweep == 'all':
        sweep_ensemble_size(n_seeds=args.seeds)
    
    if args.sweep == 'pool' or args.sweep == 'all':
        sweep_pool_size(n_seeds=args.seeds)
    
    print("\n" + "="*70)
    print("HYPERPARAMETER SWEEP COMPLETE")
    print("="*70)