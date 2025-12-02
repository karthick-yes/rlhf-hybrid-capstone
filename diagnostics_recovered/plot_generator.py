import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Ensure output directory exists
os.makedirs('plots', exist_ok=True)

# 1. Load Data
try:
    query_df = pd.read_csv('diagnostics_recovered/query_log_recovered.csv')
    sac_df = pd.read_csv('diagnostics_recovered/sac_performance_recovered.csv')
    # Fill missing rounds/sources if any (forward fill)
    query_df['round'] = query_df['round'].fillna(method='ffill')
except FileNotFoundError:
    print("Error: CSV files not found. Please upload 'query_log_recovered.csv' and 'sac_performance_recovered.csv'.")
    exit()

# Set global style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})

# --- Plot 1: Ensemble Uncertainty (Why Adaptive Beta didn't trigger) ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=query_df, x='iteration', y='ensemble_std_mean', hue='round', palette='viridis', marker='o')
plt.axhline(y=50.0, color='red', linestyle='--', linewidth=2, label='Adaptive Beta Threshold (50.0)')
plt.text(0, 52, 'Threshold Not Triggered', color='red', fontweight='bold')
plt.title('Ensemble Uncertainty vs. Active Learning Iterations')
plt.ylabel('Mean Ensemble Std Dev')
plt.xlabel('Query Iteration')
plt.legend(title='Round', loc='upper right')
plt.tight_layout()
plt.savefig('plots/ensemble_uncertainty.png', dpi=300)
plt.close()

# --- Plot 2: Active Learning Efficiency (The 37x Multiplier) ---
# We recreate a cumulative view. "graph_edges" is the total knowledge. "total_human_queries" is the cost.
plt.figure(figsize=(10, 6))
# Create a cumulative index for x-axis
query_df['global_step'] = range(len(query_df))

plt.plot(query_df['global_step'], query_df['graph_edges'], label='Total Knowledge (Auto + Human)', color='green', linewidth=3)
plt.plot(query_df['global_step'], query_df['total_human_queries'], label='Human Effort (Queries)', color='blue', linewidth=3, linestyle='--')

# Fill the gap to visualize the "gain"
plt.fill_between(query_df['global_step'], query_df['total_human_queries'], query_df['graph_edges'], color='green', alpha=0.1)

plt.title('Active Learning Efficiency: Knowledge vs. Cost')
plt.ylabel('Count (Pairs)')
plt.xlabel('Active Learning Steps')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/al_efficiency.png', dpi=300)
plt.close()

# --- Plot 3: SAC Performance Recovery (Round 1 to 5) ---
# We use the cumulative steps to show the timeline
plt.figure(figsize=(12, 6))
sac_df['global_step'] = (sac_df['round'] - 1) * 10000 + sac_df['step']

sns.lineplot(data=sac_df, x='global_step', y='avg_reward', color='purple', linewidth=2.5)
plt.axhline(y=21.5, color='gray', linestyle='--', label='Random Baseline (21.5)', linewidth=2)
plt.title('Agent Performance Recovery: Phase 3 Training')
plt.ylabel('Average Episode Reward')
plt.xlabel('Total Training Steps')
plt.legend()
plt.tight_layout()
plt.savefig('plots/sac_recovery.png', dpi=300)
plt.close()

print("Plots generated in 'plots/' directory.")