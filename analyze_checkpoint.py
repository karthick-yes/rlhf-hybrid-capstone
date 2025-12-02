import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

class ExperimentRecovery:
    """Recover and regenerate all diagnostics from checkpoints and logs"""
    
    def __init__(self, checkpoint_dir="checkpoints", log_file="final_results.txt"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_file = Path(log_file)
        self.data = {
            'query_log': [],
            'defender_changes': [],
            'dual_agent_stats': [],
            'reward_correlations': [],
            'sac_performance': [],
            'auto_label_accuracy': []
        }
        
    def load_all_checkpoints(self):
        """Load all available checkpoints and extract data"""
        print("=" * 70)
        print("LOADING CHECKPOINTS")
        print("=" * 70)
        
        checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        for ckpt_path in checkpoint_files:
            print(f"\nLoading: {ckpt_path.name}")
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                
                # Extract available data
                if 'query_log' in ckpt and ckpt['query_log']:
                    self.data['query_log'].extend(ckpt['query_log'])
                    print(f"  → Found {len(ckpt['query_log'])} query log entries")
                
                if 'loop_stats' in ckpt and ckpt['loop_stats']:
                    self.data['dual_agent_stats'].extend(ckpt['loop_stats'])
                    print(f"  → Found {len(ckpt['loop_stats'])} dual-agent stats")
                
                if 'reward_correlations' in ckpt and ckpt['reward_correlations']:
                    self.data['reward_correlations'].extend(ckpt['reward_correlations'])
                    print(f"  → Found {len(ckpt['reward_correlations'])} correlation records")
                
                if 'sac_performance' in ckpt and ckpt['sac_performance']:
                    self.data['sac_performance'].extend(ckpt['sac_performance'])
                    print(f"  → Found {len(ckpt['sac_performance'])} SAC performance records")
                    
            except Exception as e:
                print(f"  ✗ Error loading {ckpt_path.name}: {e}")
        
        # Remove duplicates and sort
        self._deduplicate_data()
        
        print("\n" + "=" * 70)
        print("CHECKPOINT LOADING COMPLETE")
        print("=" * 70)
        self._print_data_summary()
    
    def _deduplicate_data(self):
        """Remove duplicate entries from loaded data"""
        # Query log: dedupe by (round, iteration)
        if self.data['query_log']:
            df = pd.DataFrame(self.data['query_log'])
            df = df.drop_duplicates(subset=['round', 'iteration'], keep='last')
            self.data['query_log'] = df.to_dict('records')
        
        # Dual-agent stats: dedupe by round
        if self.data['dual_agent_stats']:
            df = pd.DataFrame(self.data['dual_agent_stats'])
            df = df.drop_duplicates(subset=['round'], keep='last')
            self.data['dual_agent_stats'] = df.to_dict('records')
        
        # Correlations: dedupe by iteration
        if self.data['reward_correlations']:
            df = pd.DataFrame(self.data['reward_correlations'])
            df = df.drop_duplicates(subset=['iteration'], keep='last')
            self.data['reward_correlations'] = df.to_dict('records')
        
        # SAC performance: dedupe by (round, step)
        if self.data['sac_performance']:
            df = pd.DataFrame(self.data['sac_performance'])
            df = df.drop_duplicates(subset=['round', 'step'], keep='last')
            self.data['sac_performance'] = df.to_dict('records')
    
    def parse_terminal_output(self):
        """Parse the terminal output to extract defender changes and other stats"""
        print("\n" + "=" * 70)
        print("PARSING TERMINAL OUTPUT")
        print("=" * 70)
        
        if not self.log_file.exists():
            print(f"✗ Log file not found: {self.log_file}")
            return
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract defender changes
        defender_changes = []
        
        # Pattern 1: Natural dethronement
        natural_pattern = r"NATURAL DETHRONE\s+(\d+)\((\w+),.*?→\s+(\d+)\((\w+)"
        for match in re.finditer(natural_pattern, content):
            defender_changes.append({
                'type': 'Natural Dethrone',
                'old_id': int(match.group(1)),
                'old_source': match.group(2),
                'new_id': int(match.group(3)),
                'new_source': match.group(4)
            })
        
        # Pattern 2: Human query dethronement
        human_pattern = r"HUMAN QUERY DETHRONE\s+(\d+)\((\w+),.*?>.*?Defender\s+(\d+)\((\w+)"
        for match in re.finditer(human_pattern, content):
            defender_changes.append({
                'type': 'Human Query Dethrone',
                'new_id': int(match.group(1)),
                'new_source': match.group(2),
                'old_id': int(match.group(3)),
                'old_source': match.group(4)
            })
        
        # Pattern 3: PageRank reset
        pagerank_pattern = r"DEFENDER RESET \(PageRank\)\s+(\d+)\((\w+),.*?→\s+(\d+)\((\w+)"
        for match in re.finditer(pagerank_pattern, content):
            defender_changes.append({
                'type': 'PageRank Reset',
                'old_id': int(match.group(1)),
                'old_source': match.group(2),
                'new_id': int(match.group(3)),
                'new_source': match.group(4)
            })
        
        self.data['defender_changes'] = defender_changes
        
        print(f"\nFound {len(defender_changes)} defender changes:")
        for i, change in enumerate(defender_changes, 1):
            print(f"  {i}. {change['type']}: {change.get('old_id', 'N/A')}({change.get('old_source', 'N/A')}) "
                  f"→ {change['new_id']}({change['new_source']})")
        
        # Extract round summaries
        self._parse_round_summaries(content)
        
        print("\n" + "=" * 70)
        print("TERMINAL PARSING COMPLETE")
        print("=" * 70)
    
    def _parse_round_summaries(self, content):
        """Extract detailed round summaries from terminal output"""
        round_pattern = r"ROUND (\d+) SUMMARY.*?Defender Changes: (\d+).*?SAC Wins:\s+(\d+).*?Entropy Wins:\s+(\d+).*?SAC Auto-Reject:\s+(\d+).*?Entropy Auto-Reject:\s+(\d+)"
        
        for match in re.finditer(round_pattern, content, re.DOTALL):
            round_num = int(match.group(1))
            
            # Check if we already have this data from checkpoints
            existing = next((s for s in self.data['dual_agent_stats'] if s['round'] == round_num), None)
            
            if not existing:
                self.data['dual_agent_stats'].append({
                    'round': round_num,
                    'defender_changes': int(match.group(2)),
                    'SAC_Wins': int(match.group(3)),
                    'Entropy_Wins': int(match.group(4)),
                    'SAC_Auto': int(match.group(5)),
                    'Entropy_Auto': int(match.group(6))
                })
    
    def _print_data_summary(self):
        """Print summary of loaded data"""
        print("\nData Summary:")
        print(f"  Query Log Entries:        {len(self.data['query_log'])}")
        print(f"  Dual-Agent Stats:         {len(self.data['dual_agent_stats'])}")
        print(f"  Reward Correlations:      {len(self.data['reward_correlations'])}")
        print(f"  SAC Performance Records:  {len(self.data['sac_performance'])}")
        print(f"  Defender Changes:         {len(self.data['defender_changes'])}")
    
    def generate_all_visualizations(self, output_dir="diagnostics_recovered"):
        """Generate all diagnostic plots with recovered data"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # 1. Main diagnostics (6-panel)
        if self.data['query_log'] and self.data['reward_correlations'] and self.data['sac_performance']:
            self._plot_main_diagnostics(output_dir)
        else:
            print("✗ Insufficient data for main diagnostics")
        
        # 2. Dual-agent statistics
        if self.data['dual_agent_stats']:
            self._plot_dual_agent_stats(output_dir)
        else:
            print("✗ Insufficient data for dual-agent stats")
        
        # 3. Defender timeline
        if self.data['defender_changes']:
            self._plot_defender_timeline(output_dir)
        else:
            print("✗ No defender changes to plot")
        
        # 4. Detailed round-by-round analysis
        if self.data['query_log']:
            self._plot_round_analysis(output_dir)
        else:
            print("✗ Insufficient data for round analysis")
        
        print(f"\n✓ All visualizations saved to: {output_dir}/")
    
    def _plot_main_diagnostics(self, output_dir):
        """Regenerate the main 6-panel diagnostic plot"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        df_query = pd.DataFrame(self.data['query_log'])
        df_corr = pd.DataFrame(self.data['reward_correlations'])
        df_perf = pd.DataFrame(self.data['sac_performance'])
        
        # Plot 1: Augmentation Ratio
        ax1 = fig.add_subplot(gs[0, 0])
        for round_num in df_query['round'].unique():
            round_data = df_query[df_query['round'] == round_num]
            ax1.plot(round_data['iteration'], round_data['augmentation'], 
                    marker='o', markersize=3, alpha=0.7, label=f'Round {round_num}')
        ax1.axhline(y=3.0, color='red', linestyle='--', label='Target: 3.0x', alpha=0.5)
        ax1.set_xlabel('Iteration Number')
        ax1.set_ylabel('Augmentation Ratio')
        ax1.set_title('Data Augmentation Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Auto-label Rate
        ax2 = fig.add_subplot(gs[0, 1])
        for round_num in df_query['round'].unique():
            round_data = df_query[df_query['round'] == round_num]
            ax2.plot(round_data['iteration'], round_data['auto_rate'], 
                    marker='s', markersize=3, alpha=0.7, label=f'Round {round_num}')
        ax2.set_xlabel('Iteration Number')
        ax2.set_ylabel('Auto-label Rate (%)')
        ax2.set_title('UCB/LCB Filter Efficiency', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Graph Growth
        ax3 = fig.add_subplot(gs[1, 0])
        for round_num in df_query['round'].unique():
            round_data = df_query[df_query['round'] == round_num]
            ax3.plot(round_data['iteration'], round_data['graph_edges'], 
                    marker='^', markersize=3, alpha=0.7, label=f'Round {round_num}')
        ax3.set_xlabel('Iteration Number')
        ax3.set_ylabel('Total Graph Edges')
        ax3.set_title('Preference Graph Growth', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Ensemble Uncertainty
        ax4 = fig.add_subplot(gs[1, 1])
        for round_num in df_query['round'].unique():
            round_data = df_query[df_query['round'] == round_num]
            ax4.plot(round_data['iteration'], round_data['ensemble_std_mean'], 
                    marker='d', markersize=3, alpha=0.7, label=f'Round {round_num}')
        ax4.set_xlabel('Iteration Number')
        ax4.set_ylabel('Mean Uncertainty (σ)')
        ax4.set_title('Ensemble Uncertainty Over Time', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # Plot 5: Reward Model Accuracy
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(df_corr['iteration'], df_corr['correlation'], 
                linewidth=2, color='crimson', marker='o', markersize=4)
        ax5.axhline(y=0.9, color='green', linestyle='--', label='Target: 0.9', alpha=0.5)
        ax5.set_xlabel('Human Queries')
        ax5.set_ylabel('Spearman Correlation')
        ax5.set_title('Reward Model Accuracy', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        ax5.set_ylim([0.85, 1.05])
        
        # Plot 6: Agent Performance
        ax6 = fig.add_subplot(gs[2, 1])
        scatter = ax6.scatter(df_perf['step'], df_perf['avg_reward'], 
                            c=df_perf['round'], cmap='viridis', 
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add trendline per round
        for round_num in df_perf['round'].unique():
            round_data = df_perf[df_perf['round'] == round_num]
            ax6.plot(round_data['step'], round_data['avg_reward'], 
                    linewidth=1.5, alpha=0.5)
        
        ax6.set_xlabel('Training Steps')
        ax6.set_ylabel('Average Episode Reward')
        ax6.set_title('Agent Performance During Training', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Round')
        ax6.grid(alpha=0.3)
        
        plt.savefig(f"{output_dir}/main_diagnostics.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: main_diagnostics.png")
        plt.close()
    
    def _plot_dual_agent_stats(self, output_dir):
        """Plot dual-agent statistics"""
        df = pd.DataFrame(self.data['dual_agent_stats'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dual-Agent Statistics Across Rounds', fontsize=14, fontweight='bold')
        
        rounds = df['round'].values
        
        # Plot 1: Wins
        ax1 = axes[0, 0]
        ax1.bar(rounds - 0.15, df['SAC_Wins'], width=0.3, label='SAC Wins', color='steelblue')
        ax1.bar(rounds + 0.15, df['Entropy_Wins'], width=0.3, label='Entropy Wins', color='coral')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Number of Wins')
        ax1.set_title('Defender Dethronements by Agent', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # Plot 2: Auto-Rejections
        ax2 = axes[0, 1]
        ax2.bar(rounds - 0.15, df['SAC_Auto'], width=0.3, label='SAC Auto-Reject', color='lightcoral')
        ax2.bar(rounds + 0.15, df['Entropy_Auto'], width=0.3, label='Entropy Auto-Reject', color='lightsalmon')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Number of Auto-Rejections')
        ax2.set_title('UCB/LCB Auto-Rejections by Agent', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        
        # Plot 3: Defender Changes per Round
        ax3 = axes[1, 0]
        if 'defender_changes' in df.columns:
            ax3.bar(rounds, df['defender_changes'], color='purple', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Number of Defender Changes')
            ax3.set_title('Defender Stability', fontweight='bold')
            ax3.grid(alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'Defender change data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Total Engagement
        ax4 = axes[1, 1]
        sac_total = df.get('SAC_Query', pd.Series([0]*len(df))) + df['SAC_Auto']
        entropy_total = df.get('Entropy_Query', pd.Series([0]*len(df))) + df['Entropy_Auto']
        
        ax4.plot(rounds, sac_total, marker='o', linewidth=2, label='SAC Total', color='steelblue')
        ax4.plot(rounds, entropy_total, marker='s', linewidth=2, label='Entropy Total', color='coral')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Total Interactions')
        ax4.set_title('Agent Engagement Over Time', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dual_agent_stats.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: dual_agent_stats.png")
        plt.close()
    
    def _plot_defender_timeline(self, output_dir):
        """Create a timeline visualization of defender changes"""
        if not self.data['defender_changes']:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        changes = self.data['defender_changes']
        
        # Create timeline
        for i, change in enumerate(changes):
            color = {'Natural Dethrone': 'green', 
                    'Human Query Dethrone': 'red',
                    'PageRank Reset': 'blue'}.get(change['type'], 'gray')
            
            ax.scatter(i, 0, s=200, c=color, alpha=0.7, edgecolors='black', linewidth=2)
            ax.text(i, 0.05, f"{change['new_id']}\n({change['new_source']})", 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i, -0.05, change['type'].replace(' ', '\n'), 
                   ha='center', va='top', fontsize=7, alpha=0.7)
        
        ax.set_ylim([-0.15, 0.15])
        ax.set_xlim([-0.5, len(changes) - 0.5])
        ax.set_xlabel('Change Event Number', fontsize=12)
        ax.set_title('Defender Change Timeline', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_yticks([])
        ax.grid(alpha=0.3, axis='x')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Natural Dethrone'),
            Patch(facecolor='red', label='Human Query'),
            Patch(facecolor='blue', label='PageRank Reset')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/defender_timeline.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: defender_timeline.png")
        plt.close()
    
    def _plot_round_analysis(self, output_dir):
        """Create detailed per-round analysis"""
        df = pd.DataFrame(self.data['query_log'])
        
        # FIX: Convert to int to avoid numpy float issue
        n_rounds = int(df['round'].max())
        
        fig, axes = plt.subplots(n_rounds, 1, figsize=(12, 3 * n_rounds))
        
        if n_rounds == 1:
            axes = [axes]
        
        for i, round_num in enumerate(range(1, n_rounds + 1)):
            round_data = df[df['round'] == round_num]
            
            ax = axes[i]
            ax2 = ax.twinx()
            
            # Auto-rate
            ax.plot(round_data['iteration'], round_data['auto_rate'], 
                   color='green', marker='o', label='Auto-Rate (%)', linewidth=2)
            ax.set_ylabel('Auto-Label Rate (%)', color='green')
            ax.tick_params(axis='y', labelcolor='green')
            
            # Graph edges
            ax2.plot(round_data['iteration'], round_data['graph_edges'], 
                    color='blue', marker='s', label='Graph Edges', linewidth=2)
            ax2.set_ylabel('Graph Edges', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            ax.set_xlabel('Iteration')
            ax.set_title(f'Round {round_num} Detailed Analysis', fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Mark defender changes
            if 'defender_changed' in round_data.columns:
                change_points = round_data[round_data['defender_changed'] == True]
                for _, point in change_points.iterrows():
                    ax.axvline(x=point['iteration'], color='red', 
                             linestyle='--', alpha=0.5, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/round_analysis.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: round_analysis.png")
        plt.close()
    
    def export_csv_reports(self, output_dir="diagnostics_recovered"):
        """Export all data to CSV files"""
        print("\n" + "=" * 70)
        print("EXPORTING CSV REPORTS")
        print("=" * 70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Query log
        if self.data['query_log']:
            df = pd.DataFrame(self.data['query_log'])
            path = f"{output_dir}/query_log_recovered.csv"
            df.to_csv(path, index=False)
            print(f"✓ Exported: {path}")
        
        # Dual-agent stats
        if self.data['dual_agent_stats']:
            df = pd.DataFrame(self.data['dual_agent_stats'])
            path = f"{output_dir}/dual_agent_stats_recovered.csv"
            df.to_csv(path, index=False)
            print(f"✓ Exported: {path}")
            print("\nDual-Agent Stats Summary:")
            print(df.to_string(index=False))
        
        # Defender changes
        if self.data['defender_changes']:
            df = pd.DataFrame(self.data['defender_changes'])
            path = f"{output_dir}/defender_changes.csv"
            df.to_csv(path, index=False)
            print(f"\n✓ Exported: {path}")
            print("\nDefender Changes:")
            print(df.to_string(index=False))
        
        # Correlations
        if self.data['reward_correlations']:
            df = pd.DataFrame(self.data['reward_correlations'])
            path = f"{output_dir}/reward_correlations_recovered.csv"
            df.to_csv(path, index=False)
            print(f"\n✓ Exported: {path}")
        
        # SAC performance
        if self.data['sac_performance']:
            df = pd.DataFrame(self.data['sac_performance'])
            path = f"{output_dir}/sac_performance_recovered.csv"
            df.to_csv(path, index=False)
            print(f"\n✓ Exported: {path}")
    
    def run_full_recovery(self):
        """Execute complete recovery pipeline"""
        print("\n" + "#" * 70)
        print("##  EXPERIMENT DATA RECOVERY & REGENERATION")
        print("#" * 70 + "\n")
        
        # Step 1: Load checkpoints
        self.load_all_checkpoints()
        
        # Step 2: Parse terminal output
        self.parse_terminal_output()
        
        # Step 3: Generate visualizations
        self.generate_all_visualizations()
        
        # Step 4: Export CSV reports
        self.export_csv_reports()
        
        print("\n" + "#" * 70)
        print("##  RECOVERY COMPLETE")
        print("#" * 70)
        print(f"\nAll outputs saved to: diagnostics_recovered/")
        print("Review the plots and CSV files to verify accuracy.")


if __name__ == "__main__":
    recovery = ExperimentRecovery(
        checkpoint_dir="checkpoints",
        log_file="final_results.txt"
    )
    recovery.run_full_recovery()