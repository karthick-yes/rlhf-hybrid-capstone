import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ExperimentLogger:
    """
    Advanced Logger for Dual-Agent Hybrid RLHF.
    Tracks detailed diagnostics per generation (SAC vs Entropy).
    """
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # High-level metrics
        self.metrics = {
            'query_counts': [],
            'auto_label_counts': [],
            'transitive_counts': [],
            'augmentation_ratios': [],
            'defender_changes': []
        }
        
        # Detailed Diagnostics
        self.loop_stats = [] 
        self.csv_path = os.path.join(self.log_dir, 'loop_diagnostics.csv')
        
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Round', 'Defender_ID', 'Defender_Source', 
                    'SAC_Wins', 'Entropy_Wins', 
                    'SAC_Auto_Reject', 'Entropy_Auto_Reject',
                    'SAC_Queries', 'Entropy_Queries'
                ])

    def log_iteration(self, query_count, auto_count, trans_count, defender_changed):
        """Log standard graph metrics"""
        self.metrics['query_counts'].append(query_count)
        self.metrics['auto_label_counts'].append(auto_count)
        self.metrics['transitive_counts'].append(trans_count)
        
        total = query_count + auto_count + trans_count
        ratio = total / query_count if query_count > 0 else 0
        self.metrics['augmentation_ratios'].append(ratio)
        self.metrics['defender_changes'].append(1 if defender_changed else 0)
        
        # Save immediately
        self.save_metrics()

    def log_dual_agent_stats(self, round_num, defender_id, source, wins, rejects, queries):
        """
        Log the battle between SAC and Entropy agents.
        """
        row = {
            'Round': round_num,
            'Defender_ID': defender_id,
            'Defender_Source': source,
            'SAC_Wins': wins.get('SAC', 0),
            'Entropy_Wins': wins.get('Entropy', 0),
            'SAC_Auto_Reject': rejects.get('SAC_Auto', 0),
            'Entropy_Auto_Reject': rejects.get('Entropy_Auto', 0),
            'SAC_Queries': queries.get('SAC_Query', 0),
            'Entropy_Queries': queries.get('Entropy_Query', 0)
        }
        self.loop_stats.append(row)
        
        # Append to CSV immediately
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        
    def save_text_summary(self, content):
        """Append text content to a summary log file"""
        path = os.path.join(self.log_dir, "experiment_summary.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(content + "\n")
            
    def save_metrics(self):
        """Save generic metrics to JSON"""
        path = os.path.join(self.log_dir, 'metrics.json')
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def plot_augmentation_over_time(self):
        """Generate the plot"""
        if not self.metrics['query_counts']: return
        
        save_path = os.path.join(self.log_dir, 'augmentation_curve.png')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        queries = np.cumsum(self.metrics['query_counts'])
        ratios = self.metrics['augmentation_ratios']
        
        ax.plot(queries, ratios, linewidth=2.5, color='purple', marker='o')
        ax.axhline(y=3.0, color='red', linestyle='--', label='Target: 3.0x')
        
        ax.set_xlabel('Cumulative Human Queries')
        ax.set_ylabel('Augmentation Ratio')
        ax.set_title('Hybrid Framework Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()