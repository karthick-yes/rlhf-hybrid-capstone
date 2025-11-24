import json
import numpy as np
import matplotlib.pyplot as plt

class ExperimentLogger:
    """
    Logs metrics and generates plots for experiments.
    """
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.metrics = {
            'query_counts': [],
            'auto_label_counts': [],
            'transitive_counts': [],
            'augmentation_ratios': [],
            'defender_changes': []
        }
    
    def log_iteration(self, query_count, auto_count, trans_count, defender_changed):
        """Log metrics for one iteration"""
        self.metrics['query_counts'].append(query_count)
        self.metrics['auto_label_counts'].append(auto_count)
        self.metrics['transitive_counts'].append(trans_count)
        
        total = query_count + auto_count + trans_count
        ratio = total / query_count if query_count > 0 else 0
        self.metrics['augmentation_ratios'].append(ratio)
        self.metrics['defender_changes'].append(1 if defender_changed else 0)
    
    def plot_augmentation_over_time(self, save_path='augmentation_curve.png'):
        """Plot augmentation ratio over queries"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        queries = np.cumsum(self.metrics['query_counts'])
        ratios = self.metrics['augmentation_ratios']
        
        ax.plot(queries, ratios, linewidth=2.5, color='purple', marker='o')
        ax.axhline(y=3.0, color='red', linestyle='--', label='Target: 3.0x', linewidth=2)
        ax.set_xlabel('Cumulative Human Queries', fontsize=13)
        ax.set_ylabel('Augmentation Ratio', fontsize=13)
        ax.set_title('Data Augmentation Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, linestyle=':')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[LOG] Saved: {save_path}")
        plt.close()
    
    def save_metrics(self, filepath='metrics.json'):
        """Save metrics to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[LOG] Metrics saved: {filepath}")