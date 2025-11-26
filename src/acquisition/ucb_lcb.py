import numpy as np
import matplotlib.pyplot as plt

class UCBLCBFilter:
    """
    UCB/LCB-based pseudo-labeling for trajectory preferences.
    
    Rule:
    - LCB(defender) > UCB(challenger) -> Defender wins (auto-label)
    - LCB(challenger) > UCB(defender) -> Challenger wins (auto-label + dethrone)
    - Otherwise -> Uncertain (needs human query)
    
    """
    
    def __init__(self, beta=3.0, adaptive_beta=True):
        """
        Args:
            beta: Confidence interval width (3.0 = 99.7% under Gaussian)
            adaptive_beta: If True, reduce beta when defender is uncertain
        """
        self.beta = beta
        self.adaptive_beta = adaptive_beta
    
    def compute_bounds(self, mu, std, beta=None):
        """
        Compute UCB and LCB.
        
        Args:
            mu: Mean cumulative reward (scalar or array)
            std: Std dev (scalar or array)
            beta: Override default beta (optional)
        
        Returns:
            ucb, lcb: Upper and lower bounds
        """
        if beta is None:
            beta = self.beta
            
        ucb = mu + beta * std
        lcb = mu - beta * std
        return ucb, lcb
    
    def filter_candidates(self, defender_mu, defender_std, 
                         candidate_mus, candidate_stds, candidate_ids):
        """
        Apply UCB/LCB filtering with adaptive beta.
        
        Args:
            defender_mu, defender_std: Defender statistics
            candidate_mus, candidate_stds: Arrays [n_candidates]
            candidate_ids: List of trajectory IDs
        
        Returns:
            auto_labels: List[(winner_id, loser_id)] - auto-labeled pairs
            uncertain_indices: List[int] - indices needing human query
            dethroned: bool - True if defender was dethroned
            new_defender_id: int or None - New defender if dethroned
        """
        
        if self.adaptive_beta and defender_std > 1.0:
            effective_beta = self.beta * 1.0 # Widen intervals
            print(f"   [Adaptive Beta] Defender std={defender_std:.2f} > 1.0, using beta={effective_beta:.1f}")
        else:
            effective_beta = self.beta
        
        def_ucb, def_lcb = self.compute_bounds(defender_mu, defender_std, effective_beta)
        
        auto_labels = []
        uncertain_indices = []
        dethroned = False
        new_defender_id = None
        
        for i, cand_id in enumerate(candidate_ids):
            cand_mu = candidate_mus[i]
            cand_std = candidate_stds[i]
            cand_ucb, cand_lcb = self.compute_bounds(cand_mu, cand_std, effective_beta)
            
            # Case 1: Defender clearly better
            if def_lcb > cand_ucb:
                auto_labels.append(('defender', cand_id))
            
            # Case 2: Challenger clearly better -> DETHRONE
            elif cand_lcb > def_ucb:
                auto_labels.append((cand_id, 'defender'))
                dethroned = True
                new_defender_id = cand_id
                print(f"   [DETHRONING] Challenger {cand_id} beats Defender (LCB={cand_lcb:.2f} > UCB={def_ucb:.2f})")
                break  # Stop, defender changed
            
            # Case 3: Intervals overlap -> uncertain
            else:
                uncertain_indices.append(i)
        
        return auto_labels, uncertain_indices, dethroned, new_defender_id
    
    def visualize_bounds(self, defender_mu, defender_std, 
                        candidate_mus, candidate_stds, 
                        labels, save_path='ucb_lcb_viz.png'):
        """
        Visualize confidence intervals.
        
        Args:
            labels: List of ['auto_def', 'auto_chal', 'uncertain'] for each candidate
        """
        n = len(candidate_mus)
        def_ucb, def_lcb = self.compute_bounds(defender_mu, defender_std)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Defender (gold star at x=0)
        ax.errorbar(0, defender_mu, yerr=self.beta*defender_std, 
                   fmt='*', markersize=15, capsize=8, 
                   color='gold', label='Defender', linewidth=3, zorder=10)
        
        # Candidates
        colors = {'auto_def': 'red', 'auto_chal': 'green', 'uncertain': 'gray'}
        label_names = {'auto_def': 'Auto: Def Wins', 'auto_chal': 'Auto: Chal Wins', 'uncertain': 'Uncertain'}
        
        # Plot by category
        for label_type in ['auto_def', 'auto_chal', 'uncertain']:
            indices = [i for i, lbl in enumerate(labels) if lbl == label_type]
            if indices:
                x_pos = [i+1 for i in indices]
                mus = [candidate_mus[i] for i in indices]
                errs = [self.beta*candidate_stds[i] for i in indices]
                
                ax.errorbar(x_pos, mus, yerr=errs,
                           fmt='o', markersize=8, capsize=5,
                           color=colors[label_type], alpha=0.7,
                           label=label_names[label_type], linewidth=2)
        
        # Reference lines
        ax.axhline(y=def_ucb, color='gold', linestyle='--', alpha=0.4, linewidth=2)
        ax.axhline(y=def_lcb, color='gold', linestyle='--', alpha=0.4, linewidth=2)
        
        ax.set_xlabel('Trajectory Index (0 = Defender)', fontsize=13)
        ax.set_ylabel('Cumulative Reward Estimate', fontsize=13)
        ax.set_title(f'UCB/LCB Filtering (beta={self.beta})', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3, linestyle=':')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved: {save_path}")
        plt.close()