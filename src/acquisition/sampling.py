import numpy as np
from scipy.stats import norm

class ActiveDethroning:
    """
    Active Dethroning: Select challenger with maximum information gain.
    
    Strategy: Choose pair where P(challenger wins) ≈ 0.5 (max entropy).
    This targets the decision boundary.
    """
    
    def __init__(self):
        pass
    
    def compute_win_probability(self, mu_def, std_def, mu_chal, std_chal):
        """
        Compute P(challenger > defender) under Gaussian assumption.
        
        Model:
            Δ = r_chal - r_def ~ N(μ_Δ, σ_Δ²)
            P(Win) = P(Δ > 0) = Φ(μ_Δ / σ_Δ)
        
        Args:
            mu_def, std_def: Defender statistics
            mu_chal, std_chal: Challenger statistics
        
        Returns:
            p_win: Probability ∈ [0, 1]
        """
        mu_delta = mu_chal - mu_def
        std_delta = np.sqrt(std_def**2 + std_chal**2)
        
        if std_delta < 1e-6:
            return 0.5  # No uncertainty, assume 50-50
        
        p_win = norm.cdf(mu_delta / std_delta)
        return p_win
    
    def max_entropy_score(self, p_win):
        """
        Binary entropy score: proportional to p(1-p).
        
        Peaks at p=0.5, normalized to [0, 1] by multiplying by 4.
        """
        return 4 * p_win * (1 - p_win)
    
    def select_best_challenger(self, defender_mu, defender_std,
                               candidate_mus, candidate_stds, candidate_indices):
        """
        Select candidate with highest entropy (closest to 50-50).
        
        Args:
            defender_mu, defender_std: Defender stats
            candidate_mus, candidate_stds: Arrays [n_candidates]
            candidate_indices: Original indices in full candidate list
        
        Returns:
            best_index: Index in original list
            best_score: Entropy score [0, 1]
            best_p_win: Win probability [0, 1]
            all_scores: Array of all scores (for debugging)
        """
        if len(candidate_indices) == 0:
            return None, 0.0, 0.5, np.array([])
        
        scores = []
        p_wins = []
        
        for i in range(len(candidate_indices)):
            p_win = self.compute_win_probability(
                defender_mu, defender_std,
                candidate_mus[i], candidate_stds[i]
            )
            score = self.max_entropy_score(p_win)
            
            scores.append(score)
            p_wins.append(p_win)
        
        scores = np.array(scores)
        p_wins = np.array(p_wins)
        
        best_local_idx = np.argmax(scores)
        best_original_idx = candidate_indices[best_local_idx]
        
        return best_original_idx, scores[best_local_idx], p_wins[best_local_idx], scores