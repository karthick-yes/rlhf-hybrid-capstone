import numpy as np
from scipy.stats import norm

def compute_win_prob(mu_def, std_def, mu_chal, std_chal):
    mu_delta = mu_chal - mu_def
    std_delta = np.sqrt(std_def**2 + std_chal**2)
    return norm.cdf(mu_delta / std_delta)

def entropy_score(p):
    return 4 * p * (1 - p)

print("--- Example 1: Exact Tie (The 'Confusion' Case) ---")
d_mu, d_std = 100.0, 5.0
c_mu, c_std = 100.0, 20.0
p = compute_win_prob(d_mu, d_std, c_mu, c_std)
print(f"Defender:   Mean={d_mu}, Std={d_std}")
print(f"Challenger: Mean={c_mu}, Std={c_std}")
print(f"P(Challenger Wins) = {p:.4f}")
print(f"Entropy Score      = {entropy_score(p):.4f}")

print("\n--- Example 2: High Uncertainty (The 'Maybe' Case) ---")
d_mu, d_std = 100.0, 5.0
c_mu, c_std = 105.0, 50.0  # Higher mean, but huge uncertainty
p = compute_win_prob(d_mu, d_std, c_mu, c_std)
print(f"Defender:   Mean={d_mu}, Std={d_std}")
print(f"Challenger: Mean={c_mu}, Std={c_std}")
print(f"P(Challenger Wins) = {p:.4f}")
print(f"Entropy Score      = {entropy_score(p):.4f}")

print("\n--- Example 3: Clear Winner (Low Entropy) ---")
d_mu, d_std = 100.0, 5.0
c_mu, c_std = 150.0, 10.0
p = compute_win_prob(d_mu, d_std, c_mu, c_std)
print(f"Defender:   Mean={d_mu}, Std={d_std}")
print(f"Challenger: Mean={c_mu}, Std={c_std}")
print(f"P(Challenger Wins) = {p:.4f}")
print(f"Entropy Score      = {entropy_score(p):.4f}")
