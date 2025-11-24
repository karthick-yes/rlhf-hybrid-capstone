import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.reward_ensemble import RewardEnsemble

def test_ensemble_calibration():
    print("\n--- Checkpoint 1: Ensemble Calibration (Bootstrapped) ---")
    
    # 1. Setup
    # Using a smaller Learning Rate to prevent jumping to saturation immediately
    config = {'ensemble_size': 5, 'lr': 3e-4} 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = RewardEnsemble(state_dim=2, action_dim=1, config=config, device=device)
    
    # 2. Generate Training Data
    # Simple Pattern: Reward is positive if inputs are close to zero
    N_train = 500 # More data
    s0 = torch.randn(N_train, 10, 2).to(device) * 0.5 # Compact cluster
    a0 = torch.randn(N_train, 10, 1).to(device) * 0.5
    s1 = torch.randn(N_train, 10, 2).to(device) * 0.5
    a1 = torch.randn(N_train, 10, 1).to(device) * 0.5
    
    # Ground Truth: Preference depends on distance from zero
    # Closer to zero = Better
    r0_proxy = -s0.abs().sum(dim=1).sum(dim=1)
    r1_proxy = -s1.abs().sum(dim=1).sum(dim=1)
    labels = (r1_proxy > r0_proxy).float().to(device)
    
    # 3. Train with BOOTSTRAPPING (The Fix)
    # Each model sees a different random subset of data
    print("Training ensemble with bootstrapping...")
    for epoch in range(300): # Train longer
        
        for k in range(ensemble.K):
            # Create a random mask for this specific model 'k'
            # This ensures Model 1 sees different data than Model 2
            indices = torch.randint(0, N_train, (N_train,), device=device)
            
            # Sample batched data
            b_s0, b_a0 = s0[indices], a0[indices]
            b_s1, b_a1 = s1[indices], a1[indices]
            b_labels = labels[indices]
            
            # We manually update model 'k'
            opt = ensemble.optimizers[k]
            opt.zero_grad()
            
            r0 = ensemble.models[k](b_s0, b_a0).sum(dim=1)
            r1 = ensemble.models[k](b_s1, b_a1).sum(dim=1)
            logits = r1 - r0
            loss = torch.nn.BCEWithLogitsLoss()(logits, b_labels.unsqueeze(1))
            
            loss.backward()
            opt.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} complete")
        
    # 4. Validation
    # In-Distribution (ID): Similar to training (Compact cluster)
    val_s = torch.randn(20, 10, 2).to(device) * 0.5
    val_a = torch.randn(20, 10, 1).to(device) * 0.5
    
    # Out-of-Distribution (OOD): CHAOS (Not shifted, just wide spread)
    # Multiplied by 5.0. Some values will be 5, some -5, some 0.
    # This is "confusing" data, not "obviously good" data.
    ood_s = torch.randn(20, 10, 2).to(device) * 5.0 
    ood_a = torch.randn(20, 10, 1).to(device) * 5.0
    
    unc_id = ensemble.predict_uncertainty(val_s, val_a).mean().item()
    unc_ood = ensemble.predict_uncertainty(ood_s, ood_a).mean().item()
    
    print(f"Uncertainty ID: {unc_id:.5f}")
    print(f"Uncertainty OOD: {unc_ood:.5f}")
    
    # 5. Diagnostic
    with torch.no_grad():
        p_ood = ensemble.models[0](ood_s, ood_a).sum(dim=1).mean().item()
        print(f"Avg Prediction (Model 0) on OOD: {p_ood:.4f} (Should not be 10.0)")

    # Assertion
    # We expect OOD uncertainty to be at least 2x ID uncertainty
    assert unc_ood > unc_id, f"Ensemble failed! ID: {unc_id:.4f} vs OOD: {unc_ood:.4f}"
    print("SUCCESS: Deep Ensemble is calibrated.")

if __name__ == "__main__":
    test_ensemble_calibration()