import sys
import os
import numpy as np
import torch

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.envs.wrappers import make_env
from src.agents.entropy_agents import SimpleEntropySAC
from src.agents.sac import SAC
from src.graph.preference_graph import PreferenceGraph
from src.models.reward_ensemble import RewardEnsemble
from src.acquisition.ucb_lcb import UCBLCBFilter
from src.acquisition.sampling import ActiveDethroning
from src.envs.oracle import Oracle
from src.utils.replay_buffer import ReplayBuffer

def run_demo():
    print("===================================================")
    print("🚀 STARTING CAPSTONE HYBRID FRAMEWORK DEMO")
    print("===================================================")
    
    # --- CONFIGURATION ---
    ENV_NAME = "CartPole-v1"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WARMUP_EPISODES = 20      # Small number for demo
    QUERY_BUDGET = 10         # Small budget for demo
    SAC_STEPS = 1000          # Short training for demo
    SEGMENT_LEN = 50
    
    # --- INITIALIZATION ---
    env_wrapper = make_env(ENV_NAME, segment_length=SEGMENT_LEN)
    state_dim = env_wrapper.state_dim
    action_dim = env_wrapper.action_dim
    
    oracle = Oracle(ENV_NAME)
    pref_graph = PreferenceGraph()
    
    # Config dict for ensemble
    ensemble_config = {'ensemble_size': 3, 'lr': 3e-4} # K=3 for speed
    reward_model = RewardEnsemble(state_dim, action_dim, ensemble_config, DEVICE)
    
    ucb_filter = UCBLCBFilter(beta=3.0)
    dethroner = ActiveDethroning()
    
    # --- PHASE 1: UNSUPERVISED WARMUP (PEBBLE) ---
    print("\n[PHASE 1] Unsupervised Warmup (Entropy Maximization)...")
    entropy_agent = SimpleEntropySAC(state_dim, action_dim, env_wrapper.action_range, device=DEVICE)
    
    # This collects trajectories driven by intrinsic entropy reward
    trajectories = entropy_agent.warmup(env_wrapper.env, n_episodes=WARMUP_EPISODES)
    
    # Flatten trajectories into a Segment Pool
    # A "segment" is just a snippet of a trajectory
    segment_pool = []
    for (states, actions, rewards) in trajectories:
        # Slice trajectory into segments
        length = len(states)
        for i in range(0, length - SEGMENT_LEN, SEGMENT_LEN):
            seg = {
                'states': states[i:i+SEGMENT_LEN],
                'actions': actions[i:i+SEGMENT_LEN],
                'rewards': rewards[i:i+SEGMENT_LEN],
                'true_return': np.sum(rewards[i:i+SEGMENT_LEN]) # Proxy for ground truth
            }
            segment_pool.append(seg)
            
    print(f"-> Generated {len(segment_pool)} candidate segments.")

    # --- PHASE 2: ACTIVE REWARD LEARNING ---
    print("\n[PHASE 2] Active Preference Learning Loop...")
    
    # Pick initial defender (random)
    defender_idx = np.random.randint(len(segment_pool))
    defender = segment_pool[defender_idx]
    
    # Pre-train ensemble slightly on dummy data to initialize? 
    # Usually we need at least 1 query to start training.
    # Let's make 1 random query to start.
    chal_idx = (defender_idx + 1) % len(segment_pool)
    challenger = segment_pool[chal_idx]
    
    # Query Oracle
    label = oracle.compare(defender['true_return'], challenger['true_return'])
    if label == 1: # Defender (Seg 1) wins
        pref_graph.add_preference(defender_idx, chal_idx)
    else: # Challenger (Seg 2) wins
        pref_graph.add_preference(chal_idx, defender_idx)
        defender = challenger
        defender_idx = chal_idx
        
    # Training Loop
    for query_i in range(QUERY_BUDGET):
        # 1. Train Ensemble on current Graph
        training_pairs = pref_graph.get_training_pairs()
        
        # Convert indices back to tensors for training
        # (In real code, you'd batch this properly)
        total_loss = 0
        for _ in range(10): # Train steps
            # Sample a pair
            idxA, idxB = training_pairs[np.random.randint(len(training_pairs))]
            
            s0 = torch.FloatTensor(segment_pool[idxA]['states']).unsqueeze(0).to(DEVICE)
            a0 = torch.FloatTensor(segment_pool[idxA]['actions']).unsqueeze(0).to(DEVICE)
            s1 = torch.FloatTensor(segment_pool[idxB]['states']).unsqueeze(0).to(DEVICE)
            a1 = torch.FloatTensor(segment_pool[idxB]['actions']).unsqueeze(0).to(DEVICE)
            label_t = torch.FloatTensor([0.0]).to(DEVICE) # In graph, first is always winner, so label 0 (s0 > s1)?
            # Wait, reward_ensemble expects label=1 if s1 > s0.
            # If graph stores (winner, loser), then winner is s0, so s0 > s1.
            # So label should be 0 (assuming logic: logits = r1 - r0).
            
            loss = reward_model.train_step(s0, a0, s1, a1, label_t)
            total_loss += loss
            
        # 2. Sample Candidates
        candidate_indices = np.random.choice(len(segment_pool), 10, replace=False)
        
        # 3. Predict Rewards & Uncertainty
        cand_mus, cand_stds = [], []
        for idx in candidate_indices:
            s = torch.FloatTensor(segment_pool[idx]['states']).unsqueeze(0).to(DEVICE)
            a = torch.FloatTensor(segment_pool[idx]['actions']).unsqueeze(0).to(DEVICE)
            std = reward_model.predict_uncertainty(s, a).item()
            # Mean logic needed here from ensemble (simple mean of means)
            # Using a hack for demo:
            with torch.no_grad():
                r_preds = [m(s,a).sum().item() for m in reward_model.models]
                mu = np.mean(r_preds)
            cand_mus.append(mu)
            cand_stds.append(std)
            
        # Defender stats
        s_def = torch.FloatTensor(defender['states']).unsqueeze(0).to(DEVICE)
        a_def = torch.FloatTensor(defender['actions']).unsqueeze(0).to(DEVICE)
        def_std = reward_model.predict_uncertainty(s_def, a_def).item()
        with torch.no_grad():
             def_mu = np.mean([m(s_def, a_def).sum().item() for m in reward_model.models])

        # 4. UCB/LCB Filter
        auto_labels, uncertain_idxs, dethroned, new_def_id = ucb_filter.filter_candidates(
            def_mu, def_std, 
            np.array(cand_mus), np.array(cand_stds), 
            candidate_indices
        )
        
        print(f"   Query {query_i+1}: {len(auto_labels)} Auto-labels, {len(uncertain_idxs)} Uncertain")
        
        # Add auto-labels to graph
        for winner, loser in auto_labels:
            w_id = defender_idx if winner == 'defender' else winner
            l_id = defender_idx if loser == 'defender' else loser
            pref_graph.add_preference(w_id, l_id, is_human_label=False)
            
        # 5. Active Dethroning (if needed)
        if len(uncertain_idxs) > 0:
            best_idx, score, _, _ = dethroner.select_best_challenger(
                def_mu, def_std, 
                np.array(cand_mus)[uncertain_idxs], np.array(cand_stds)[uncertain_idxs], 
                np.array(candidate_indices)[uncertain_idxs]
            )
            
            # Query Oracle
            chal = segment_pool[best_idx]
            # Note: Oracle compares True Return, not predicted
            label = oracle.compare(defender['true_return'], chal['true_return'])
            
            if label == 1: # Defender wins
                pref_graph.add_preference(defender_idx, best_idx)
            else:
                pref_graph.add_preference(best_idx, defender_idx)
                defender = chal
                defender_idx = best_idx
                print("   -> Defender Dethroned by Oracle!")

    # --- PHASE 3: POLICY LEARNING (SAC) ---
    print("\n[PHASE 3] Policy Learning with Learned Rewards...")
    
    # Initialize SAC
    sac_agent = SAC(state_dim, action_dim, device=DEVICE)
    sac_buffer = ReplayBuffer(state_dim, action_dim, capacity=5000)
    
    # Fill buffer with some random data for training
    s, _ = env_wrapper.reset()
    for _ in range(1000):
        a = env_wrapper.env.action_space.sample()
        # Handle discrete for CartPole
        a_env = int(a > 0.5) if ENV_NAME == 'CartPole-v1' else a
        
        ns, r, done, trunc, _ = env_wrapper.step(a_env)
        # CRITICAL: Use PREDICTED reward, not env reward
        # But for buffer storage, we store dummy first, then relabel
        sac_buffer.add(s, [a], 0.0, ns, done or trunc)
        s = ns
        if done or trunc: s, _ = env_wrapper.reset()
        
    # Relabeling
    print("   Relabeling buffer...")
    states, actions, _, _, _ = sac_buffer.get_all_transitions()
    states_t = torch.FloatTensor(states).to(DEVICE)
    actions_t = torch.FloatTensor(actions).to(DEVICE)
    
    mu, std = reward_model.predict_reward(states_t, actions_t)
    # Use mean - alpha * std (conservative bound) or just mean
    learned_rewards = mu.cpu().numpy()
    sac_buffer.relabel_rewards(learned_rewards)
    
    # Train SAC
    print(f"   Training SAC for {SAC_STEPS} steps...")
    for i in range(SAC_STEPS):
        sac_agent.update(sac_buffer)
        
    print("\n✅ DEMO COMPLETE! All systems functional.")

if __name__ == "__main__":
    run_demo()