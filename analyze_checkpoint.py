"""
Analyze a trained checkpoint without re-training.
Computes all diagnostics retroactively.

Usage:
    python analyze_checkpoint.py --checkpoint checkpoints/phase3_done.pt --config configs/cartpole.yaml
"""

import sys
import os
import argparse
import yaml
import torch
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.reward_ensemble import RewardEnsemble
from src.graph.preference_graph import PreferenceGraph
from src.envs.wrappers import make_env
from src.utils.replay_buffer import ReplayBuffer
from src.agents.sac import SAC

def load_checkpoint(checkpoint_path, config):
    """Load trained model from checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Reconstruct components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env = make_env(config['env_name'])
    
    ensemble = RewardEnsemble(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=config,
        device=device
    )
    
    # Load ensemble weights
    for i, state_dict in enumerate(checkpoint['ensemble_state']):
        ensemble.models[i].load_state_dict(state_dict)
    
    # Load SAC policy
    sac = SAC(env.state_dim, env.action_dim, device=device)
    sac.actor.load_state_dict(checkpoint['sac_actor'])
    sac.critic.load_state_dict(checkpoint['sac_critic'])
    
    print(f" Loaded model from iteration {checkpoint['total_human_queries']}")
    
    return ensemble, sac, env, checkpoint

def evaluate_policy(env, agent, n_episodes=20, deterministic=True):
    """Evaluate agent performance"""
    scores = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=deterministic)
            
            # Handle discrete CartPole
            if env.env_name == 'CartPole-v1':
                action = int(action > 0.5)
            
            state, reward, term, trunc, _ = env.step(action)
            score += reward
            done = term or trunc
        
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

def compute_reward_model_accuracy(ensemble, env, pref_buffer, device):
    """Compute correlation between predicted and true rewards"""
    all_ids = pref_buffer.get_all_ids()
    
    if len(all_ids) < 5:
        return 0.0
    
    predicted = []
    true = []
    
    for tid in all_ids:
        traj = pref_buffer.get_trajectory(tid)
        
        states = torch.FloatTensor(traj['states']).unsqueeze(0).to(device)
        actions = torch.FloatTensor(traj['actions']).unsqueeze(0).to(device)
        
        mean, _ = ensemble.predict_segment(states, actions)
        
        predicted.append(mean.item())
        true.append(traj['cumulative_reward'])
    
    correlation, _ = spearmanr(predicted, true)
    
    return correlation

def validate_graph_edges(graph, pref_buffer):
    """Check how many graph edges are correct"""
    edges = graph.get_training_pairs()
    
    correct = 0
    total = len(edges)
    
    for winner_id, loser_id in edges:
        winner_traj = pref_buffer.get_trajectory(winner_id)
        loser_traj = pref_buffer.get_trajectory(loser_id)
        
        if winner_traj['cumulative_reward'] > loser_traj['cumulative_reward']:
            correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    return accuracy, correct, total

def analyze_checkpoint(checkpoint_path, config_path):
    """Main analysis function"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    ensemble, sac, env, checkpoint = load_checkpoint(checkpoint_path, config)
    device = ensemble.device
    
    # Reconstruct preference buffer (from checkpoint if saved, else warn)
    if 'graph_edges' not in checkpoint:
        print("  Warning: Checkpoint doesn't contain preference graph")
        print("    Re-run training with updated code to save graph state")
        return
    
    # Reconstruct graph
    graph = PreferenceGraph()
    for winner, loser in checkpoint['graph_edges']:
        graph.add_preference(winner, loser, is_human_label=False)
    
    print("\n" + "="*70)
    print("CHECKPOINT ANALYSIS REPORT")
    print("="*70)
    
    # 1. Graph Statistics
    stats = graph.get_stats()
    print(f"\nPreference Graph:")
    print(f"   Total edges: {stats['total']}")
    print(f"   Human queries: {stats['direct']}")
    print(f"   Auto-labels: {stats['auto']}")
    print(f"   Transitive: {stats['transitive']}")
    print(f"   Augmentation ratio: {stats['ratio']:.2f}x")
    
    # 2. Edge Accuracy (if we have buffer)
    # Note: We'd need to save the replay buffer in checkpoint too
    # For now, just report what we can from graph structure
    
    # 3. Policy Performance
    print(f"\n Policy Evaluation:")
    
    # Random baseline
    random_scores = []
    for _ in range(10):
        s, _ = env.reset()
        score = 0
        done = False
        while not done:
            a = env.env.action_space.sample()
            ns, r, term, trunc, _ = env.step(a)
            score += r
            done = term or trunc
        random_scores.append(score)
    
    random_mean = np.mean(random_scores)
    random_std = np.std(random_scores)
    
    # Trained agent
    agent_mean, agent_std = evaluate_policy(env, sac, n_episodes=20)
    
    improvement = agent_mean / random_mean if random_mean > 0 else 0
    
    print(f"   Random policy: {random_mean:.1f} ± {random_std:.1f}")
    print(f"   Trained agent: {agent_mean:.1f} ± {agent_std:.1f}")
    print(f"   Improvement: {improvement:.1f}x")
    
    if improvement > 2.0:
        print("    SUCCESS: Agent significantly better than random!")
    elif improvement > 1.2:
        print("     PARTIAL: Agent somewhat better than random")
    else:
        print("    FAILURE: Agent not better than random")
    
    # 4. Reward Model Quality (if we had buffer)
    # Would need: correlation, calibration, etc.
    
    # 5. Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    success_criteria = {
        'Augmentation > 3x': stats['ratio'] > 3.0,
        'Agent > 2x random': improvement > 2.0,
        'Total queries < 200': stats['direct'] < 200,
    }
    
    for criterion, passed in success_criteria.items():
        status = "✓" if passed else "❌"
        print(f"   {status} {criterion}")
    
    all_passed = all(success_criteria.values())
    
    if all_passed:
        print("\n All success criteria met! Model is ready for deployment.")
    else:
        print("\n  Some criteria not met. Consider:")
        if not success_criteria['Augmentation > 3x']:
            print("   - Check transitive closure implementation")
        if not success_criteria['Agent > 2x random']:
            print("   - Increase SAC training steps")
            print("   - Check reward model correlation")
        if not success_criteria['Total queries < 200']:
            print("   - Use more aggressive UCB/LCB filtering (increase β)")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f" Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f" Config not found: {args.config}")
        sys.exit(1)
        
    analyze_checkpoint(args.checkpoint, args.config)