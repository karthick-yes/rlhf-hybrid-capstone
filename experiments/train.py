import sys
import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import networkx as nx

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.reward_ensemble import RewardEnsemble
from src.graph.preference_graph import PreferenceGraph
from src.acquisition.ucb_lcb import UCBLCBFilter
from src.acquisition.sampling import ActiveDethroning
from src.agents.entropy_agents import SimpleEntropySAC
from src.agents.sac import SAC
from src.envs.wrappers import make_env
from src.utils.replay_buffer import ReplayBuffer
from src.utils.buffer_adapter import SACBufferAdapter
from src.utils.logger import ExperimentLogger
from src.envs.oracle import Oracle


class TeeOutput:
    """Captures terminal output to a file while still displaying it."""
    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode, encoding='utf-8', buffering=1)  # Line buffering
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()


def load_config(config_path):
    """Load configuration from YAML file or return defaults"""
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        numeric_keys = ['lr', 'beta', 'weight_decay', 'gamma', 'tau', 'alpha',
                        'min_warmup_reward', 'max_defender_uncertainty', 
                        'exploration_epsilon', 'std_threshold']
        
        for key in numeric_keys:
            if key in config and isinstance(config[key], str):
                try:
                    config[key] = float(config[key])
                    print(f"Converted {key} from string to float: {config[key]}")
                except ValueError:
                    print(f"Warning: Could not convert {key}={config[key]} to float")
        
        return config
    else:
        print("Using default configuration")
        return {
            'env_name': 'CartPole-v1',
            'state_dim': 4,
            'action_dim': 1,
            'beta': 7.0,
            'n_trajectories': 50,
            'n_queries_per_round': 20,
            'n_rounds': 5,
            'n_bootstrap': 10,
            'pool_size': 20,
            'update_freq': 10,
            'ensemble_size': 3,
            'hidden_dim': 256,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'sac_steps_per_round': 5000,
            'segment_length': 50,
            'max_episode_steps': 500,
            'buffer_capacity': 10000,
            'seed': 42,
            'min_warmup_reward': 20.0,
            'max_defender_uncertainty': 1.5,
            'exploration_epsilon': 0.1,
            'defender_reset_freq': 25,
            'entropy_full_buffer': False,
            'sac_collection_episodes': 10,
            'entropy_collection_episodes': 10
        }


class HybridDualAgentTrainer:
    """
    Complete Dual-Agent Hybrid RLHF System with Enhanced Diagnostics.
    
    Architecture: Iterative loop alternating between exploration (Entropy) and exploitation (SAC)
    Features: All safety checks, diagnostics, and verbose logging from Document 1
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print(f"DUAL-AGENT HYBRID PREFERENCE LEARNING SYSTEM")
        print(f"{'='*70}")
        print(f"   Device: {self.device}")
        print(f"   Environment: {config.get('env_name', 'CartPole-v1')}")
        print(f"   Architecture: Iterative Dual-Agent Loop")
        print(f"   Beta (UCB/LCB): {config.get('beta', 7.0)}")
        print(f"   Ensemble Size: {config.get('ensemble_size', 3)}")
        print(f"   Training Rounds: {config.get('n_rounds', 5)}")
        print(f"{'='*70}\n")
        
        print(f"\n[CONFIG VERIFICATION]")
        print(f"   Learning Rate (lr): {config.get('lr')}")
        print(f"   Beta: {config.get('beta')}")
        print(f"   Std Threshold: {config.get('std_threshold')}")
        print(f"   Hidden Dim: {config.get('hidden_dim')}")
        print(f"   Entropy Full Buffer: {config.get('entropy_full_buffer')}")
        print("-" * 30 + "\n")
        
        # --- 1. Environment Setup ---
        self.env_name = config.get('env_name', 'CartPole-v1')
        self.segment_length = config.get('segment_length', 50)  # For queries
        self.max_episode_steps = config.get('max_episode_steps', 500)  # For collection
        
        self.env = make_env(
            self.env_name,
            segment_length=self.segment_length,
            seed=config.get('seed', 42)
        )
        
        print(f"[ENV CONFIG]")
        print(f"   Query Length (segment_length): {self.segment_length}")
        print(f"   Collection Length (max_episode_steps): {self.max_episode_steps}")
        print(f"   State Dim: {self.env.state_dim}")
        print(f"   Action Dim: {self.env.action_dim}")
        print(f"   Is Discrete: {self.env.is_discrete}\n")
        
        # --- 2. Core Components ---
        self.ensemble = RewardEnsemble(
            state_dim=self.env.state_dim,
            action_dim=self.env.agent_action_dim,
            config=config,
            device=self.device
        )
        self.graph = PreferenceGraph()
        self.ucb_lcb = UCBLCBFilter(
            beta=config.get('beta', 7.0),
            adaptive_beta=True,
            std_threshold= config.get('std_threshold', 50.0)
        )
        self.dethroning = ActiveDethroning()
        self.logger = ExperimentLogger()
        self.oracle = Oracle(self.env_name)

        # --- 3. Data Storage ---
        self.pref_buffer = ReplayBuffer(capacity=config.get('buffer_capacity', 10000))
        self.sac_buffer = SACBufferAdapter(
            self.env.state_dim, 
            self.env.action_dim, 
            capacity=config.get('buffer_capacity', 100000),
            device=self.device
        )
        
        self.traj_sources = {}  # {traj_id: "SAC" or "Entropy"}
        
        # --- 5. Dual Agents ---
        self.entropy_agent = SimpleEntropySAC(
            self.env.state_dim,
            self.env.action_dim,
            self.env.action_range,
            k=5,
            device=self.device
        )
        self.sac_agent = SAC(
            self.env.state_dim,
            self.env.action_dim,
            device=self.device,
            lr=config.get('lr', 3e-4)
        )
        
        # --- 6. State Tracking ---
        self.defender_id = None
        self.total_human_queries = 0
        self.queried_pairs = set()
        
        # --- 7. Enhanced Diagnostics (From Doc 1) ---
        self.query_log = []
        self.reward_correlations = []
        self.sac_performance = []
        self.loop_stats = []  # Dual-agent specific stats per round
        self.auto_label_accuracy_history = []
        
        # Create directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("videos", exist_ok=True)
        os.makedirs("diagnostics", exist_ok=True)
        
        print("[INITIALIZATION COMPLETE]\n")

    def save_checkpoint(self, tag="latest"):
        """Save training state to disk (Doc 1 feature)"""
        path = os.path.join("checkpoints", f"checkpoint_{tag}.pt")
        
        ensemble_state = [m.state_dict() for m in self.ensemble.models]
        optimizer_state = [opt.state_dict() for opt in self.ensemble.optimizers]
        
        checkpoint = {
            'ensemble_state': ensemble_state,
            'optimizer_state': optimizer_state,
            'sac_actor': self.sac_agent.actor.state_dict(),
            'sac_critic': self.sac_agent.critic.state_dict(),
            'entropy_actor': self.entropy_agent.agent.actor.state_dict() if hasattr(self.entropy_agent.agent, 'actor') else None,
            'total_human_queries': self.total_human_queries,
            'defender_id': self.defender_id,
            'query_log': self.query_log,
            'reward_correlations': self.reward_correlations,
            'sac_performance': self.sac_performance,
            'loop_stats': self.loop_stats,
            'traj_sources': self.traj_sources,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"   Checkpoint saved: {path}")

    # ==================== COLLECTION HELPER (Doc 2 Architecture) ====================
    
    def _collect_and_store(self, agent, n_episodes, label="Unknown"):
        """
        Collects full episodes using max_episode_steps, stores in buffer, 
        and tags the source (SAC or Entropy).
        
        Returns: all_states (concatenated for knowledge sharing)
        """
        new_ids = []
        all_states = []
        rewards = []
        
        iterator = range(n_episodes)
        if n_episodes > 2: 
            iterator = tqdm(iterator, desc=f"   Collecting ({label})", leave=False, unit="ep")

        for _ in iterator:
            # Use max_episode_steps for full task execution
            if label == "Entropy":
                states, actions, int_r, ext_r, _ = agent.collect_trajectory(
                    self.env.env, 
                    max_steps=self.max_episode_steps
                )
                true_reward = np.sum(ext_r)
            else:
                # SAC Agent
                states, actions, true_reward = self.env.collect_segment(
                    policy=lambda s: agent.select_action(s, deterministic=False),
                    max_steps=self.max_episode_steps
                )
            
            # Store full trajectory
            tid = self.pref_buffer.add_trajectory(states, actions, float(true_reward))
            
            # Tag Source (Doc 2 feature)
            self.traj_sources[tid] = label
            
            new_ids.append(tid)
            rewards.append(true_reward)
            all_states.append(states)
        
        # Stats (Doc 1 verbosity)
        if rewards:
            print(f"   {label} Collection: {len(rewards)} episodes, "
                  f"Avg Reward: {np.mean(rewards):.1f}, Max: {np.max(rewards):.1f}, "
                  f"Min: {np.min(rewards):.1f}, Std: {np.std(rewards):.1f}")
        
        if len(all_states) > 0:
            return np.concatenate(all_states, axis=0)
        return np.array([])

    def _compute_true_reward(self, rewards):
        """Helper for reward computation"""
        return float(np.sum(rewards))

    # ==================== PHASE 1: WARMUP (Doc 1 Safety Checks) ====================
    
    def phase1_warmup(self, n_episodes=None, retries=0):
        """
        Phase 1: Unsupervised entropy-driven exploration.
        Includes retry logic to prevent infinite loops on tasks where entropy != reward.
        """
        if n_episodes is None:
            n_episodes = self.config.get('n_trajectories', 50)
            
        print(f"\n{'='*70}")
        print(f"[PHASE 1] UNSUPERVISED WARMUP - Attempt {retries+1}")
        print(f"{'='*70}")
        print(f"   Target Episodes: {n_episodes}")
        print(f"   Using: Entropy Agent (Max-Entropy Exploration)")
        print()
        
        # Collect using entropy agent
        _ = self._collect_and_store(self.entropy_agent, n_episodes, label="Entropy")
        
        # Quality Check (Doc 1 Safety Feature)
        all_ids = self.pref_buffer.get_all_ids()
        if all_ids:
            rewards = [self.pref_buffer.get_trajectory(tid)['cumulative_reward'] for tid in all_ids]
            reward_mean = np.mean(rewards)
            reward_max = np.max(rewards)
            reward_std = np.std(rewards)
            
            print(f"\n   {'─'*66}")
            print(f"   QUALITY CHECK")
            print(f"   {'─'*66}")
            print(f"   Mean Reward: {reward_mean:.2f}")
            print(f"   Max Reward:  {reward_max:.2f}")
            print(f"   Std Reward:  {reward_std:.2f}")
            print(f"   {'─'*66}")
            
            # Threshold from config
            min_quality = self.config.get('min_warmup_reward', 20.0)
            
            if reward_max < min_quality:
                # RETRY LOGIC (Doc 1 Safety)
                if retries < 2:  # Allow 2 retries (3 attempts total)
                    print(f"\n     LOW REWARD DIVERSITY (<{min_quality})")
                    print(f"   Entropy agent is struggling (expected for some environments)")
                    print(f"   Retrying with {n_episodes + 20} episodes...\n")
                    return self.phase1_warmup(n_episodes=n_episodes + 20, retries=retries+1)
                else:
                    # FORCE PROCEED
                    print(f"\n     WARNING: Failed to meet reward threshold {min_quality} after 3 attempts")
                    print(f"   Proceeding with best available data (Max: {reward_max:.2f})")
                    print(f"   Active Learning phase will attempt to learn from these episodes\n")
            else:
                print(f"   ✓ Quality threshold met ({reward_max:.2f} >= {min_quality})")
        
        print(f"\n   Phase 1 Complete: {len(all_ids)} trajectories collected")
        print(f"   All trajectories tagged as: Entropy")
        print(f"{'='*70}\n")
        
        self.save_checkpoint("phase1_done")

    # ==================== BOOTSTRAP (Doc 1 Validation) ====================
    
    def bootstrap_ensemble(self, n_bootstrap=None):
        """Initialize ensemble with random human queries"""
        if n_bootstrap is None:
            n_bootstrap = self.config.get('n_bootstrap', 10)
            
        print(f"\n{'='*70}")
        print(f"[BOOTSTRAP] ENSEMBLE INITIALIZATION")
        print(f"{'='*70}")
        print(f"   Random Queries: {n_bootstrap}")
        print()
        
        all_ids = self.pref_buffer.get_all_ids()
        if len(all_ids) < 2:
            print("    Error: Not enough trajectories for bootstrap!")
            return
        
        for i in range(n_bootstrap):
            id1, id2 = np.random.choice(all_ids, 2, replace=False)
            r1 = self.pref_buffer.get_trajectory(id1)['cumulative_reward']
            r2 = self.pref_buffer.get_trajectory(id2)['cumulative_reward']
            label = self.oracle.compare(r1, r2)
            
            src1 = self.traj_sources.get(id1, 'Unknown')
            src2 = self.traj_sources.get(id2, 'Unknown')
            
            if label == 1:
                self.graph.add_preference(id1, id2, is_human_label=True)
                print(f"   Bootstrap {i+1:2d}: {id1}({src1}) > {id2}({src2}) | Rewards: {r1:.1f} > {r2:.1f}")
            else:
                self.graph.add_preference(id2, id1, is_human_label=True)
                print(f"   Bootstrap {i+1:2d}: {id2}({src2}) > {id1}({src1}) | Rewards: {r2:.1f} > {r1:.1f}")
            
            self.total_human_queries += 1
            self.queried_pairs.add((min(id1, id2), max(id1, id2)))
        
        # VALIDATION (Doc 1 Safety Feature)
        true_rewards = {tid: self.pref_buffer.get_trajectory(tid)['cumulative_reward'] 
                       for tid in all_ids}
        best_id = max(true_rewards, key=true_rewards.get)
        true_best_reward = true_rewards[best_id]
        
        # Print reward distribution for diagnostics
        rewards_list = list(true_rewards.values())
        print(f"\n   {'─'*66}")
        print(f"   REWARD DISTRIBUTION")
        print(f"   {'─'*66}")
        print(f"   Min:  {min(rewards_list):.1f}")
        print(f"   Max:  {max(rewards_list):.1f}")
        print(f"   Mean: {np.mean(rewards_list):.1f}")
        print(f"   Std:  {np.std(rewards_list):.1f}")
        print(f"   {'─'*66}")
        
        # WARNING if no high-quality trajectories (Doc 1 Safety)
        if max(rewards_list) < 100:
            print(f"\n     WARNING: Max reward only {max(rewards_list):.1f} < 100")
            print(f"   Agent may underperform due to lack of good examples")
            print(f"   Consider extending warmup phase\n")
        
        print(f"\n   Training ensemble on {n_bootstrap} bootstrap labels...")
        self.retrain_ensemble_and_relabel(verbose=False)
        
        # Initialize defender
        self.update_defender()
        
        # Check defender uncertainty (Doc 1 Safety)
        _, def_std = self.compute_ensemble_stats(self.defender_id)
        max_defender_uncertainty = self.config.get('std_threshold', 50.0)
        if def_std > max_defender_uncertainty:
            print(f"\n     Defender uncertainty high (σ={def_std:.2f} > {max_defender_uncertainty})")
            print(f"   Temporarily widening beta to force more human queries...")
            self.original_beta = self.config['beta']
            self.config['beta'] = 2.0
        
        stats = self.graph.get_stats()
        print(f"\n   Bootstrap Complete:")
        print(f"   - Total Edges: {stats['total']}")
        print(f"   - Augmentation: {stats['ratio']:.2f}x")
        print(f"   - Human Queries: {self.total_human_queries}")
        print(f"{'='*70}\n")
        
        # Track initial correlation (Doc 1 Diagnostic)
        corr = self.compute_reward_correlation()
        self.reward_correlations.append({
            'round': 0,
            'iteration': 0, 
            'correlation': corr
        })
        print(f"   Initial Reward Correlation: {corr:.3f}\n")
        
        self.save_checkpoint("bootstrap_done")

    def update_defender(self):
        """
        Select best trajectory based on current Reward Model.
        Uses heuristic sampling for efficiency (Doc 2 approach).
        """
        all_ids = self.pref_buffer.get_all_ids()
        if not all_ids:
            print("     Warning: No trajectories available for defender update")
            return
            
        best_id, best_val = None, -float('inf')
        
        # Heuristic: Check recent 100 + random 50 to balance recency and diversity
        candidates = list(all_ids[-100:])
        if len(all_ids) > 100:
            remaining = [tid for tid in all_ids[:-100]]
            if len(remaining) > 50:
                candidates += list(np.random.choice(remaining, 50, replace=False))
            else:
                candidates += remaining
        
        for tid in candidates:
            mu, _ = self.compute_ensemble_stats(tid)
            if mu > best_val:
                best_val = mu
                best_id = tid
        
        self.defender_id = best_id
        
        # Log Source of Defender (Doc 2 Feature + Doc 1 Verbosity)
        source = self.traj_sources.get(self.defender_id, 'Unknown')
        true_reward = self.pref_buffer.get_trajectory(self.defender_id)['cumulative_reward']
        print(f"     Defender Updated: {self.defender_id} ({source})")
        print(f"      - Predicted Reward: {best_val:.1f}")
        print(f"      - True Reward:      {true_reward:.1f}")

    # ==================== COMPUTE STATS (Doc 1 Diagnostics) ====================
    
    def compute_ensemble_stats(self, traj_id):
        """Compute ensemble prediction mean and std for a trajectory"""
        traj = self.pref_buffer.get_trajectory(traj_id)
        states = torch.FloatTensor(traj['states']).unsqueeze(0).to(self.device)
        actions = torch.FloatTensor(traj['actions']).unsqueeze(0).to(self.device)
        mean, std = self.ensemble.predict_segment(states, actions)
        return mean.item(), std.item()
    
    def compute_reward_correlation(self):
        """
        Compute Spearman correlation between predicted and ground truth rewards.
        This tells us how well the ensemble has learned to rank trajectories.
        (Doc 1 Diagnostic Feature)
        """
        all_ids = self.pref_buffer.get_all_ids()
        if len(all_ids) < 5:
            return 0.0
        
        predicted_rewards = []
        true_rewards = []
        
        for tid in all_ids:
            traj = self.pref_buffer.get_trajectory(tid)
            states = torch.FloatTensor(traj['states']).unsqueeze(0).to(self.device)
            actions = torch.FloatTensor(traj['actions']).unsqueeze(0).to(self.device)
            mean, _ = self.ensemble.predict_segment(states, actions)
            
            predicted_rewards.append(mean.item())
            true_rewards.append(traj['cumulative_reward'])
        
        correlation, _ = spearmanr(predicted_rewards, true_rewards)
        return correlation if not np.isnan(correlation) else 0.0

    # ==================== ACTIVE LEARNING (Full Diagnostics + Dual-Agent Tracking) ====================
    
    def phase2_active_learning(self, n_queries=None, pool_size=None, round_num=1):
        """
        Active Preference Learning with:
        """
        if n_queries is None:
            n_queries = self.config.get('n_queries_per_round', 20)
        if pool_size is None:
            pool_size = self.config.get('pool_size', 20)
            
        print(f"\n{'='*70}")
        print(f"[ACTIVE LEARNING] Round {round_num}")
        print(f"{'='*70}")
        print(f"   Queries: {n_queries} | Pool Size: {pool_size}")
        print()
        
        all_ids = self.pref_buffer.get_all_ids()
        if not all_ids:
            print("    Error: Buffer empty!")
            return
        
        # Get ground truth for validation (Doc 1)
        true_rewards = {tid: self.pref_buffer.get_trajectory(tid)['cumulative_reward'] 
                       for tid in all_ids}
        true_best_id = max(true_rewards, key=true_rewards.get)
        
        # Track defender changes
        defender_change_count = 0
        
        # Dual-agent diagnostics (Doc 2 Feature)
        dual_agent_stats = {
            'SAC_Wins': 0, 
            'Entropy_Wins': 0, 
            'SAC_Auto': 0, 
            'Entropy_Auto': 0,
            'SAC_Query': 0,
            'Entropy_Query': 0
        }
        
        pbar = tqdm(range(n_queries), desc="   Querying", unit="query")
        
        for iteration in pbar:
            # PERIODIC DEFENDER VALIDATION (Doc 1 Safety)
            reset_freq = self.config.get('defender_reset_freq', 25)
            if reset_freq > 0 and (iteration + 1) % reset_freq == 0:
                if len(self.graph.G) > 3:
                    rev_G = self.graph.G.reverse()
                    try:
                        scores = nx.pagerank(rev_G)
                        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        best_of_top3 = max(top_3, key=lambda x: true_rewards.get(x[0], -999))[0]
                        
                        if best_of_top3 != self.defender_id:
                            old_reward = true_rewards.get(self.defender_id, 0)
                            new_reward = true_rewards.get(best_of_top3, 0)
                            old_source = self.traj_sources.get(self.defender_id, 'Unknown')
                            new_source = self.traj_sources.get(best_of_top3, 'Unknown')
                            print(f"\n    DEFENDER RESET (PageRank)")
                            print(f"      {self.defender_id}({old_source}, r={old_reward:.1f}) → "
                                  f"{best_of_top3}({new_source}, r={new_reward:.1f})")
                            self.defender_id = best_of_top3
                            defender_change_count += 1
                    except:
                        pass
            
            # Sample Candidates
            available = []
            for tid in all_ids:
                if tid == self.defender_id: 
                    continue
                pair = (min(self.defender_id, tid), max(self.defender_id, tid))
                if pair not in self.queried_pairs: 
                    available.append(tid)
            
            if len(available) == 0:
                print("\n     No new candidates available!")
                break
            
            pool = np.random.choice(available, size=min(pool_size, len(available)), replace=False)
            
            # Compute Stats
            def_mu, def_std = self.compute_ensemble_stats(self.defender_id)
            cand_mus = np.array([self.compute_ensemble_stats(cid)[0] for cid in pool])
            cand_stds = np.array([self.compute_ensemble_stats(cid)[1] for cid in pool])
            
            # UCB/LCB Filter (with adaptive beta)
            auto_labels, uncertain_idx, dethroned, new_def = self.ucb_lcb.filter_candidates(
                def_mu, def_std, cand_mus, cand_stds, pool.tolist()
            )
            
            # --- TRACK AUTO-LABELS BY SOURCE (Doc 2 Feature) ---
            for winner, loser in auto_labels:
                w_id = self.defender_id if winner == 'defender' else winner
                l_id = self.defender_id if loser == 'defender' else loser
                
                loser_source = self.traj_sources.get(l_id, 'Unknown')
                if loser_source == 'SAC':
                    dual_agent_stats['SAC_Auto'] += 1
                elif loser_source == 'Entropy':
                    dual_agent_stats['Entropy_Auto'] += 1
                
                self.graph.add_preference(w_id, l_id, is_human_label=False, is_auto_label=True)
            
            # Diagnostics: Auto-rate and exploration epsilon (Doc 1)
            auto_rate = len(auto_labels) / len(pool) if len(pool) > 0 else 0
            epsilon = self.config.get('exploration_epsilon', 0.1)
            
            # Exploration epsilon override (Doc 1 Safety)
            if auto_rate > 0.95:
                pass  # Allow high auto-rate
            elif np.random.random() < epsilon:
                uncertain_idx = list(range(len(pool)))
                auto_labels = []
                dethroned = False
            
            # Handle Dethronement or Human Query
            defender_changed = False
            if dethroned:
                old_reward = true_rewards.get(self.defender_id, 0)
                new_reward = true_rewards.get(new_def, 0)
                old_source = self.traj_sources.get(self.defender_id, 'Unknown')
                new_source = self.traj_sources.get(new_def, 'Unknown')
                
                print(f"\n    NATURAL DETHRONE")
                print(f"      {self.defender_id}({old_source}, r={old_reward:.1f}) → "
                      f"{new_def}({new_source}, r={new_reward:.1f})")
                
                self.defender_id = new_def
                defender_changed = True
                defender_change_count += 1
                
                if new_source == 'SAC':
                    dual_agent_stats['SAC_Wins'] += 1
                elif new_source == 'Entropy':
                    dual_agent_stats['Entropy_Wins'] += 1
                
            elif len(uncertain_idx) > 0:
                uncertain_ids = pool[uncertain_idx]
                u_mus = cand_mus[uncertain_idx]
                u_stds = cand_stds[uncertain_idx]
                
                best_idx, score, _, _ = self.dethroning.select_best_challenger(
                    def_mu, def_std, u_mus, u_stds, uncertain_ids.tolist()
                )
                challenger_id = best_idx
                
                def_r = self.pref_buffer.get_trajectory(self.defender_id)['cumulative_reward']
                chal_r = self.pref_buffer.get_trajectory(challenger_id)['cumulative_reward']
                
                label = self.oracle.compare(chal_r, def_r)
                self.total_human_queries += 1
                self.queried_pairs.add((min(self.defender_id, challenger_id), 
                                       max(self.defender_id, challenger_id)))
                
                # Track which agent was queried (Doc 2)
                challenger_source = self.traj_sources.get(challenger_id, 'Unknown')
                if challenger_source == 'SAC':
                    dual_agent_stats['SAC_Query'] += 1
                elif challenger_source == 'Entropy':
                    dual_agent_stats['Entropy_Query'] += 1
                
                if label == 1:
                    self.graph.add_preference(challenger_id, self.defender_id, is_human_label=True)
                    
                    def_source = self.traj_sources.get(self.defender_id, 'Unknown')
                    chal_source = self.traj_sources.get(challenger_id, 'Unknown')
                    
                    print(f"\n    HUMAN QUERY DETHRONE")
                    print(f"      {challenger_id}({chal_source}, r={chal_r:.1f}) > "
                          f"Defender {self.defender_id}({def_source}, r={def_r:.1f})")
                    
                    self.defender_id = challenger_id
                    defender_changed = True
                    defender_change_count += 1
                    
                    if chal_source == 'SAC':
                        dual_agent_stats['SAC_Wins'] += 1
                    elif chal_source == 'Entropy':
                        dual_agent_stats['Entropy_Wins'] += 1
                else:
                    self.graph.add_preference(self.defender_id, challenger_id, is_human_label=True)
            
            # Log metrics (Doc 1 Diagnostic)
            stats = self.graph.get_stats()
            self.query_log.append({
                'round': round_num,
                'iteration': iteration + 1,
                'total_human_queries': self.total_human_queries,
                'auto_labels': len(auto_labels),
                'auto_rate': auto_rate*100,
                'graph_edges': stats['total'],
                'augmentation': stats['ratio'],
                'defender_changed': defender_changed,
                'ensemble_std_mean': cand_stds.mean(),
                'defender_id': self.defender_id,
                'defender_source': self.traj_sources.get(self.defender_id, 'Unknown'),
                'defender_true_reward': true_rewards.get(self.defender_id, 0)
            })
            
            # Update Progress Bar (Doc 1 + Doc 2)
            is_optimal = (self.defender_id == true_best_id)
            opt_marker = "✓" if is_optimal else "✗"
            def_source = self.traj_sources.get(self.defender_id, 'Unknown')
            pbar.set_postfix({
                'Aug': f"{stats['ratio']:.1f}x", 
                'Auto': f"{auto_rate*100:.0f}%",
                'Def': f"{self.defender_id}({def_source[0]}){opt_marker}"
            })
            
            # Restore beta (Doc 1 Safety)
            if hasattr(self, 'original_beta'):
                self.config['beta'] = self.original_beta
                delattr(self, 'original_beta')
            
            # Periodic Updates
            update_freq = self.config.get('update_freq', 10)
            if (iteration + 1) % update_freq == 0:
                self.retrain_ensemble_and_relabel(verbose=False)
                
                # Track reward correlation (Doc 1 Diagnostic)
                corr = self.compute_reward_correlation()
                self.reward_correlations.append({
                    'round': round_num,
                    'iteration': self.total_human_queries,
                    'correlation': corr
                })
                
                self.save_checkpoint(f"round{round_num}_iter{iteration+1}")
        
        # FINAL VALIDATION (Doc 1 Safety)
        print(f"\n   {'─'*66}")
        print(f"   ROUND {round_num} SUMMARY")
        print(f"   {'─'*66}")
        print(f"   Defender Changes: {defender_change_count}")
        print(f"   Final Defender: {self.defender_id} ({self.traj_sources.get(self.defender_id, 'Unknown')})")
        print(f"   - Predicted: {def_mu:.1f}")
        print(f"   - True:      {true_rewards.get(self.defender_id, 0):.1f}")
        print(f"   True Best: {true_best_id} (r={true_rewards[true_best_id]:.1f})")
        
        if self.defender_id == true_best_id:
            print(f"    SUCCESS: Found the true best trajectory!")
        else:
            gap = true_rewards[true_best_id] - true_rewards.get(self.defender_id, 0)
            print(f"   ️  Defender is suboptimal (gap: {gap:.1f})")
        
        if defender_change_count == 0:
            print(f"     CRITICAL: Defender NEVER changed - likely stuck!")
        elif defender_change_count < 3:
            print(f"   ️  Only {defender_change_count} changes - may be under-exploring")
        else:
            print(f"    Good exploration: {defender_change_count} defender changes")
        
        # Dual-Agent Stats (Doc 2 Feature)
        print(f"\n   DUAL-AGENT BREAKDOWN:")
        print(f"   - SAC Wins:        {dual_agent_stats['SAC_Wins']}")
        print(f"   - Entropy Wins:    {dual_agent_stats['Entropy_Wins']}")
        print(f"   - SAC Auto-Reject: {dual_agent_stats['SAC_Auto']}")
        print(f"   - Entropy Auto-Reject: {dual_agent_stats['Entropy_Auto']}")
        print(f"   - SAC Queries:     {dual_agent_stats['SAC_Query']}")
        print(f"   - Entropy Queries: {dual_agent_stats['Entropy_Query']}")
        print(f"   {'─'*66}\n")
        
        # Store round stats (Doc 2)
        self.loop_stats.append({
            'round': round_num,
            'defender_id': self.defender_id,
            'defender_source': self.traj_sources.get(self.defender_id, 'Unknown'),
            'defender_true_reward': true_rewards.get(self.defender_id, 0),
            'defender_changes': defender_change_count,
            **dual_agent_stats
        })
        
        # Use custom logger (Doc 2)
        self.logger.log_dual_agent_stats(
            round_num,
            self.defender_id,
            self.traj_sources.get(self.defender_id, 'Unknown'),
            {'SAC': dual_agent_stats['SAC_Wins'], 'Entropy': dual_agent_stats['Entropy_Wins']},
            {'SAC_Auto': dual_agent_stats['SAC_Auto'], 'Entropy_Auto': dual_agent_stats['Entropy_Auto']},
            {'SAC_Query': dual_agent_stats['SAC_Query'], 'Entropy_Query': dual_agent_stats['Entropy_Query']}
        )

    def retrain_ensemble_and_relabel(self, verbose=True):
        """Retrain ensemble on current graph (Doc 1 implementation)"""
        pairs = self.graph.get_training_pairs()
        
        if len(pairs) < 5:
            if verbose: 
                print("     Not enough data to train yet.")
            return

        epochs = 30
        samples_per_epoch = 32
        patience = 3
        best_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            print(f"\n   {'='*60}")
            print(f"   RETRAINING ENSEMBLE")
            print(f"   {'='*60}")
            iterator = tqdm(range(epochs), desc="   Training", unit="epoch")
        else:
            iterator = range(epochs)
        
        for epoch in iterator:
            indices = np.random.randint(0, len(pairs), size=samples_per_epoch)
            epoch_loss = 0
            
            for i in indices:
                wid, lid = pairs[i]
                wt = self.pref_buffer.get_trajectory(wid)
                lt = self.pref_buffer.get_trajectory(lid)
                
                s0 = torch.FloatTensor(wt['states']).unsqueeze(0).to(self.device)
                a0 = torch.FloatTensor(wt['actions']).unsqueeze(0).to(self.device)
                s1 = torch.FloatTensor(lt['states']).unsqueeze(0).to(self.device)
                a1 = torch.FloatTensor(lt['actions']).unsqueeze(0).to(self.device)
                
                label_t = torch.tensor([0.0]).to(self.device)
                loss = self.ensemble.train_step(s0, a0, s1, a1, label_t)
                epoch_loss += loss
            
            avg_loss = epoch_loss / samples_per_epoch
            
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\n   Early stopping at epoch {epoch+1}/{epochs} (loss={avg_loss:.4f})")
                break
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(loss=f"{avg_loss:.4f}", patience=f"{patience_counter}/{patience}")
        
        if verbose: 
            print("    Ensemble Trained.")
        
        # Relabel SAC buffer (Doc 1)
        self.sac_buffer.relabel_and_flatten(self.pref_buffer, self.ensemble)
        
        if verbose: 
            print(f"   SAC Buffer Relabeled (Size: {self.sac_buffer.size})")

    # ==================== PHASE 3: POLICY LEARNING (Doc 1 + Doc 2) ====================

    def phase3_policy_learning(self, n_steps=None, round_num=1):
        """Train SAC policy on learned rewards (Doc 1 diagnostics + Doc 2 architecture)"""
        if n_steps is None:
            n_steps = self.config.get('sac_steps_per_round', 5000)
            
        print(f"\n{'='*70}")
        print(f"[PHASE 3] POLICY LEARNING - Round {round_num}")
        print(f"{'='*70}")
        print(f"   Training Steps: {n_steps}")
        print(f"   SAC Buffer Size: {self.sac_buffer.size}")
        print()
        
        if self.sac_buffer.size < 100:
            print("   ️  Error: SAC Buffer too small (<100 samples)")
            return
        
        #if round_num > 1: 
        #    print("   [SAC] Detect non-stationary reward scale. Resetting Critic...")
        #     self.sac_agent.reset_critic()
             
        # Evaluate performance periodically during training (Doc 1)
        eval_frequency = max(1000, n_steps // 10)
        
        pbar = tqdm(range(n_steps), desc="   Training SAC", unit="step")
        
        for i in pbar:
            if self.sac_buffer.size < 256: 
                print(f"\n     Buffer exhausted at step {i}")
                break
                
            actor_loss, critic_loss = self.sac_agent.update(self.sac_buffer, batch_size=256)
            
            # Periodic evaluation (Doc 1 Diagnostic)
            if (i+1) % eval_frequency == 0:
                avg_reward = self.quick_evaluation(n_episodes=5)
                self.sac_performance.append({
                    'round': round_num,
                    'step': i+1,
                    'avg_reward': avg_reward
                })
                pbar.set_postfix(
                    act_loss=f"{actor_loss:.3f}",
                    crit_loss=f"{critic_loss:.3f}",
                    reward=f"{avg_reward:.1f}"
                )
            elif (i+1) % 100 == 0:
                pbar.set_postfix(
                    act_loss=f"{actor_loss:.3f}", 
                    crit_loss=f"{critic_loss:.3f}"
                )
        
        # Final evaluation for this round
        final_reward = self.quick_evaluation(n_episodes=10)
        print(f"\n   Round {round_num} Final Performance: {final_reward:.1f}")
        print(f"{'='*70}\n")
        
        self.save_checkpoint(f"round{round_num}_sac_done")

    # ==================== EVALUATION & DIAGNOSTICS (Doc 1) ====================
    
    def quick_evaluation(self, n_episodes=5):
        """Quick evaluation of current policy (no rendering)"""
        scores = []
        for _ in range(n_episodes):
            s, _ = self.env.reset()
            score = 0
            done = False
            steps = 0
            max_steps = 1000
            
            while not done and steps < max_steps:
                a = self.sac_agent.select_action(s, deterministic=True)
                ns, r, term, trunc, _ = self.env.step(a)
                score += r
                s = ns
                done = term or trunc
                steps += 1
            scores.append(score)
        return np.mean(scores)
    
    def validate_auto_labels(self):
        """
        Check how many auto-labels were correct by comparing with ground truth.
        This is a crucial diagnostic to ensure UCB/LCB is working properly.
        (Doc 1 Diagnostic Feature)
        """
        print(f"\n{'='*70}")
        print(f"VALIDATING AUTO-LABEL CORRECTNESS")
        print(f"{'='*70}")
        
        all_edges = self.graph.get_training_pairs()
        if len(all_edges) == 0:
            print("No edges to validate!")
            return 0.0
        
        correct = 0
        total = 0
        
        for winner_id, loser_id in all_edges:
            try:
                winner_reward = self.pref_buffer.get_trajectory(winner_id)['cumulative_reward']
                loser_reward = self.pref_buffer.get_trajectory(loser_id)['cumulative_reward']
                
                # Check if the edge direction matches ground truth
                is_correct = winner_reward > loser_reward
                
                if is_correct:
                    correct += 1
                total += 1
            except KeyError:
                continue
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"   Total preferences in graph: {total}")
        print(f"   Correct predictions: {correct}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy > 95:
            print(f"    EXCELLENT: Auto-labels are highly accurate!")
        elif accuracy > 85:
            print(f"    GOOD: Auto-labels are mostly correct")
        elif accuracy > 70:
            print(f"     WARNING: Moderate accuracy, consider increasing beta")
        else:
            print(f"    CRITICAL: Low accuracy, UCB/LCB may be failing")
        
        print(f"{'='*70}\n")
        
        self.auto_label_accuracy_history.append({
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        })
        
        return accuracy
    
    def record_video(self, save_path='videos', n_episodes=3):
        """
        Record video of agent behavior. Works with any gym environment.
        """
        try:
            import gymnasium as gym
            from gymnasium.wrappers import RecordVideo
            
            print(f"\n{'='*70}")
            print(f"[VIDEO RECORDING]")
            print(f"{'='*70}")
            print(f"   Recording {n_episodes} episodes...")
            print(f"   Output: {save_path}/")
            print()
            
            # Create temporary environment with video recording
            video_env = gym.make(self.env_name, render_mode='rgb_array')
            video_env = RecordVideo(
                video_env, 
                video_folder=save_path,
                name_prefix='final_agent',
                episode_trigger=lambda x: True
            )
            
            for ep in range(n_episodes):
                obs, _ = video_env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                while not done and steps < 1000:
                    obs_tensor = obs
                    action = self.sac_agent.select_action(obs_tensor, deterministic=True)
                    
                    # Handle discrete actions
                    if hasattr(video_env.action_space, 'n'):
                        action = int(action > 0.5)
                    
                    obs, reward, terminated, truncated, _ = video_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1
                
                print(f"   Episode {ep+1}/{n_episodes}: {steps} steps, reward={total_reward:.1f}")
            
            video_env.close()
            print(f"\n   ✓ Videos saved to: {save_path}/")
            print(f"{'='*70}\n")
            
        except ImportError:
            print("    Video recording requires: pip install moviepy")
        except Exception as e:
            print(f"    Video recording failed: {e}")
    
    def evaluate_final_performance(self, n_episodes=10):
        """Compare final agent to random baseline (Doc 1)"""
        print(f"\n{'='*70}")
        print(f"FINAL EVALUATION ({n_episodes} episodes)")
        print(f"{'='*70}")
        
        # Random policy
        print("\n   Evaluating Random Policy...")
        random_scores = []
        for _ in tqdm(range(n_episodes), desc="   Random", leave=False):
            s, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                a = self.env.env.action_space.sample()
                ns, r, term, trunc, _ = self.env.step(a)
                score += r
                done = term or trunc
            random_scores.append(score)
            
        # Trained agent
        print("   Evaluating Trained Agent...")
        agent_scores = []
        for _ in tqdm(range(n_episodes), desc="   Trained", leave=False):
            s, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                a = self.sac_agent.select_action(s, deterministic=True)
                ns, r, term, trunc, _ = self.env.step(a)
                score += r
                s = ns
                done = term or trunc
            agent_scores.append(score)
            
        avg_rnd = np.mean(random_scores)
        avg_agt = np.mean(agent_scores)
        std_rnd = np.std(random_scores)
        std_agt = np.std(agent_scores)
        improvement = (avg_agt / avg_rnd) if avg_rnd > 0 else float('inf')
        
        print(f"\n   {'─'*66}")
        print(f"   PERFORMANCE COMPARISON")
        print(f"   {'─'*66}")
        print(f"   Random Policy:  {avg_rnd:.1f} ± {std_rnd:.1f}")
        print(f"   Trained Agent:  {avg_agt:.1f} ± {std_agt:.1f}")
        print(f"   Improvement:    {improvement:.2f}x")
        print(f"   {'─'*66}")
        
        if avg_agt > avg_rnd * 2:
            print(f"\n    SUCCESS: Agent significantly outperforms random!")
        elif avg_agt > avg_rnd * 1.2:
            print(f"\n    PARTIAL SUCCESS: Agent shows improvement")
        else:
            print(f"\n     WARNING: Agent not significantly better than random")
        
        print(f"{'='*70}\n")
        
        return avg_rnd, avg_agt

    def plot_diagnostics(self, save_path='diagnostics/diagnostics.png'):
        """Enhanced diagnostics with 6 plots (Doc 1)"""
        if len(self.query_log) == 0:
            print("     No query log data to plot.")
            return
        
        print(f"\n   Generating diagnostic plots...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Extract data
        iterations = [log['iteration'] for log in self.query_log]
        rounds = [log['round'] for log in self.query_log]
        
        # Plot 1: Augmentation Ratio
        ax1 = fig.add_subplot(gs[0, 0])
        ratios = [log['augmentation'] for log in self.query_log]
        ax1.plot(iterations, ratios, linewidth=2, color='purple', marker='o', markersize=3, alpha=0.7)
        ax1.axhline(y=3.0, color='red', linestyle='--', label='Target: 3.0x', alpha=0.5)
        ax1.set_xlabel('Iteration Number', fontsize=11)
        ax1.set_ylabel('Augmentation Ratio', fontsize=11)
        ax1.set_title('Data Augmentation Over Time', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Auto-label Rate
        ax2 = fig.add_subplot(gs[0, 1])
        auto_rates = [log['auto_rate'] for log in self.query_log]
        ax2.plot(iterations, auto_rates, linewidth=2, color='green', marker='s', markersize=3, alpha=0.7)
        ax2.set_xlabel('Iteration Number', fontsize=11)
        ax2.set_ylabel('Auto-label Rate (%)', fontsize=11)
        ax2.set_title('UCB/LCB Filter Efficiency', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Graph Growth
        ax3 = fig.add_subplot(gs[1, 0])
        edges = [log['graph_edges'] for log in self.query_log]
        ax3.plot(iterations, edges, linewidth=2, color='blue', marker='^', markersize=3, alpha=0.7)
        ax3.set_xlabel('Iteration Number', fontsize=11)
        ax3.set_ylabel('Total Graph Edges', fontsize=11)
        ax3.set_title('Preference Graph Growth', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Ensemble Uncertainty
        ax4 = fig.add_subplot(gs[1, 1])
        stds = [log['ensemble_std_mean'] for log in self.query_log]
        ax4.plot(iterations, stds, linewidth=2, color='orange', marker='d', markersize=3, alpha=0.7)
        ax4.set_xlabel('Iteration Number', fontsize=11)
        ax4.set_ylabel('Mean Uncertainty (σ)', fontsize=11)
        ax4.set_title('Ensemble Uncertainty Over Time', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Plot 5: Reward Model Accuracy
        ax5 = fig.add_subplot(gs[2, 0])
        if len(self.reward_correlations) > 0:
            corr_iters = [c['iteration'] for c in self.reward_correlations]
            corr_vals = [c['correlation'] for c in self.reward_correlations]
            ax5.plot(corr_iters, corr_vals, linewidth=2, color='crimson', marker='o', markersize=4)
            ax5.axhline(y=0.9, color='green', linestyle='--', label='Target: 0.9', alpha=0.5)
            ax5.set_xlabel('Human Queries', fontsize=11)
            ax5.set_ylabel('Spearman Correlation', fontsize=11)
            ax5.set_title('Reward Model Accuracy', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No correlation data', ha='center', va='center', fontsize=12)
            ax5.set_title('Reward Model Accuracy', fontsize=12, fontweight='bold')
        
        # Plot 6: Agent Performance
        ax6 = fig.add_subplot(gs[2, 1])
        if len(self.sac_performance) > 0:
            perf_steps = [p['step'] for p in self.sac_performance]
            perf_rewards = [p['avg_reward'] for p in self.sac_performance]
            perf_rounds = [p['round'] for p in self.sac_performance]
            
            # Color by round
            scatter = ax6.scatter(perf_steps, perf_rewards, c=perf_rounds, cmap='viridis', 
                                 s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax6.plot(perf_steps, perf_rewards, linewidth=1.5, color='green', alpha=0.5)
            
            # Add random baseline if available
            if hasattr(self, '_random_baseline'):
                ax6.axhline(y=self._random_baseline, color='gray', linestyle='--', 
                           label='Random Policy', alpha=0.7, linewidth=2)
            
            ax6.set_xlabel('Training Steps', fontsize=11)
            ax6.set_ylabel('Average Episode Reward', fontsize=11)
            ax6.set_title('Agent Performance During Training', fontsize=12, fontweight='bold')
            cbar = plt.colorbar(scatter, ax=ax6)
            cbar.set_label('Round', fontsize=10)
            if hasattr(self, '_random_baseline'):
                ax6.legend()
            ax6.grid(alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No performance data', ha='center', va='center', fontsize=12)
            ax6.set_title('Agent Performance During Training', fontsize=12, fontweight='bold')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Diagnostics saved: {save_path}")
        plt.close()
    
    def plot_dual_agent_stats(self, save_path='diagnostics/dual_agent_stats.png'):
        """Plot dual-agent specific statistics (Doc 2 Feature)"""
        if len(self.loop_stats) == 0:
            print("     No dual-agent stats to plot.")
            return
        
        print(f"   Generating dual-agent diagnostic plots...")
        
        df = pd.DataFrame(self.loop_stats)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dual-Agent Statistics Across Rounds', fontsize=14, fontweight='bold')
        
        rounds = df['round'].values
        
        # Plot 1: Wins by Agent
        ax1 = axes[0, 0]
        ax1.bar(rounds - 0.15, df['SAC_Wins'], width=0.3, label='SAC Wins', color='steelblue')
        ax1.bar(rounds + 0.15, df['Entropy_Wins'], width=0.3, label='Entropy Wins', color='coral')
        ax1.set_xlabel('Round', fontsize=11)
        ax1.set_ylabel('Number of Wins', fontsize=11)
        ax1.set_title('Defender Dethronements by Agent', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # Plot 2: Auto-Rejections by Agent
        ax2 = axes[0, 1]
        ax2.bar(rounds - 0.15, df['SAC_Auto'], width=0.3, label='SAC Auto-Reject', color='lightcoral')
        ax2.bar(rounds + 0.15, df['Entropy_Auto'], width=0.3, label='Entropy Auto-Reject', color='lightsalmon')
        ax2.set_xlabel('Round', fontsize=11)
        ax2.set_ylabel('Number of Auto-Rejections', fontsize=11)
        ax2.set_title('UCB/LCB Auto-Rejections by Agent', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        
        # Plot 3: Defender Source Over Rounds
        ax3 = axes[1, 0]
        source_colors = {'SAC': 'steelblue', 'Entropy': 'coral', 'Unknown': 'gray'}
        colors = [source_colors.get(src, 'gray') for src in df['defender_source']]
        ax3.bar(rounds, df['defender_true_reward'], color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Round', fontsize=11)
        ax3.set_ylabel('Defender True Reward', fontsize=11)
        ax3.set_title('Defender Quality by Source', fontsize=12, fontweight='bold')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', label='SAC'),
                          Patch(facecolor='coral', label='Entropy')]
        ax3.legend(handles=legend_elements)
        ax3.grid(alpha=0.3, axis='y')
        
        # Plot 4: Total Engagement (Queries + Auto)
        ax4 = axes[1, 1]
        sac_total = df['SAC_Query'] + df['SAC_Auto']
        entropy_total = df['Entropy_Query'] + df['Entropy_Auto']
        
        ax4.plot(rounds, sac_total, marker='o', linewidth=2, label='SAC Total', color='steelblue')
        ax4.plot(rounds, entropy_total, marker='s', linewidth=2, label='Entropy Total', color='coral')
        ax4.set_xlabel('Round', fontsize=11)
        ax4.set_ylabel('Total Interactions', fontsize=11)
        ax4.set_title('Agent Engagement Over Time', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Dual-agent stats saved: {save_path}")
        plt.close()

    # ==================== MAIN DUAL-AGENT LOOP (Doc 2 Architecture) ====================

    def run(self):
        """
        Complete Dual-Agent Training Pipeline.
        Architecture from Doc 2, Diagnostics from Doc 1.
        """
        print(f"\n{'#'*70}")
        print(f"{'#'*70}")
        print(f"##  STARTING DUAL-AGENT HYBRID RLHF TRAINING")
        print(f"{'#'*70}")
        print(f"{'#'*70}\n")
        
        # ==================== INITIAL WARMUP ====================
        self.phase1_warmup(n_episodes=self.config.get('n_trajectories', 50))
        self.bootstrap_ensemble()
        
        # Store random baseline for comparison (Doc 1)
        print(f"\n[BASELINE EVALUATION]")
        print("Measuring random policy performance...")
        self._random_baseline = self.quick_evaluation(n_episodes=10)
        print(f"Random policy average: {self._random_baseline:.1f}\n")
        
        # ==================== DUAL-AGENT ITERATIVE LOOP ====================
        n_rounds = self.config.get('n_rounds', 5)
        entropy_decay = self.config.get('entropy_decay', 0.8)
        for round_num in range(1, n_rounds + 1):
            print(f"\n{'#'*70}")
            print(f"##  ROUND {round_num} / {n_rounds}")
            print(f"{'#'*70}\n")
            
            # A. Active Learning (Find the best data)
            self.update_defender()
            self.phase2_active_learning(
                n_queries=self.config.get('n_queries_per_round', 20),
                pool_size=self.config.get('pool_size', 20),
                round_num=round_num
            )
            self.retrain_ensemble_and_relabel(verbose=True)
            
            # B. Train Policy (SAC)
            self.phase3_policy_learning(
                n_steps=self.config.get('sac_steps_per_round', 5000),
                round_num=round_num
            )
            
            # C. Generate New Data (Dual-Agent Collection)
            print(f"\n{'='*70}")
            print(f"[DATA GENERATION] Round {round_num}")
            print(f"{'='*70}\n")
            
            # 1. Exploitation (SAC) - Generate from learned policy
            print("   [1/2] SAC Collection (Exploitation)...")
            sac_episodes = self.config.get('sac_collection_episodes', 10)
            sac_states = self._collect_and_store(self.sac_agent, sac_episodes, "SAC")
            
            # 2. Exploration (Entropy) - Continue exploring
            print("\n   [2/2] Entropy Collection (Exploration)...")
            
            base_entropy_eps = self.config.get('entropy_collection_episodes', 10)
            current_entropy_eps = int(base_entropy_eps * (entropy_decay ** (round_num - 1)))
            current_entropy_eps = max(1, current_entropy_eps) # Safety: always collect at least 1
            
            print(f"   > Decay applied: {base_entropy_eps} -> {current_entropy_eps} episodes (Rate: {entropy_decay})")
            
            # Shared Knowledge Feature (Doc 2)
            if len(sac_states) > 0 and self.config.get('entropy_full_buffer', False):
                print("   Sharing SAC states with Entropy agent...")
                if self.config.get('entropy_full_buffer', False):
                    # Full Buffer (Thorough but Slow)
                   for s in tqdm(sac_states, desc="   Sharing", leave=False):
                       self.entropy_agent.agent.add_state(s)
                else:
                    # Subsampling (Fast)
                    n_samples = min(len(sac_states), 2000)
                    idx = np.random.choice(len(sac_states), n_samples, replace=False)
                    for s in tqdm(sac_states[idx], desc="   Sharing", leave=False):
                        self.entropy_agent.agent.add_state(s)
            
            self._collect_and_store(self.entropy_agent, current_entropy_eps, "Entropy")
            
            
            print(f"\n   Total Buffer Size: {len(self.pref_buffer.get_all_ids())} trajectories")
            print(f"{'='*70}\n")
            
            # Intermediate Diagnostics
            print(f"[INTERMEDIATE EVALUATION] Round {round_num}")
            current_performance = self.quick_evaluation(n_episodes=10)
            improvement = (current_performance / self._random_baseline) if self._random_baseline > 0 else 0
            print(f"   Current Performance: {current_performance:.1f}")
            print(f"   vs Random Baseline:  {self._random_baseline:.1f}")
            print(f"   Improvement:         {improvement:.2f}x\n")
            
            # Save intermediate checkpoint
            self.save_checkpoint(f"round{round_num}_complete")
        
        # ==================== FINAL EVALUATION ====================
        print(f"\n{'#'*70}")
        print(f"##  FINAL EVALUATION & DIAGNOSTICS")
        print(f"{'#'*70}\n")
        
        # Validate auto-labels (Doc 1)
        self.validate_auto_labels()
        
        # Graph summary
        self.graph.print_summary()
        
        # Final performance comparison (Doc 1)
        avg_random, avg_agent = self.evaluate_final_performance(n_episodes=20)
        
        # Generate all diagnostics (Doc 1)
        self.plot_diagnostics()
        self.plot_dual_agent_stats()
        
        # Export CSV logs (Doc 2)
        print("\n[EXPORTING LOGS]")
        
        # Query log
        if self.query_log:
            df_queries = pd.DataFrame(self.query_log)
            query_path = os.path.join(self.logger.log_dir, "query_log.csv")
            df_queries.to_csv(query_path, index=False)
            print(f"    Query log: {query_path}")
        
        # Dual-agent stats
        if self.loop_stats:
            df_loop = pd.DataFrame(self.loop_stats)
            loop_path = os.path.join(self.logger.log_dir, "dual_agent_stats.csv")
            df_loop.to_csv(loop_path, index=False)
            print(f"    Dual-agent stats: {loop_path}")
            print(f"\n{df_loop.to_string()}\n")
        
        # Reward correlations
        if self.reward_correlations:
            df_corr = pd.DataFrame(self.reward_correlations)
            corr_path = os.path.join(self.logger.log_dir, "reward_correlations.csv")
            df_corr.to_csv(corr_path, index=False)
            print(f"    Reward correlations: {corr_path}")
        
        # SAC performance
        if self.sac_performance:
            df_perf = pd.DataFrame(self.sac_performance)
            perf_path = os.path.join(self.logger.log_dir, "sac_performance.csv")
            df_perf.to_csv(perf_path, index=False)
            print(f"    SAC performance: {perf_path}")
        
        # Record final video (Doc 1)
        self.record_video(n_episodes=3)
        
        # Final summary
        print(f"\n{'#'*70}")
        print(f"##  EXPERIMENT COMPLETE")
        print(f"{'#'*70}")
        print(f"   Total Human Queries: {self.total_human_queries}")
        print(f"   Final Performance:   {avg_agent:.1f}")
        print(f"   Random Baseline:     {avg_random:.1f}")
        print(f"   Improvement:         {(avg_agent/avg_random):.2f}x")
        print(f"   Graph Edges:         {len(self.graph.get_training_pairs())}")
        print(f"   Trajectories:        {len(self.pref_buffer.get_all_ids())}")
        print(f"{'#'*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dual-Agent Hybrid RLHF Training System')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--beta', type=float, default=None, help='Override beta')
    parser.add_argument('--rounds', type=int, default=None, help='Override number of rounds')
    parser.add_argument('--bootstrap', type=int, default=None, help='Override bootstrap queries')
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.beta is not None: config['beta'] = args.beta
    if args.rounds is not None: config['n_rounds'] = args.rounds
    if args.bootstrap is not None: config['n_bootstrap'] = args.bootstrap
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Redirect stdout and stderr to file while keeping terminal output
    log_file = TeeOutput('logs/training_output.txt', mode='w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    
    try:
        trainer = HybridDualAgentTrainer(config)
        trainer.run()
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"Training output saved to: logs/training_output.txt")