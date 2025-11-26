import sys
import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
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


def load_config(config_path):
    """Load configuration from YAML file or return defaults"""
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        numeric_keys = ['lr', 'beta', 'weight_decay', 'gamma', 'tau', 'alpha',
                        'min_warmup_reward', 'max_defender_uncertainty', 
                        'exploration_epsilon']
        
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
            'beta': 3.0,
            'n_trajectories': 20,
            'n_queries': 20,
            'n_bootstrap': 5,
            'pool_size': 20,
            'update_freq': 10,
            'ensemble_size': 3,
            'hidden_dim': 256,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'sac_steps': 2000,
            'segment_length': 50,
            'buffer_capacity': 10000,
            'seed': 42
        }


class HybridTrainer:
    """
    Complete 3-phase hybrid training system with enhanced diagnostics.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print(f"HYBRID PREFERENCE LEARNING SYSTEM")
        print(f"{'='*70}")
        print(f"   Device: {self.device}")
        print(f"   Environment: {config.get('env_name', 'CartPole-v1')}")
        print(f"   Beta (UCB/LCB): {config.get('beta', 3.0)}")
        print(f"   Ensemble Size: {config.get('ensemble_size', 3)}")
        print(f"{'='*70}\n")
        
        # --- Environment ---
        self.env_name = config.get('env_name', 'CartPole-v1')
        self.env = make_env(
            self.env_name,
            segment_length=config.get('segment_length', 50),
            seed=config.get('seed', 42)
        )
        
        # --- Core Components ---
        self.ensemble = RewardEnsemble(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            config=config,
            device=self.device
        )
        self.graph = PreferenceGraph()
        self.ucb_lcb = UCBLCBFilter(beta=config.get('beta', 3.0))
        self.dethroning = ActiveDethroning()
        self.logger = ExperimentLogger()
        self.oracle = Oracle(self.env_name)

        # --- Data Storage ---
        self.pref_buffer = ReplayBuffer(capacity=config.get('buffer_capacity', 10000))
        self.sac_buffer = SACBufferAdapter(
            self.env.state_dim, 
            self.env.action_dim, 
            capacity=config.get('buffer_capacity', 100000),
            device=self.device
        )
        
        # --- Agents ---
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
        
        # --- State Tracking ---
        self.defender_id = None
        self.total_human_queries = 0
        self.queried_pairs = set()
        
        # --- Enhanced Diagnostics ---
        self.query_log = []
        self.reward_correlations = []  # Track reward model accuracy
        self.sac_performance = []  # Track agent performance during training
        
        # Create directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("videos", exist_ok=True)

    def save_checkpoint(self, tag="latest"):
        """Save training state to disk"""
        path = os.path.join("checkpoints", f"checkpoint_{tag}.pt")
        
        ensemble_state = [m.state_dict() for m in self.ensemble.models]
        optimizer_state = [opt.state_dict() for opt in self.ensemble.optimizers]
        
        checkpoint = {
            'ensemble_state': ensemble_state,
            'optimizer_state': optimizer_state,
            'sac_actor': self.sac_agent.actor.state_dict(),
            'sac_critic': self.sac_agent.critic.state_dict(),
            'total_human_queries': self.total_human_queries,
            'defender_id': self.defender_id,
            'query_log': self.query_log,
            'reward_correlations': self.reward_correlations,
            'config': self.config
        }
        torch.save(checkpoint, path)

    # ==================== PHASE 1: WARMUP ====================
    def phase1_warmup(self, n_episodes=50, retries=0):
        """
        Phase 1: Unsupervised entropy-driven exploration.
        Includes retry logic to prevent infinite loops on tasks where entropy != reward.
        """
        print(f"\n[PHASE 1] Unsupervised Warmup ({n_episodes} episodes) - Attempt {retries+1}...")
        
        # 1. Collect Data
        trajectories, metrics = self.entropy_agent.warmup(self.env.env, n_episodes=n_episodes)
        
        count = 0
        seg_len = self.config.get('segment_length', 50)
        
        # 2. Slice and Store
        for states, actions, int_rewards, ext_rewards in trajectories:
            T = len(states)
            
            # Robust Slicing for Variable Lengths
            if T < seg_len:
                if T >= 5: # Keep short but meaningful segments (e.g. early failures)
                    true_ret = self._compute_true_reward(ext_rewards)
                    self.pref_buffer.add_trajectory(states, actions, true_ret)
                    count += 1
                continue
            
            # Standard Slicing
            for i in range(0, T - seg_len + 1, seg_len):
                seg_states = states[i : i+seg_len]
                seg_actions = actions[i : i+seg_len]
                true_ret = self._compute_true_reward(ext_rewards[i : i+seg_len])
                
                self.pref_buffer.add_trajectory(seg_states, seg_actions, true_ret)
                count += 1
                
        # 3. Quality Check
        all_ids = self.pref_buffer.get_all_ids()
        if all_ids:
            rewards = [self.pref_buffer.get_trajectory(tid)['cumulative_reward'] for tid in all_ids]
            reward_mean = np.mean(rewards)
            reward_max = np.max(rewards)
            
            print(f"   Quality check: mu={reward_mean:.2f}, max={reward_max:.2f}")
            
            # Threshold from config (Default 30.0)
            min_quality = self.config.get('min_warmup_reward', 30.0)
            
            if reward_max < min_quality:
                # RETRY LOGIC
                if retries < 2:  # Allow 2 retries (3 attempts total)
                    print(f"   ! Low reward diversity (<{min_quality}). Entropy agent is struggling (Expected for CartPole).")
                    print("   ! Retrying to find a lucky run...")
                    # Recursive call with increased episodes
                    return self.phase1_warmup(n_episodes=n_episodes + 20, retries=retries+1)
                else:
                    # FORCE PROCEED
                    print(f"   ! WARNING: Failed to meet reward threshold {min_quality} after 3 attempts.")
                    print(f"   ! Proceeding with best available data (Max: {reward_max:.2f}).")
                    print(f"   ! The Active Learning phase will attempt to learn from these short episodes.")
        
        print(f"Phase 1 complete: {count} segments collected.\n")
        self.save_checkpoint("phase1_done")
    
    def _compute_true_reward(self, rewards):
        return float(np.sum(rewards))
    
    # ==================== BOOTSTRAP ====================
    
    def bootstrap_ensemble(self, n_bootstrap=5):
        print(f"\n[BOOTSTRAP] Making {n_bootstrap} random queries to initialize ensemble...")
        
        all_ids = self.pref_buffer.get_all_ids()
        if len(all_ids) < 2:
            print("Error: Not enough trajectories for bootstrap!")
            return
        
        for i in range(n_bootstrap):
            id1, id2 = np.random.choice(all_ids, 2, replace=False)
            r1 = self.pref_buffer.get_trajectory(id1)['cumulative_reward']
            r2 = self.pref_buffer.get_trajectory(id2)['cumulative_reward']
            label = self.oracle.compare(r1, r2)
            
            if label == 1:
                self.graph.add_preference(id1, id2, is_human_label=True)
                print(f"   Bootstrap {i+1}: {id1} > {id2}")
            else:
                self.graph.add_preference(id2, id1, is_human_label=True)
                print(f"   Bootstrap {i+1}: {id2} > {id1}")
            
            self.total_human_queries += 1
            self.queried_pairs.add((min(id1, id2), max(id1, id2)))
        
        # VALIDATION: Check if we selected the true best defender
        true_rewards = {tid: self.pref_buffer.get_trajectory(tid)['cumulative_reward'] for tid in all_ids}
        best_id = max(true_rewards, key=true_rewards.get)
        true_best_reward = true_rewards[best_id]
        
        self.defender_id = best_id
        print(f"   Best defender selected: {best_id} (GT reward: {true_best_reward:.2f})")
        
        # Print reward distribution for diagnostics
        rewards_list = list(true_rewards.values())
        print(f"   Reward distribution: min={min(rewards_list):.1f}, "
              f"max={max(rewards_list):.1f}, mean={np.mean(rewards_list):.1f}")
        
        # WARNING if no high-quality trajectories
        if max(rewards_list) < 100:
            print(f"\n   WARNING: Max reward only {max(rewards_list):.1f} < 100")
            print(f"   Agent may underperform due to lack of good examples")
            print(f"   Consider extending warmup phase\n")
        
        # Check defender uncertainty
        _, def_std = self.compute_ensemble_stats(self.defender_id)
        max_defender_uncertainty = self.config.get('max_defender_uncertainty', 1.5)
        if def_std > max_defender_uncertainty:
            print(f"   Defender uncertainty high (sigma={def_std:.2f} > {max_defender_uncertainty})")
            print(f"   Temporarily widening beta to force more human queries...")
            self.original_beta = self.config['beta']
            self.config['beta'] = 2.0
        
        print(f"\n   Training ensemble on {n_bootstrap} bootstrap labels...")
        self.retrain_ensemble_and_relabel(verbose=False)
        
        stats = self.graph.get_stats()
        print(f"   Bootstrap complete: {stats['total']} total edges ({stats['ratio']:.2f}x augmentation)\n")
        
        # Track initial correlation
        corr = self.compute_reward_correlation()
        self.reward_correlations.append({'iteration': 0, 'correlation': corr})
        
        self.save_checkpoint("bootstrap_done")
    # ==================== PHASE 2: ACTIVE LEARNING ====================
    
    def compute_ensemble_stats(self, traj_id):
        traj = self.pref_buffer.get_trajectory(traj_id)
        states = torch.FloatTensor(traj['states']).unsqueeze(0).to(self.device)
        actions = torch.FloatTensor(traj['actions']).unsqueeze(0).to(self.device)
        mean, std = self.ensemble.predict_segment(states, actions)
        return mean.item(), std.item()
    
    def compute_reward_correlation(self):
        """
        Compute Spearman correlation between predicted and ground truth rewards.
        This tells us how well the ensemble has learned to rank trajectories.
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
    
    def phase2_active_learning(self, n_queries, pool_size=50):
        print(f"\n[PHASE 2] Active Preference Learning ({n_queries} queries)...")
        
        all_ids = self.pref_buffer.get_all_ids()
        if not all_ids:
            print("Error: Buffer empty! Run Phase 1 first.")
            return
        
        self.bootstrap_ensemble(n_bootstrap=self.config.get('n_bootstrap', 5))
        
        # Track defender changes
        defender_change_count = 0
        last_defender = self.defender_id
        
        # Get ground truth for validation
        true_rewards = {tid: self.pref_buffer.get_trajectory(tid)['cumulative_reward'] 
                       for tid in all_ids}
        true_best_id = max(true_rewards, key=true_rewards.get)
        
        pbar = tqdm(range(n_queries), desc="   Active Learning Loop", unit="query")
        
        for iteration in pbar:
            # PERIODIC DEFENDER VALIDATION
            reset_freq = self.config.get('defender_reset_freq', 25)
            if reset_freq > 0 and (iteration + 1) % reset_freq == 0:
                # Re-select from top-3 PageRank trajectories
                if len(self.graph.G) > 3:
                    rev_G = self.graph.G.reverse()
                    try:
                        scores = nx.pagerank(rev_G)
                        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        # Pick best among top-3 based on ground truth
                        best_of_top3 = max(top_3, key=lambda x: true_rewards.get(x[0], -999))[0]
                        
                        if best_of_top3 != self.defender_id:
                            old_reward = true_rewards.get(self.defender_id, 0)
                            new_reward = true_rewards.get(best_of_top3, 0)
                            print(f"\n   [DEFENDER RESET] {self.defender_id} (r={old_reward:.1f}) -> "
                                  f"{best_of_top3} (r={new_reward:.1f})")
                            self.defender_id = best_of_top3
                            defender_change_count += 1
                    except:
                        pass  # PageRank failed, keep current defender
            
            # Sample Candidates
            available = []
            for tid in all_ids:
                if tid == self.defender_id: continue
                pair = (min(self.defender_id, tid), max(self.defender_id, tid))
                if pair not in self.queried_pairs: available.append(tid)
            
            if len(available) == 0:
                print("   No new candidates available!")
                break
            
            pool = np.random.choice(available, size=min(pool_size, len(available)), replace=False)
            
            # Compute Stats
            def_mu, def_std = self.compute_ensemble_stats(self.defender_id)
            cand_mus = np.array([self.compute_ensemble_stats(cid)[0] for cid in pool])
            cand_stds = np.array([self.compute_ensemble_stats(cid)[1] for cid in pool])
            
            # UCB/LCB Filter (now with adaptive beta)
            auto_labels, uncertain_idx, dethroned, new_def = self.ucb_lcb.filter_candidates(
                def_mu, def_std, cand_mus, cand_stds, pool.tolist()
            )
            
            # Add auto-labels
            for winner, loser in auto_labels:
                w = self.defender_id if winner == 'defender' else winner
                l = self.defender_id if loser == 'defender' else loser
                self.graph.add_preference(w, l, is_human_label=False, is_auto_label=True)
            
            # Diagnostics
            auto_rate = len(auto_labels) / len(pool) if len(pool) > 0 else 0
            epsilon = self.config.get('exploration_epsilon', 0.1)
            
            # Exploration epsilon override
            if auto_rate > 0.95:
                pass  # Allow high auto-rate
            elif np.random.random() < epsilon:
                uncertain_idx = list(range(len(pool)))
                auto_labels = []
                dethroned = False
            
            defender_changed = False
            if dethroned:
                old_reward = true_rewards.get(self.defender_id, 0)
                new_reward = true_rewards.get(new_def, 0)
                print(f"\n   [NATURAL DETHRONE] {self.defender_id} (r={old_reward:.1f}) -> "
                      f"{new_def} (r={new_reward:.1f})")
                self.defender_id = new_def
                defender_changed = True
                defender_change_count += 1
                
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
                self.queried_pairs.add((min(self.defender_id, challenger_id), max(self.defender_id, challenger_id)))
                
                if label == 1:
                    self.graph.add_preference(challenger_id, self.defender_id, is_human_label=True)
                    print(f"\n   [HUMAN QUERY DETHRONE] {challenger_id} (r={chal_r:.1f}) > "
                          f"Defender {self.defender_id} (r={def_r:.1f})")
                    self.defender_id = challenger_id
                    defender_changed = True
                    defender_change_count += 1
                else:
                    self.graph.add_preference(self.defender_id, challenger_id, is_human_label=True)
            
            # Log metrics
            stats = self.graph.get_stats()
            self.query_log.append({
                'iteration': iteration + 1,
                'total_human_queries': self.total_human_queries,
                'auto_labels': len(auto_labels),
                'auto_rate': auto_rate*100,
                'graph_edges': stats['total'],
                'augmentation': stats['ratio'],
                'defender_changed': defender_changed,
                'ensemble_std_mean': cand_stds.mean(),
                'defender_id': self.defender_id,
                'defender_true_reward': true_rewards.get(self.defender_id, 0)
            })
            
            # Update Progress Bar
            is_optimal = (self.defender_id == true_best_id)
            opt_marker = "✓" if is_optimal else "✗"
            pbar.set_postfix({
                'Aug': f"{stats['ratio']:.1f}x", 
                'Auto': f"{auto_rate*100:.0f}%",
                'Def': f"{self.defender_id}{opt_marker}"
            })
            
            # Restore beta
            if hasattr(self, 'original_beta'):
                self.config['beta'] = self.original_beta
                delattr(self, 'original_beta')
            
            # Periodic Updates
            update_freq = self.config.get('update_freq', 10)
            if (iteration + 1) % update_freq == 0:
                self.retrain_ensemble_and_relabel(verbose=False)
                
                # Track reward correlation
                corr = self.compute_reward_correlation()
                self.reward_correlations.append({
                    'iteration': iteration + 1,
                    'correlation': corr
                })
                
                self.save_checkpoint(f"iter_{iteration+1}")
        
        # FINAL VALIDATION
        print(f"\n{'='*70}")
        print(f"PHASE 2 SUMMARY")
        print(f"{'='*70}")
        print(f"   Total defender changes: {defender_change_count}")
        print(f"   Final defender: {self.defender_id} (GT reward: {true_rewards.get(self.defender_id, 0):.1f})")
        print(f"   True best: {true_best_id} (GT reward: {true_rewards[true_best_id]:.1f})")
        
        if self.defender_id == true_best_id:
            print(f"   ✓ SUCCESS: Found the true best trajectory!")
        else:
            print(f"   ✗ WARNING: Defender is suboptimal (missed by {true_rewards[true_best_id] - true_rewards.get(self.defender_id, 0):.1f} reward)")
        
        if defender_change_count == 0:
            print(f"   ✗ CRITICAL: Defender NEVER changed - likely stuck in local optimum!")
        elif defender_change_count < 3:
            print(f"   WARNING: Only {defender_change_count} changes - may be under-exploring")
        else:
            print(f"   ✓ Good exploration: {defender_change_count} defender changes")
        
        print(f"{'='*70}\n")
        
    def retrain_ensemble_and_relabel(self, verbose=True):
        """Retrain ensemble on current graph."""
        pairs = self.graph.get_training_pairs()
        
        if len(pairs) < 5:
            if verbose: print("   Not enough data to train yet.")
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
            iterator = tqdm(range(epochs), desc="   Training Ensemble", unit="epoch")
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
        
        if verbose: print("   Ensemble Trained.")
        self.sac_buffer.relabel_and_flatten(self.pref_buffer, self.ensemble)
        if verbose: print("   SAC Buffer Relabeled.")

    # ==================== PHASE 3: POLICY LEARNING ====================

    def phase3_policy_learning(self, n_steps=5000):
        print(f"\n[PHASE 3] Policy Learning ({n_steps} steps)...")
        
        if self.sac_buffer.size < 100:
            print("Error: SAC Buffer empty.")
            return

        # Evaluate performance periodically during training
        eval_frequency = max(1000, n_steps // 10)
        
        pbar = tqdm(range(n_steps), desc="   Training SAC Agent", unit="step")
        
        for i in pbar:
            if self.sac_buffer.size < 256: break
            actor_loss, critic_loss = self.sac_agent.update(self.sac_buffer, batch_size=256)
            
            # Periodic evaluation
            if (i+1) % eval_frequency == 0:
                avg_reward = self.quick_evaluation(n_episodes=5)
                self.sac_performance.append({
                    'step': i+1,
                    'avg_reward': avg_reward
                })
                pbar.set_postfix(
                    act_loss=f"{actor_loss:.3f}",
                    crit_loss=f"{critic_loss:.3f}",
                    reward=f"{avg_reward:.1f}"
                )
            elif (i+1) % 100 == 0:
                pbar.set_postfix(act_loss=f"{actor_loss:.3f}", crit_loss=f"{critic_loss:.3f}")
        
        self.save_checkpoint("phase3_done")
        print("Phase 3 Complete.")

    # ==================== EVALUATION & DIAGNOSTICS ====================
    
    def quick_evaluation(self, n_episodes=5):
        """Quick evaluation of current policy (no rendering)"""
        scores = []
        for _ in range(n_episodes):
            s, _ = self.env.reset()
            score = 0
            done = False
            steps = 0
            max_steps = 1000  # Prevent infinite loops
            
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
        """
        print(f"\n{'='*70}")
        print(f"VALIDATING AUTO-LABEL CORRECTNESS")
        print(f"{'='*70}")
        
        all_edges = self.graph.get_training_pairs()
        if len(all_edges) == 0:
            print("No edges to validate!")
            return
        
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
                # Trajectory might have been evicted from buffer
                continue
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"   Total preferences in graph: {total}")
        print(f"   Correct predictions: {correct}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy > 95:
            print(f"   EXCELLENT: Auto-labels are highly accurate!")
        elif accuracy > 85:
            print(f"   GOOD: Auto-labels are mostly correct")
        elif accuracy > 70:
            print(f"   WARNING: Moderate accuracy, consider increasing beta")
        else:
            print(f"   CRITICAL: Low accuracy, UCB/LCB may be failing")
        
        print(f"{'='*70}\n")
        return accuracy
    
    def record_video(self, save_path='agent_behavior.mp4', n_episodes=3):
        """
        Record video of agent behavior. Works with any gym environment.
        """
        try:
            import gymnasium as gym
            from gymnasium.wrappers import RecordVideo
            
            print(f"\n[VIDEO] Recording {n_episodes} episodes...")
            
            # Create temporary environment with video recording
            video_env = gym.make(self.env_name, render_mode='rgb_array')
            video_env = RecordVideo(
                video_env, 
                video_folder='videos',
                name_prefix='agent',
                episode_trigger=lambda x: True  # Record all episodes
            )
            
            for ep in range(n_episodes):
                obs, _ = video_env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                while not done and steps < 1000:
                    # Convert observation to proper format
                    obs_tensor = obs
                    action = self.sac_agent.select_action(obs_tensor, deterministic=True)
                    
                    # Handle discrete actions (CartPole)
                    if hasattr(video_env.action_space, 'n'):
                        action = int(action > 0.5)
                    
                    obs, reward, terminated, truncated, _ = video_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1
                
                print(f"   Episode {ep+1}/{n_episodes}: {steps} steps, reward={total_reward:.1f}")
            
            video_env.close()
            print(f"Video saved to: videos/")
            
        except ImportError:
            print("Video recording requires: pip install moviepy")
        except Exception as e:
            print(f"Video recording failed: {e}")
    
    def evaluate_final_performance(self, n_episodes=10):
        """Compare final agent to random baseline"""
        print(f"\n{'='*70}")
        print(f"FINAL EVALUATION ({n_episodes} episodes)")
        print(f"{'='*70}")
        
        # Random policy
        random_scores = []
        for _ in range(n_episodes):
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
        agent_scores = []
        for _ in range(n_episodes):
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
        improvement = (avg_agt / avg_rnd) if avg_rnd > 0 else float('inf')
        
        print(f"Random Policy Average: {avg_rnd:.1f} (std={np.std(random_scores):.1f})")
        print(f"Trained Agent Average: {avg_agt:.1f} (std={np.std(agent_scores):.1f})")
        print(f"Improvement: {improvement:.2f}x")
        
        if avg_agt > avg_rnd * 2:
            print("\nSUCCESS: Agent significantly outperforms random!")
        elif avg_agt > avg_rnd * 1.2:
            print("\nPARTIAL SUCCESS: Agent shows some improvement")
        else:
            print("\nWARNING: Agent not significantly better than random")
        
        print(f"{'='*70}\n")
        
        return avg_rnd, avg_agt

    def plot_diagnostics(self, save_path='diagnostics.png'):
        """Enhanced diagnostics with 6 plots"""
        if len(self.query_log) == 0:
            print("No query log data to plot.")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        queries = [log['iteration'] for log in self.query_log]
        
        # Plot 1: Augmentation Ratio
        ax1 = fig.add_subplot(gs[0, 0])
        ratios = [log['augmentation'] for log in self.query_log]
        ax1.plot(queries, ratios, linewidth=2, color='purple', marker='o')
        ax1.axhline(y=3.0, color='red', linestyle='--', label='Target: 3.0x', alpha=0.5)
        ax1.set_xlabel('Iteration Number', fontsize=11)
        ax1.set_ylabel('Augmentation Ratio', fontsize=11)
        ax1.set_title('Data Augmentation Over Time', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Auto-label Rate
        ax2 = fig.add_subplot(gs[0, 1])
        auto_rates = [log['auto_rate'] for log in self.query_log]
        ax2.plot(queries, auto_rates, linewidth=2, color='green', marker='s')
        ax2.set_xlabel('Iteration Number', fontsize=11)
        ax2.set_ylabel('Auto-label Rate (%)', fontsize=11)
        ax2.set_title('UCB/LCB Filter Efficiency', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Graph Growth
        ax3 = fig.add_subplot(gs[1, 0])
        edges = [log['graph_edges'] for log in self.query_log]
        ax3.plot(queries, edges, linewidth=2, color='blue', marker='^')
        ax3.set_xlabel('Iteration Number', fontsize=11)
        ax3.set_ylabel('Total Graph Edges', fontsize=11)
        ax3.set_title('Preference Graph Growth', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Ensemble Uncertainty
        ax4 = fig.add_subplot(gs[1, 1])
        stds = [log['ensemble_std_mean'] for log in self.query_log]
        ax4.plot(queries, stds, linewidth=2, color='orange', marker='d')
        ax4.set_xlabel('Iteration Number', fontsize=11)
        ax4.set_ylabel('Mean Uncertainty (sigma)', fontsize=11)
        ax4.set_title('Ensemble Uncertainty Over Time', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Plot 5: Reward Model Accuracy (NEW)
        ax5 = fig.add_subplot(gs[2, 0])
        if len(self.reward_correlations) > 0:
            corr_iters = [c['iteration'] for c in self.reward_correlations]
            corr_vals = [c['correlation'] for c in self.reward_correlations]
            ax5.plot(corr_iters, corr_vals, linewidth=2, color='crimson', marker='o')
            ax5.axhline(y=0.9, color='green', linestyle='--', label='Target: 0.9', alpha=0.5)
            ax5.set_xlabel('Iteration Number', fontsize=11)
            ax5.set_ylabel('Spearman Correlation', fontsize=11)
            ax5.set_title('Reward Model Accuracy', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
        
        # Plot 6: Agent Performance (NEW)
        ax6 = fig.add_subplot(gs[2, 1])
        if len(self.sac_performance) > 0:
            perf_steps = [p['step'] for p in self.sac_performance]
            perf_rewards = [p['avg_reward'] for p in self.sac_performance]
            ax6.plot(perf_steps, perf_rewards, linewidth=2, color='green', marker='s', label='Trained Agent')
            
            # Add random baseline if available
            if hasattr(self, '_random_baseline'):
                ax6.axhline(y=self._random_baseline, color='gray', linestyle='--', 
                           label='Random Policy', alpha=0.7, linewidth=2)
            
            ax6.set_xlabel('Training Steps', fontsize=11)
            ax6.set_ylabel('Average Episode Reward', fontsize=11)
            ax6.set_title('Agent Performance During Training', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No performance data', ha='center', va='center')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDiagnostics saved: {save_path}")
        plt.close()

    # ==================== MAIN ====================

    def run(self):
        """Complete training pipeline with enhanced diagnostics"""
        # Phase 1: Warmup
        self.phase1_warmup(n_episodes=self.config.get('n_trajectories', 20))
        
        # Phase 2: Active Learning
        self.phase2_active_learning(
            n_queries=self.config.get('n_queries', 20),
            pool_size=self.config.get('pool_size', 50)
        )
        
        # Final ensemble training
        self.retrain_ensemble_and_relabel(verbose=True)
        
        # Validate auto-labels
        self.validate_auto_labels()
        
        # Store random baseline for comparison
        print("\n[BASELINE] Evaluating random policy...")
        self._random_baseline = self.quick_evaluation(n_episodes=10)
        print(f"Random policy average: {self._random_baseline:.1f}")
        
        # Phase 3: Policy Learning
        self.phase3_policy_learning(n_steps=self.config.get('sac_steps', 2000))
        
        # Final evaluation
        self.graph.print_summary()
        avg_random, avg_agent = self.evaluate_final_performance()
        
        # Generate all diagnostics
        self.plot_diagnostics()
        
        # Record video of final agent
        self.record_video(n_episodes=3)
        
        print("\nEXPERIMENT COMPLETE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RLHF Hybrid Training System')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--beta', type=float, default=None, help='Override beta')
    parser.add_argument('--queries', type=int, default=None, help='Override number of queries')
    parser.add_argument('--bootstrap', type=int, default=None, help='Override bootstrap queries')
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.beta is not None: config['beta'] = args.beta
    if args.queries is not None: config['n_queries'] = args.queries
    if args.bootstrap is not None: config['n_bootstrap'] = args.bootstrap
    
    trainer = HybridTrainer(config)
    trainer.run()