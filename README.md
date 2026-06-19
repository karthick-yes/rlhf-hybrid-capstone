# Efficient Preference-Based Reward Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A Hybrid Framework Combining Structural Transitivity, Uncertainty-Driven Active Learning, and Stability Mechanisms**

A novel hybrid approach to Reinforcement Learning from Human Feedback (RLHF) that achieves **2× query efficiency** and **3.2:1 effective data augmentation** by synergistically combining three orthogonal techniques: transitive preference graphs (SeqRank), deep ensemble uncertainty quantification, and principled confidence-bound pseudo-labeling.

---

## Overview

Reinforcement Learning from Human Feedback (RLHF) is critical for aligning AI systems with human preferences, but it suffers from severe data inefficiency—often requiring tens of thousands of expensive human preference queries. This project introduces a hybrid framework that bridges the gap between LLM-centric computational efficiency and robotics-centric sample efficiency.

### Key Innovations

| Component | Contribution | Benefit |
|-----------|-----------|---------|
| **Transitive Augmentation** (SeqRank) | Preference graph with transitive closure | O(N²) labels from N queries |
| **UCB/LCB Pseudo-Labeling** | Confidence-bound auto-labeling rule | >99.8% accuracy, robust to epistemic uncertainty |
| **Active Dethroning** | Max-entropy acquisition function | Targets decision boundaries, solves the "CartPole Trap" |
| **Stability Mechanisms** (PEBBLE) | Entropy warmup + reward relabeling | Prevents cold-start collapse and reward drift |

---

## Architecture

The framework operates in three phases:

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Phase 1:       │────▶│  Phase 2:              │────▶│  Phase 3:       │
│  Unsupervised   │     │  Hybrid Active Loop    │     │  Policy Learning│
│  Warmup         │     │  (Core Contribution)   │     │  + Relabeling   │
│  (Entropy Max)  │     │                        │     │  (SAC)          │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐          ┌──────────┐
   │Preference│         │  Deep    │          │  UCB/LCB │
   │  Graph   │         │ Ensemble │          │  Filter  │
   │ (SeqRank)│         │(K=5 MLPs)│          │(β = 3.0) │
   └─────────┘          └──────────┘          └──────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                        ┌──────────────┐
                        │Active        │
                        │Dethroning    │
                        │(Max Entropy) │
                        └──────────────┘
```

---

## Installation

### Prerequisites
- Python >= 3.9
- CUDA-capable GPU (16GB VRAM recommended for MetaWorld)
- Linux or macOS

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/preference-rl-hybrid.git
cd preference-rl-hybrid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install MetaWorld (for manipulation tasks)
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0 | Deep ensemble reward models |
| `gymnasium` | >= 0.29 | Environment interface |
| `metaworld` | latest | Manipulation benchmarks |
| `networkx` | >= 3.0 | Preference graph & transitive closure |
| `numpy` | >= 1.24 | Numerical operations |
| `matplotlib` | >= 3.7 | Visualization |
| `seaborn` | >= 0.12 | Publication-quality plots |

---

## Quick Start

### 1. CartPole (Validation & Debugging)

```bash
python experiments/run_cartpole.py \
    --n_queries 100 \
    --ensemble_size 5 \
    --beta 3.0 \
    --warmup_steps 2000 \
    --seed 42
```

### 2. MetaWorld Door-Open

```bash
python experiments/run_metaworld.py \
    --task door-open-v2 \
    --n_queries 5000 \
    --warmup_steps 5000 \
    --checkpoint_dir checkpoints/
```

### 3. Ablation Study

```bash
python experiments/ablation.py \
    --components all transitivity ucb_lcb dethroning \
    --n_seeds 5 \
    --output_dir results/ablation/
```

---

## Project Structure

```
preference_rl_hybrid/
├── configs/                    # Environment hyperparameters
│   ├── cartpole.yaml
│   └── metaworld_door_open.yaml
├── src/
│   ├── envs/                   # Environment wrappers & oracle
│   │   ├── wrappers.py
│   │   └── oracle.py           # Simulated teacher (ground truth)
│   ├── models/                 # Neural network architectures
│   │   ├── reward_ensemble.py  # Deep ensemble (K=5 MLPs)
│   │   └── policy.py           # SAC actor-critic
│   ├── graph/                  # Preference graph logic
│   │   ├── preference_graph.py # NetworkX DAG + transitive closure
│   │   └── seqrank.py          # Defender selection (PageRank)
│   ├── selection/              # Active learning components
│   │   ├── ucb_lcb.py          # Confidence bound filtering
│   │   └── dethroning.py       # Max-entropy acquisition
│   ├── agents/                 # RL algorithms
│   │   ├── sac.py              # Soft Actor-Critic
│   │   └── entropy_agent.py    # Warmup exploration
│   ├── training/               # Training orchestration
│   │   ├── trainer.py          # Main hybrid loop
│   │   └── logger.py           # Metrics & checkpointing
│   └── utils/                  # Helpers
│       ├── replay_buffer.py
│       └── visualization.py    # Plotting utilities
├── tests/                      # Unit & integration tests
│   ├── test_ensemble.py
│   ├── test_graph.py
│   └── test_ucb_lcb.py
├── experiments/                # Experiment scripts
│   ├── run_cartpole.py
│   ├── run_metaworld.py
│   └── ablation.py
└── checkpoints/                # Saved models & logs
```

---

## Methodology Highlights

### UCB/LCB Augmentation Rule
Instead of ADPO's threshold-based pseudo-labeling, we use confidence bounds:

```python
UCB(σ) = μ(σ) + β · s(σ)
LCB(σ) = μ(σ) − β · s(σ)

# Auto-label only if intervals are disjoint
if LCB(σ_def) > UCB(σ_chal):
    label: σ_def ≻ σ_chal      # Defender wins (no human query)
elif LCB(σ_chal) > UCB(σ_def):
    label: σ_chal ≻ σ_def      # Challenger dethrones
else:
    query_human()                # Model uncertain—ask oracle
```

With β = 3.0, this guarantees >99.87% pseudo-label correctness under Gaussian assumptions.

### Active Dethroning Acquisition
When the model is uncertain, we select challengers that maximize information gain:

```
P(Win) = Φ(μ_Δ / s_Δ)
Acquisition = P(Win) · (1 − P(Win))   # Max-entropy
```

This targets the decision boundary where P(Win) ≈ 0.5, solving embedding-reward misalignment in unstable control tasks.

---

## Results

| Metric | Hybrid (Ours) | PEBBLE | SeqRank |
|--------|--------------|--------|---------|
| **Queries to ρ > 0.9** | **60** | 120+ | 100+ |
| **Augmentation Ratio** | **3.2:1** | 1.0:1 | 1.5–2.0:1 |
| **CartPole Convergence** | **100%** | 60% | 75% |
| **MetaWorld Success Rate** | **>90%** | ~85% | ~80% |

*Evaluated on continuous control benchmarks including CartPole and MetaWorld manipulation tasks. Simulated oracle based on ground-truth rewards.*

---

## Testing

Run the checkpoint-driven test suite:

```bash
# Unit tests (continuous)
pytest tests/test_ensemble.py      # Ensemble calibration
pytest tests/test_graph.py         # Transitive closure
pytest tests/test_ucb_lcb.py       # Pseudo-label accuracy

# Integration test (Day 7 equivalent)
python tests/integration_test.py   # 2D navigation toy problem
```

### Validation Checkpoints

| Checkpoint | Validation Criteria | Status |
|------------|---------------------|--------|
| Ensemble Calibration | Pearson ρ(error, uncertainty) > 0.5 | ✅ |
| Transitive Augmentation | Ratio > 1.5× | ✅ |
| UCB/LCB Correctness | Auto-label accuracy > 95% | ✅ |
| Dethroning Sanity | Selected P(Win) ≈ 0.5 ± 0.1 | ✅ |
| CartPole Convergence | ρ > 0.9 in ≤ 100 queries | ✅ |
| MetaWorld Door Open | Success > 90% | ✅ |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{karthik_2025_hybrid_rlhf,
  title={Efficient Preference-Based Reward Learning: A Hybrid Framework Combining Structural Transitivity, Uncertainty-Driven Active Learning, and Stability Mechanisms},
  author={Karthik},
  year={2025},
  institution={Department of Computer Science and Engineering},
  type={B.Tech Capstone Project}
}
```

---

## References

- **SeqRank**: Hwang et al., "Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback," NeurIPS 2023
- **PEBBLE**: Lee et al., "PEBBLE: Feedback-Efficient Interactive RL via Relabeling Experience and Unsupervised Pre-training," ICML 2021
- **Deep Ensembles**: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," NeurIPS 2017
- **ADPO**: Myers et al., "Reinforcement Learning from Human Feedback with Active Queries," 2024
- **SAC**: Haarnoja et al., "Soft Actor-Critic," ICML 2018

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work was submitted in partial fulfillment of the requirements for the degree of Bachelor of Technology in Computer Science & Engineering by **Karthik**. Special thanks to the open-source RL community for PEBBLE, SeqRank, and MetaWorld implementations.

---

**Note**: This project uses simulated oracles (scripted teachers based on ground-truth rewards) for reproducible experimentation. Extension to real human annotators is identified as future work.
