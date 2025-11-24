import random
import matplotlib.pyplot as plt
import numpy as np
from src.graph.preference_graph import PreferenceGraph

def test_transitivity_with_plot():
    print("\n--- Checkpoint 2a: Transitive Augmentation ---")
    graph = PreferenceGraph()
    items = list(range(50))
    
    direct_history = []
    total_history = []
    
    for i in range(100):
        a, b = random.sample(items, 2)
        winner, loser = (a, b) if a > b else (b, a)
        graph.add_preference(winner, loser)
        
        if (i+1) % 5 == 0:
            stats = graph.get_stats()
            direct = stats['direct']
            total = len(graph.get_training_pairs())
            direct_history.append(direct)
            total_history.append(total)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(direct_history, label='Direct Queries', linewidth=2)
    plt.plot(total_history, label='Total Pairs (w/ Transitive)', linewidth=2)
    plt.fill_between(range(len(direct_history)), direct_history, total_history, 
                     alpha=0.3, label='Augmented Data')
    plt.xlabel('Query Batch (every 5 queries)')
    plt.ylabel('Number of Pairs')
    plt.title('SeqRank Transitive Augmentation')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('transitivity_test.png', dpi=300)
    print("Plot saved: transitivity_test.png")
    
    final_stats = graph.get_stats()
    final_direct = final_stats['direct']
    final_total = len(graph.get_training_pairs())
    ratio = final_total / final_direct
    print(f"Final Ratio: {ratio:.2f}x")
    assert ratio > 1.5, "Augmentation failed!"
    print("SUCCESS: Transitivity works")

def test_root_defender_with_ground_truth():
    print("\n--- Checkpoint 2b: Root Pairwise Defender ---")
    graph = PreferenceGraph()
    
    # Ground truth: 49=best, 0=worst
    print("Building graph where 49 should dominate...")
    
    comparisons = [
        (49, 45), (49, 40), (49, 35), (49, 30),
        (45, 30), (45, 20),
        (40, 25), (40, 15),
        (30, 10), (20, 5)
    ]
    
    for winner, loser in comparisons:
        graph.add_preference(winner, loser)
    
    defender = graph.get_defender()
    win_counts = {node: 0 for node in graph.G.nodes()}
    
    for winner, loser in graph.G.edges():
        win_counts[winner] += 1
    
    print(f"\nWin counts: {sorted(win_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"Selected Defender: {defender}")
    print(f"Defender's wins: {win_counts.get(defender, 0)}")
    
    assert defender == 49, f"FAIL: Expected 49, got {defender}"
    print("SUCCESS: Root Pairwise selects best node ✓")

if __name__ == "__main__":
    test_transitivity_with_plot()
    test_root_defender_with_ground_truth()