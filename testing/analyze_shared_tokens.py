"""
Analyze what the 33 shared tokens actually represent
Are they capturing meaningful semantics or just edge cases?
"""

import numpy as np
from collections import Counter

# Load Layer 0 tokens for all games
pong_tok = np.load('checkpoints/rsvq_tokens/vq_tokens_ALE_Pong-v5_layer0.npy').flatten()
break_tok = np.load('checkpoints/rsvq_tokens/vq_tokens_ALE_Breakout-v5_layer0.npy').flatten()
pac_tok = np.load('checkpoints/rsvq_tokens/vq_tokens_ALE_MsPacman-v5_layer0.npy').flatten()

# Find shared tokens
shared = set(pong_tok) & set(break_tok) & set(pac_tok)
print(f"Shared tokens: {len(shared)}/256")
print(f"Token IDs: {sorted(shared)[:30]}...\n")

# Check frequency - are they common or rare?
for game_name, tokens in [('Pong', pong_tok), ('Breakout', break_tok), ('MsPacman', pac_tok)]:
    shared_usage = sum(1 for t in tokens if t in shared) / len(tokens)
    
    # Get frequency distribution of shared vs non-shared
    all_freq = Counter(tokens)
    shared_freq = [all_freq[t] for t in shared]
    non_shared = set(all_freq.keys()) - shared
    non_shared_freq = [all_freq[t] for t in non_shared]
    
    print(f"{game_name}:")
    print(f"  Shared token usage: {shared_usage*100:.2f}% of frames")
    print(f"  Avg freq (shared): {np.mean(shared_freq):.1f} ± {np.std(shared_freq):.1f}")
    print(f"  Avg freq (non-shared): {np.mean(non_shared_freq):.1f} ± {np.std(non_shared_freq):.1f}")
    
    # Are shared tokens rare or common?
    if shared_usage < 0.05:
        print(f"   Shared tokens are RARE - likely noise/menus, not semantics")
    elif shared_usage < 0.15:
        print(f"   Shared tokens are UNCOMMON - minimal cross-game benefit")
    else:
        print(f"   Shared tokens are COMMON - good cross-game semantics")
    print()

# Check if shared tokens are concentrated in specific regions
print("\n Temporal Distribution ")
for game_name, tokens in [('Pong', pong_tok[:1000]), ('Breakout', break_tok[:1000]), ('MsPacman', pac_tok[:1000])]:
    # First 1000 frames (likely to contain menus/start screens)
    early_shared = sum(1 for t in tokens if t in shared) / len(tokens)
    print(f"{game_name} (first 1000 frames): {early_shared*100:.1f}% shared")
    
# Recommendation
print("\n CONCLUSION ")
avg_shared_usage = np.mean([
    sum(1 for t in pong_tok if t in shared) / len(pong_tok),
    sum(1 for t in break_tok if t in shared) / len(break_tok),
    sum(1 for t in pac_tok if t in shared) / len(pac_tok)
])

if avg_shared_usage < 0.10:
    print(" Cross-game transfer is NOT working - shared tokens represent <10% of frames")
    print("   Recommendation: Drop multi-game requirement OR use more similar games")
elif avg_shared_usage < 0.20:
    print(" Weak cross-game transfer - shared tokens represent 10-20% of frames")
    print("   Recommendation: Consider more visually similar games or accept limitation")
else:
    print(" Strong cross-game transfer - shared tokens represent >20% of frames")
    print("   Proceed with world model training")
