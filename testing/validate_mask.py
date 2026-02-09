""" HIERARCHICAL MASK TEST """

import sys
sys.path.append('.')

import torch
from src.world_model import hierarchical_causal_mask

mask = hierarchical_causal_mask(8, torch.device('cpu'))
print("Hierarchical Causal Mask (8 positions = 2 timesteps):")
print(mask)

# Verify specific properties
# Test the hierarchical blocking
assert mask[5, 1] == float('-inf'), "L1_t1 should NOT see L1_t0 (hierarchical block)"
assert mask[5, 2] == float('-inf'), "L1_t1 should NOT see L2_t0 (hierarchical block)"
assert mask[6, 1] == float('-inf'), "L2_t1 should NOT see L1_t0 (hierarchical block)"
assert mask[6, 2] == float('-inf'), "L2_t1 should NOT see L2_t0 (hierarchical block)"

# Test that L0 from past is still visible
assert mask[5, 0] == 0, "L1_t1 SHOULD see L0_t0 (physics from past)"
assert mask[6, 0] == 0, "L2_t1 SHOULD see L0_t0 (physics from past)"

# Test within-timestep hierarchy
assert mask[5, 4] == 0, "L1_t1 SHOULD see L0_t1 (current physics)"
assert mask[6, 4] == 0, "L2_t1 SHOULD see L0_t1 (current physics)"
assert mask[6, 5] == 0, "L2_t1 SHOULD see L1_t1 (current mechanics)"

print("ALL hierarchical assertions passed")
