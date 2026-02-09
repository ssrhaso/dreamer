""" HIERARCHICAL WORLD MODEL TEST """

import sys
sys.path.append('.')

import torch
from src.world_model import WorldModelConfig, HierarchicalWorldModel

config = WorldModelConfig() 
model = HierarchicalWorldModel(config)
device = torch.device('cpu')
B, T = 2, 4

tokens = torch.randint(0, config.num_codes, (B, T, 3))
actions = torch.randint(0, config.num_actions, (B, T))
logits_l0, logits_l1, logits_l2 = model(tokens, actions)

expected = (B, T, config.num_codes)  # (2, 4, 256)

# TEST A: Output shapes
assert logits_l0.shape == expected, f"FAIL L0: {logits_l0.shape} != {expected}"
assert logits_l1.shape == expected, f"FAIL L1: {logits_l1.shape} != {expected}"
assert logits_l2.shape == expected, f"FAIL L2: {logits_l2.shape} != {expected}"
print(f" Output shapes: L0={logits_l0.shape}, L1={logits_l1.shape}, L2={logits_l2.shape}")

# TEST B: No NaN/Inf
assert not torch.isnan(logits_l0).any(), "FAIL: NaN in L0 logits"
assert not torch.isinf(logits_l0).any(), "FAIL: Inf in L0 logits"
print(f" No NaN/Inf in outputs")

# TEST C: Gradients flow end-to-end
model.zero_grad()
logits_l0, logits_l1, logits_l2 = model(tokens, actions)
dummy_loss = logits_l0.sum() + logits_l1.sum() + logits_l2.sum()
dummy_loss.backward()

assert model.embedding.token_embeds[0].weight.grad is not None, "FAIL: no grad to embedding"
assert list(model.blocks[0].parameters())[0].grad is not None, "FAIL: no grad to blocks"
assert model.headl0.weight.grad is not None, "FAIL: no grad to output heads"
print(f" Gradients flow: embedding -> blocks -> heads")

# TEST D: Parameter count
total = sum(p.numel() for p in model.parameters())
print(f" Total parameters: {total:,} ({total/1e6:.2f}M)")
print("\nALL HIERARCHICAL WORLD MODEL TESTS PASSED")
