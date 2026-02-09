""" TRANSFORMER BLOCK TEST """

import sys
sys.path.append('.')

import torch
from src.world_model import WorldModelConfig, TokenEmbedding, TransformerBlock, hierarchical_causal_mask

device = torch.device('cpu')
config = WorldModelConfig()  # defaults

# Create fake input: batch=2, timesteps=4
B, T = 2, 4
tokens = torch.randint(0, config.num_codes, (B, T, 3))   # Random HRVQ tokens
actions = torch.randint(0, config.num_actions, (B, T))    # Random actions

# Step 1: Embed (using existing TokenEmbedding)
embed = TokenEmbedding(config)
seq = embed(tokens, actions)  # (2, 16, 384)
print(f"TokenEmbedding output: {seq.shape}")

# Step 2: Create mask
mask = hierarchical_causal_mask(T * 4, device)  # (16, 16)
print(f"Mask shape: {mask.shape}")

# Step 3: Pass through ONE TransformerBlock
block = TransformerBlock(config)
out = block(seq, mask)

# TEST A: Shape preserved (must be identical in/out)
assert out.shape == seq.shape, f"FAIL: shape {out.shape} != {seq.shape}"
print(f" Shape preserved: {seq.shape} → {out.shape}")

# TEST B: Output is different from input (block actually transformed something)
assert not torch.allclose(out, seq, atol=1e-6), "FAIL: output identical to input"
print(f" Output differs from input (block did work)")

# TEST C: No NaN or Inf in output (mask didn't break anything)
assert not torch.isnan(out).any(), "FAIL: NaN in output"
assert not torch.isinf(out).any(), "FAIL: Inf in output"
print(f" No NaN/Inf in output")

# TEST D: Count parameters
num_params = sum(p.numel() for p in block.parameters())
print(f" Parameters per block: {num_params:,}")
print(f"  ( X {config.n_layers} blocks = {num_params * config.n_layers:,} total for transformer)")

# TEST E: Gradients flow through (can we train this?)
loss = out.sum()
loss.backward()
has_grads = all(p.grad is not None for p in block.parameters())
assert has_grads, "FAIL: some parameters have no gradients"
print(f" Gradients flow to all parameters")
print("\nALL TRANSFORMER BLOCK TESTS PASSED")
