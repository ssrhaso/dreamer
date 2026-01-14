"""
Complete Encoder Validation Suite
- Game state correlation (like DINOv2 analysis)
- Temporal consistency
- Feature statistics
- Pass/fail criteria
"""

import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from tqdm import tqdm

from encoder_v1 import AtariCNNEncoder


def extract_game_states(frames: np.ndarray, num_samples: int = 1000):
    """
    Extract ground-truth game state features from Pong frames
    Same method used for DINOv2 analysis
    """
    states = {
        'paddle_y': [],
        'opponent_y': [],
        'ball_x': [],
        'ball_y': [],
    }
    
    indices = np.random.choice(len(frames), num_samples, replace=False)
    
    for idx in tqdm(indices, desc="Extracting game states"):
        frame = frames[idx, -1, :, :]  # Last frame (84, 84)
        
        # Player paddle (bottom region, rows 60-80)
        player_region = frame[60:80, :]
        if player_region.max() > 100:
            bright_cols = np.where(player_region.mean(axis=0) > 100)[0]
            states['paddle_y'].append(bright_cols.mean() if len(bright_cols) > 0 else -1)
        else:
            states['paddle_y'].append(-1)
        
        # Opponent paddle (top region, rows 4-24)
        opponent_region = frame[4:24, :]
        if opponent_region.max() > 100:
            bright_cols = np.where(opponent_region.mean(axis=0) > 100)[0]
            states['opponent_y'].append(bright_cols.mean() if len(bright_cols) > 0 else -1)
        else:
            states['opponent_y'].append(-1)
        
        # Ball (center region, rows 20-64)
        ball_region = frame[20:64, :]
        if ball_region.max() > 150:
            ball_pixels = np.where(ball_region > 150)
            if len(ball_pixels[0]) > 0:
                states['ball_y'].append(ball_pixels[0].mean() + 20)
                states['ball_x'].append(ball_pixels[1].mean())
            else:
                states['ball_y'].append(-1)
                states['ball_x'].append(-1)
        else:
            states['ball_y'].append(-1)
            states['ball_x'].append(-1)
    
    for key in states:
        states[key] = np.array(states[key])
    
    return states, indices


def compute_correlations(embeddings: np.ndarray, states: dict):
    """Compute correlation between embeddings and game states"""
    correlations = {}
    
    for state_name, state_values in states.items():
        valid_mask = state_values > 0
        
        if valid_mask.sum() < 10:
            correlations[state_name] = 0.0
            continue
        
        valid_states = state_values[valid_mask]
        valid_embeddings = embeddings[valid_mask]
        
        # Max correlation across all embedding dimensions
        dim_correlations = []
        for dim in range(valid_embeddings.shape[1]):
            corr, _ = pearsonr(valid_embeddings[:, dim], valid_states)
            dim_correlations.append(abs(corr))
        
        correlations[state_name] = max(dim_correlations)
    
    return correlations


def validate_encoder(
    checkpoint_path: str = 'checkpoints/encoder_best.pt',
    data_path: str = 'data/replay_buffer_ALE_Pong-v5.npz',
    batch_size: int = 256,
):
    """Complete encoder validation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("CNN ENCODER VALIDATION")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading encoder...")
    model = AtariCNNEncoder(input_channels=4, embedding_dim=384)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (training loss: {checkpoint['loss']:.4f})")
    
    # Load data
    print("\nLoading replay buffer...")
    with np.load(data_path) as data:
        frames = data['states']
    print(f"✓ Loaded {len(frames):,} frames\n")
    
    # Extract ALL embeddings
    print("Extracting embeddings for all frames...")
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size)):
            batch = frames[i:i+batch_size].astype(np.float32) / 255.0
            batch_tensor = torch.from_numpy(batch).to(device)
            emb = model(batch_tensor).cpu().numpy()
            embeddings.append(emb)
    
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"✓ Extracted: {embeddings.shape}\n")
    
    # === TEST 1: L2 Normalization ===
    print("=" * 70)
    print("TEST 1: L2 NORMALIZATION")
    print("=" * 70)
    
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = norms.mean()
    std_norm = norms.std()
    
    print(f"Mean norm: {mean_norm:.4f} (should be ~1.0)")
    print(f"Std norm:  {std_norm:.4f} (should be ~0.0)")
    
    norm_pass = 0.99 < mean_norm < 1.01
    print(f"{'✓ PASS' if norm_pass else '✗ FAIL'}\n")
    
    # === TEST 2: Embedding Diversity ===
    print("=" * 70)
    print("TEST 2: EMBEDDING DIVERSITY")
    print("=" * 70)
    
    print("Computing pairwise distances (sample 1000)...")
    sample = embeddings[np.random.choice(len(embeddings), 1000, replace=False)]
    distances = pdist(sample, metric='euclidean')
    mean_dist = distances.mean()
    
    print(f"Mean pairwise distance: {mean_dist:.3f}")
    
    if mean_dist < 0.3:
        print("✗ FAIL - Embeddings COLLAPSED")
        print("   Retrain with temperature=0.5")
        diversity_pass = False
    elif mean_dist > 2.0:
        print("✗ FAIL - Embeddings too random")
        diversity_pass = False
    else:
        print(f"✓ PASS - Good diversity")
        diversity_pass = True
    print()
    
    # === TEST 3: Temporal Consistency ===
    print("=" * 70)
    print("TEST 3: TEMPORAL CONSISTENCY")
    print("=" * 70)
    
    # Consecutive vs random distances
    consecutive_dists = []
    for i in range(0, len(embeddings)-1, 100):  # Sample every 100th
        dist = np.linalg.norm(embeddings[i] - embeddings[i+1])
        consecutive_dists.append(dist)
    
    random_indices = np.random.choice(len(embeddings), (1000, 2), replace=False)
    random_dists = []
    for i, j in random_indices:
        dist = np.linalg.norm(embeddings[i] - embeddings[j])
        random_dists.append(dist)
    
    consecutive_mean = np.mean(consecutive_dists)
    random_mean = np.mean(random_dists)
    temporal_ratio = consecutive_mean / random_mean
    
    print(f"Consecutive frames: {consecutive_mean:.3f}")
    print(f"Random frames:      {random_mean:.3f}")
    print(f"Temporal ratio:     {temporal_ratio:.3f}")
    
    if temporal_ratio < 0.5:
        print("✓ EXCELLENT - Strong temporal structure")
        temporal_pass = True
    elif temporal_ratio < 0.7:
        print("✓ GOOD - Clear temporal structure")
        temporal_pass = True
    else:
        print("⚠ WARNING - Weak temporal structure")
        temporal_pass = False
    print()
    
    # === TEST 4: Feature Variance ===
    print("=" * 70)
    print("TEST 4: FEATURE VARIANCE")
    print("=" * 70)
    
    feature_var = embeddings.var(axis=0)
    mean_var = feature_var.mean()
    dead_features = (feature_var < 1e-6).sum()
    
    print(f"Mean variance:   {mean_var:.6f}")
    print(f"Dead features:   {dead_features}/384")
    
    variance_pass = mean_var > 0.01 and dead_features < 10
    print(f"{'✓ PASS' if variance_pass else '✗ FAIL'}\n")
    
    # === TEST 5: Game State Correlation ===
    print("=" * 70)
    print("TEST 5: GAME STATE CORRELATION")
    print("=" * 70)
    
    print("Extracting game states from frames...")
    states, state_indices = extract_game_states(frames, num_samples=1000)
    
    print("Computing correlations...")
    state_embeddings = embeddings[state_indices]
    correlations = compute_correlations(state_embeddings, states)
    
    print(f"\nCorrelations with game features:")
    for state_name, corr in correlations.items():
        print(f"  {state_name:15s}: {corr:.4f}")
    
    overall_corr = np.mean([c for c in correlations.values() if c > 0])
    print(f"\n  Overall: {overall_corr:.4f}")
    
    if overall_corr > 0.75:
        print("✓ EXCELLENT - 10x better than DINOv2 (0.08)")
        corr_pass = True
    elif overall_corr > 0.65:
        print("✓ GOOD - Still much better than DINOv2")
        corr_pass = True
    else:
        print("⚠ MODERATE - Better than DINOv2 but could improve")
        corr_pass = False
    print()
    
    # === FINAL SUMMARY ===
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Test':<35s} {'Status':<10s} {'Value'}")
    print("-" * 70)
    print(f"{'1. L2 Normalization':<35s} {'✓ PASS' if norm_pass else '✗ FAIL':<10s} {mean_norm:.3f}")
    print(f"{'2. Embedding Diversity':<35s} {'✓ PASS' if diversity_pass else '✗ FAIL':<10s} {mean_dist:.3f}")
    print(f"{'3. Temporal Consistency':<35s} {'✓ PASS' if temporal_pass else '✗ FAIL':<10s} {temporal_ratio:.3f}")
    print(f"{'4. Feature Variance':<35s} {'✓ PASS' if variance_pass else '✗ FAIL':<10s} {mean_var:.4f}")
    print(f"{'5. Game State Correlation':<35s} {'✓ PASS' if corr_pass else '✗ FAIL':<10s} {overall_corr:.3f}")
    
    all_pass = norm_pass and diversity_pass and temporal_pass and variance_pass and corr_pass
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✓✓✓ ALL TESTS PASSED - ENCODER READY FOR VQ TRAINING")
        print("\nExpected VQ performance:")
        print("  Codebook usage: 180-220/256 (70-85%)")
        print("  Perplexity: 100-150")
        print("\nNext steps:")
        print("  1. Extract embeddings: python src/extract_embeddings.py")
        print("  2. Train VQ: python src/vq_train.py")
    else:
        print("⚠ SOME TESTS FAILED")
        if not diversity_pass:
            print("\n→ Embeddings collapsed. Retrain with temperature=0.5")
        if not corr_pass:
            print("\n→ Low correlation. Consider training longer or wider model")
        if not temporal_pass:
            print("\n→ Weak temporal structure. Check contrastive loss")
    
    print("=" * 70 + "\n")
    
    return {
        'overall_correlation': overall_corr,
        'temporal_ratio': temporal_ratio,
        'mean_distance': mean_dist,
        'mean_variance': mean_var,
        'all_pass': all_pass,
    }


if __name__ == "__main__":
    results = validate_encoder()
