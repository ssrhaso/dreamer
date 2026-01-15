"""
Extract embeddings from trained multi-game CNN encoder
Saves to data/embeddings_{game}_cnn.npy for each game
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from encoder_v1 import AtariCNNEncoder


def extract_embeddings_multi(
    checkpoint_path: str = 'checkpoints/encoder_best.pt',
    games: list = ['ALE_Pong-v5', 'ALE_Breakout-v5', 'ALE_MsPacman-v5'],
    data_dir: str = 'data',
    batch_size: int = 256,
):
    """Extract embeddings for all frames across multiple games"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("MULTI-GAME EMBEDDING EXTRACTION")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}\n")
    
    # Load model once (shared across games)
    print("Loading encoder...")
    model = AtariCNNEncoder(input_channels=4, embedding_dim=384)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f})\n")
    
    # Process each game
    for game in games:
        print(f"GAME: {game}")
        print("-"*70)
        
        data_path = f"{data_dir}/replay_buffer_{game}.npz"
        output_path = f"{data_dir}/embeddings_{game}_cnn.npy"
        
        # Load data
        print("Loading replay buffer...")
        with np.load(data_path) as data:
            frames = data['states']
        print(f"  Loaded {len(frames):,} frames\n")
        
        # Extract embeddings
        print("Extracting embeddings...")
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(frames), batch_size)):
                batch = frames[i:i+batch_size].astype(np.float32) / 255.0
                batch_tensor = torch.from_numpy(batch).to(device)
                emb = model(batch_tensor).cpu().numpy()
                embeddings.append(emb)
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        print(f"\nExtracted embeddings: {embeddings.shape}")
        print(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
        print(f"  Memory: {embeddings.nbytes / 1e6:.1f} MB\n")
        
        # Save
        print(f"Saving to {output_path}...")
        np.save(output_path, embeddings)
        print(f"Saved successfully\n")


if __name__ == "__main__":
    extract_embeddings_multi()
