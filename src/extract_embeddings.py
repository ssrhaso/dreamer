"""
Extract embeddings from trained CNN encoder
Saves to data/embeddings_ALE_Pong-v5_cnn.npy
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from encoder_v1 import AtariCNNEncoder


def extract_embeddings(
    checkpoint_path: str = 'checkpoints/encoder_best.pt',
    data_path: str = 'data/replay_buffer_ALE_Pong-v5.npz',
    output_path: str = 'data/embeddings_ALE_Pong-v5_cnn.npy',
    batch_size: int = 256,
):
    """Extract embeddings for all frames"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    print("EXTRACTING EMBEDDINGS")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading encoder...")
    model = AtariCNNEncoder(input_channels=4, embedding_dim=384)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f" Model loaded (training loss: {checkpoint['loss']:.4f})\n")
    
    # Load data
    print("Loading replay buffer...")
    with np.load(data_path) as data:
        frames = data['states']
    print(f" Loaded {len(frames):,} frames\n")
    
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
    
    print(f"\n Extracted embeddings: {embeddings.shape}")
    print(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Memory: {embeddings.nbytes / 1e6:.1f} MB\n")
    
    # Save
    print(f"Saving to {output_path}...")
    np.save(output_path, embeddings)
    
    print(f" Saved successfully\n")
    
    return embeddings


if __name__ == "__main__":
    extract_embeddings()
