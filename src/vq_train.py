""" TRAIN VQ TOKENIZER ON DINOv2 EMBEDDINGS """
""" NOTE: BASELINE CODE, TO BE MODIFIED """

import os 
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from vq import VQVAE, VQTokenizer, HRVQTokenizer, load_config



class EmbeddingDataset(Dataset):
    """ DATASET WRAPPER FOR DINOv2 EMBEDDINGS
    INPUT: .npy FILE CONTAINING PRECOMPUTED EMBEDDINGS
    """
    
    # CONSTRUCTOR
    def __init__(self, embeddings_paths : list,):
        all_embeddings = []
        
        for path in embeddings_paths:
            emb = np.load(path).astype(np.float32)
            print(f"LOADED {path}, SHAPE: {emb.shape}")
            all_embeddings.append(emb)
            
        self.embeddings = np.concatenate(all_embeddings, axis=0)
        # L2 normalize to unit length
        
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings = self.embeddings / norms
        print(f"LOADED EMBEDDINGS FROM {embeddings_paths}, LENGTH: {len(self.embeddings)}")
    
    # SAMPLING METHODS
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx])
    
def compute_codebook_stats(
    tokens : torch.Tensor,
    num_codes : int = 256,
):
    """ COMPUTE CODEBOOK USAGE STATISTICS (TOKEN FREQUENCY) """
    
    # FLATTEN TOKENS (HANDLES SPATIAL DIMENSIONS)
    tokens = tokens.flatten()
    
    # UNIQUE CODES
    unique_codes = torch.unique(tokens)
    num_used = len(unique_codes)
    
    # FREQUENCY COUNTS
    token_counts = torch.bincount(tokens, minlength = num_codes).float()
    token_probs = token_counts / token_counts.sum()
    
    # PERPLEXITY (EFFECTIVE NUMBER OF CODES USED)
    token_probs_nonzero = token_probs[token_probs > 0]
    entropy = -torch.sum(token_probs_nonzero * torch.log(token_probs_nonzero))
    perplexity = torch.exp(entropy).item()
    
    # HISTOGRAM 
    usage_histogram = token_counts.cpu().numpy().tolist()
    
    return {
        'num_used_codes': int(num_used),
        'total_codes': num_codes,
        'usage_ratio': float(num_used / num_codes),
        'perplexity': float(perplexity),
        'usage_histogram': usage_histogram,
    }


def compute_hierarchical_codebook_stats(
    all_tokens : list, 
    num_codes : int = 256,
):
    """ COMPUTE CODEBOOK STATS FOR HIERARCHICAL TOKENS """
    layer_stats = []  # List, not dict!
    for layer_idx, tokens in enumerate(all_tokens):
        
        # FLATTEN
        tokens = tokens.flatten()
        
        # UNIQUE CODES
        unique_codes = torch.unique(tokens)
        num_used = len(unique_codes)
        
        # FREQUENCY COUNTS
        token_counts = torch.bincount(tokens, minlength = num_codes).float()
        token_probs = token_counts / token_counts.sum()
        
        # PERPLEXITY (EFFECTIVE NUMBER OF CODES USED)
        token_probs_nonzero = token_probs[token_probs > 0]
        entropy = -torch.sum(token_probs_nonzero * torch.log(token_probs_nonzero))
        perplexity = torch.exp(entropy).item()
        
        # STATS
        layer_stats.append({
            'layer' : layer_idx,
            'num_used_codes': int(num_used),
            'total_codes': num_codes,
            'usage_ratio': float(num_used / num_codes),
            'perplexity': float(perplexity),
        })
        
    return layer_stats
    
def train_vq(
    embeddings_paths : list  = None,
    output_dir : str = "checkpoints",
    num_codes : int = 256,
    latent_dim : int = 128,
    commitment_cost : float = 0.25,
    batch_size : int = 256,
    num_epochs : int = 50,
    learning_rate : float = 1.0e-3,
    val_split : float = 0.05,
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed : int = 42,
):
    """ TRAIN VQ TOKENIZER ON DINOv2 EMBEDDINGS """
    
    # SET SEED FOR REPRODUCIBILITY
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # OUTPUT DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    print(f"SAVING CHECKPOINTS TO: {output_dir}")
    print(f"TRAINING VQ TOKENIZER WITH {num_codes} CODES)")
    print(f" DEVICE    : {device}"
          f"\n BATCH SIZE: {batch_size}"
          f"\n EPOCHS    : {num_epochs}"
          f"\n LR        : {learning_rate}"
          f"\n VAL SPLIT : {val_split}")
    
    # LOAD DATASET
    dataset = EmbeddingDataset(embeddings_paths)
    
    # TRAIN-VAL SPLIT
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator = torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        dataset= train_dataset, 
        batch_size=batch_size, 
        num_workers= 2, 
        pin_memory= (device == 'cuda'),)
    
    val_loader = DataLoader(
        dataset= val_dataset, 
        batch_size=batch_size, 
        num_workers= 2,
        pin_memory= (device == 'cuda'),)
    
    print(f"TRAINING SAMPLES : {len(train_dataset)} | VALIDATION SAMPLES: {len(val_dataset)}")
    
    # INIT MODEL
    
    model = HRVQTokenizer(
        input_dim = 384,
        num_codes_per_layer = num_codes,
        num_layers = 3,
        # commitment_costs=[0.15, 0.25, 0.40],  # Based on MAGVIT/SoundStream
        commitment_costs=[0.05, 0.25, 0.60],  # Aggressive coarse→fine for hierarchical learning
        decay = 0.99,
        epsilon = 1e-5,
    ).to(device)
    print(f"HIERARCHICAL VQ : {model.num_layers} LAYERS, {num_codes} CODES PER LAYER")
    print(f"TOTAL CAPACITY: {model.total_capacity} BITS")
    
    # ARCHIVE SINGLE-LAYER BASELINE
    # model = VQTokenizer(
    #     input_dim = 384,
    #     num_codes = num_codes,
    #     commitment_cost = commitment_cost,
    # ).to(device)
    
    # INITIALIZE ALL LAYER CODEBOOKS FROM DATA SAMPLES
    print(f"INITIALIZING {model.num_layers} CODEBOOKS FROM DATA...")
    with torch.no_grad():
        init_batch = next(iter(train_loader)).to(device)
        init_batch = init_batch.unsqueeze(1).unsqueeze(2)
        init_samples = init_batch.reshape(-1, 384)[:num_codes]
        
        # Repeat if needed
        if init_samples.size(0) < num_codes:
            repeats = (num_codes + init_samples.size(0) - 1) // init_samples.size(0)
            init_samples = init_samples.repeat(repeats, 1)[:num_codes]
        
        # Initialize all layers with the same data (they'll diverge during training)
        for layer_idx, vq_layer in enumerate(model.vq_layers):
            vq_layer.codebook.weight.data.copy_(init_samples)
            vq_layer.ema_weight.copy_(init_samples)
    print(f"ALL {model.num_layers} CODEBOOKS INITIALIZED")
    
    # NO OPTIMIZER NEEDED - EMA updates codebook without gradients
    print(f"NOTE: Using EMA updates (no gradient-based training)")
    
    """ TRAINING LOOP"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        train_tokens_all = []
        
        pbar = tqdm(
            train_loader,
            desc = f"Epoch {epoch+1}/{num_epochs} - Training",
            leave = True
        )
        
        for batch in pbar:
            batch = batch.to(device)
            # Add spatial dimensions: [B, 384] -> [B, 1, 1, 384]
            batch = batch.unsqueeze(1).unsqueeze(2)
            
            # Forward pass (EMA updates codebook automatically)
            z_quantized, loss, all_tokens = model(batch)
            
            train_loss += loss.item()
            train_tokens_all.append([tok.cpu().detach() for tok in all_tokens])
            
            # UPDATE PROGRESS BAR
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        """ VALIDATION LOOP """
        model.eval()
        val_loss = 0.0
        val_tokens_all = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # Add spatial dimensions: [B, 384] -> [B, 1, 1, 384]
                batch = batch.unsqueeze(1).unsqueeze(2)
                z_quantized, loss, all_tokens = model(batch)
                val_loss += loss.item()
                val_tokens_all.append([tok.cpu().detach() for tok in all_tokens])
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        """ CODEBOOK ANALYSIS (PER LAYER) """
        # Reorganize: list of batches -> list of layers
        train_tokens_by_layer = [
            torch.cat([batch[i] for batch in train_tokens_all], dim=0)
            for i in range(model.num_layers)
        ]
        val_tokens_by_layer = [
            torch.cat([batch[i] for batch in val_tokens_all], dim=0)
            for i in range(model.num_layers)
        ]
        
        train_stats = compute_hierarchical_codebook_stats(
            train_tokens_by_layer, num_codes=num_codes
        )
        val_stats = compute_hierarchical_codebook_stats(
            val_tokens_by_layer, num_codes=num_codes
        )
        
        print(f"Epoch {epoch+1}/{num_epochs} Summary:"
              f"\n TRAIN LOSS: {avg_train_loss:.4f} | VAL LOSS: {avg_val_loss:.4f}")
        
        for layer_idx in range(model.num_layers):
            print(f" Layer {layer_idx} TRAIN: {train_stats[layer_idx]['num_used_codes']}/{num_codes} "
                  f"({train_stats[layer_idx]['usage_ratio']*100:.2f}%), "
                  f"Perplexity: {train_stats[layer_idx]['perplexity']:.2f}")
        for layer_idx in range(model.num_layers):
            print(f" Layer {layer_idx} VAL  : {val_stats[layer_idx]['num_used_codes']}/{num_codes} "
                  f"({val_stats[layer_idx]['usage_ratio']*100:.2f}%), "
                  f"Perplexity: {val_stats[layer_idx]['perplexity']:.2f}")
        
        """ SAVE CHECKPOINT IF BEST """
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'vq_model_best.pth'))
            print(f" Saved Best Model Checkpoint")
        
    # FINAL TOKENIZATION
    print(f"GENERATING FINAL TOKENS FOR FULL DATASET...")
    
    model.eval()
    all_tokens = []
    full_loader = DataLoader(
        dataset= dataset, 
        batch_size=batch_size, 
        num_workers= 2,
        pin_memory= (device == 'cuda'),)
    
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Tokenizing Full Dataset"):
            batch = batch.to(device)
            # Add spatial dimensions: [B, 384] -> [B, 1, 1, 384]
            batch = batch.unsqueeze(1).unsqueeze(2)
            all_tokens_batch = model.encode(batch)  # Returns list of 3 token arrays
            all_tokens.append([tok.cpu() for tok in all_tokens_batch])
    
    # Reorganize: list of batches -> list of layers
    all_tokens = [
        torch.cat([batch[i] for batch in all_tokens], dim=0).numpy()
        for i in range(model.num_layers)
    ]
    
    """ FINAL STATISTICS """
    final_stats = compute_hierarchical_codebook_stats(
        [torch.from_numpy(layer_tokens) for layer_tokens in all_tokens], num_codes = num_codes
    )
    
    with torch.no_grad():
        sample_batch = next(iter(full_loader)).to(device)
        # Add spatial dimensions: [B, 384] -> [B, 1, 1, 384]
        sample_batch = sample_batch.unsqueeze(1).unsqueeze(2)
        z_quantized, _, _ = model(sample_batch)
        quantized_stats = {
            'mean': float(z_quantized.mean().item()),
            'std': float(z_quantized.std().item()),
            'min': float(z_quantized.min().item()),
            'max': float(z_quantized.max().item()),
        }
        
    # HEALTH CHECK
    for layer_idx in range(model.num_layers):
        print(f"FINAL CODEBOOK USAGE LAYER {layer_idx}: {final_stats[layer_idx]['num_used_codes']}/{num_codes} "
              f"({final_stats[layer_idx]['usage_ratio']*100:.2f}%), PERPLEXITY: {final_stats[layer_idx]['perplexity']:.2f}"
        )
    print(f"QUANTIZED LATENT STATS: MEAN={quantized_stats['mean']:.4f}, "
          f"STD={quantized_stats['std']:.4f}, MIN={quantized_stats['min']:.4f}, MAX={quantized_stats['max']:.4f}"
    )
    
    # COLLAPSE WARNING (check any layer)
    collapsed_layers = [i for i in range(model.num_layers) 
                       if final_stats[i]['num_used_codes'] < num_codes * 0.1]
    if collapsed_layers:
        print(f"WARNING: CODEBOOK COLLAPSE DETECTED IN LAYERS {collapsed_layers}!")
    else:
        print("CODEBOOK USAGE HEALTHY ACROSS ALL LAYERS.")
    
    """ SAVE FINAL TOKENS AND STATS """
    print(f"SAVING FINAL TOKENS AND TRAINING STATS...")
    games = ['ALE_Pong-v5', 'ALE_Breakout-v5', 'ALE_MsPacman-v5']
    
    for i, (game, emb_path) in enumerate(zip(games, embeddings_paths)):
        game_embeddings = np.load(emb_path).astype(np.float32)
        norms = np.linalg.norm(game_embeddings, axis=1, keepdims=True) + 1e-8
        game_embeddings = game_embeddings / norms
        
        game_loader = DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(game_embeddings)),
            batch_size=batch_size,
            num_workers=2,
        )
        
        game_tokens = []
        with torch.no_grad():
            for (batch,) in tqdm(game_loader, desc=f"Tokenizing {game}"):
                batch = batch.to(device).unsqueeze(1).unsqueeze(2)
                all_tokens_batch = model.encode(batch)  # List of 3 token arrays
                game_tokens.append([tok.cpu() for tok in all_tokens_batch])
        
        # Reorganize: list of batches -> list of layers
        game_tokens_by_layer = [
            torch.cat([batch[i] for batch in game_tokens], dim=0).numpy()
            for i in range(model.num_layers)
        ]
        
        # Save tokens for each layer separately
        for layer_idx, layer_tokens in enumerate(game_tokens_by_layer):
            tokens_path = os.path.join(output_dir, f'vq_tokens_{game}_layer{layer_idx}.npy')
            np.save(tokens_path, layer_tokens)
            print(f" Saved Layer {layer_idx} tokens for {game}: {tokens_path} ({len(layer_tokens)} tokens)")
    
    model_path = os.path.join(output_dir, 'vq_model_final.pth')
    
    stats = {
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses],
        'final_codebook_stats': final_stats,
        'quantized_latent_stats': quantized_stats,
        'hyperparameters': {
            'num_codes': num_codes,
            'latent_dim': latent_dim,
            'commitment_cost': commitment_cost,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'val_split': val_split,
            'seed': seed,
        }
    }
    
    model_path = os.path.join(output_dir, 'vq_model_final.pth')
    stats_path = os.path.join(output_dir, 'vq_stats.json')
    
    # Save stats
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"TRAINING COMPLETE.")
    print(f" Model: {model_path}")
    print(f" Stats: {stats_path}")
    print(f" Tokens saved separately for each game in: {output_dir}")
    
    return model, all_tokens, stats


""" ENTRY """
if __name__ == "__main__":
    
    config = load_config("configs/vq.yaml")
    
    train_vq(
        embeddings_paths = config['data']['embeddings_paths'],  
        output_dir = config['training'].get('save_dir', 'checkpoints'),
        num_codes = config['model']['num_codes'],
        latent_dim = config['model']['latent_dim'],
        commitment_cost = config['model']['commitment_cost'],
        batch_size = config['training']['batch_size'],
        num_epochs = config['training']['num_epochs'],
        learning_rate = config['training']['learning_rate'],
        val_split = config['training']['val_split'],
        seed = config['seed'],
    )


