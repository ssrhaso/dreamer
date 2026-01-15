"""
Train Simple CNN Encoder with Contrastive Learning
Multi-game version using ConcatDataset (Colab-friendly).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import yaml

from encoder_v1 import AtariCNNEncoder



def load_config(config_path: str = "configs/encoder.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



class TemporalContrastiveNPZ(Dataset):
    """
    Temporal contrastive dataset that reads frames from a single .npz replay buffer.

    Positive pairs: (t, t+1)
    Negative pairs: random frame at least `min_negative_distance` away.
    """

    def __init__(
        self,
        path: str,
        temporal_distance: int = 1,
        min_negative_distance: int = 10,
    ):
        data = np.load(path)
        frames = data["states"]  # (N, 4, 84, 84)
        self.frames = frames
        self.temporal_distance = temporal_distance
        self.min_negative_distance = min_negative_distance
        self.n = len(frames) - temporal_distance

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Anchor and positive
        anchor = self.frames[idx]
        positive = self.frames[idx + self.temporal_distance]

        # Negative: far in time
        neg_idx = np.random.randint(0, len(self.frames))
        while abs(neg_idx - idx) < self.min_negative_distance:
            neg_idx = np.random.randint(0, len(self.frames))
        negative = self.frames[neg_idx]

        # To tensors in [0,1]
        anchor = torch.from_numpy(anchor.astype(np.float32) / 255.0)
        positive = torch.from_numpy(positive.astype(np.float32) / 255.0)
        negative = torch.from_numpy(negative.astype(np.float32) / 255.0)

        return anchor, positive, negative



def contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE on (anchor, positive, negative). Embeddings are assumed L2-normalized.
    """
    pos_sim = (anchor * positive).sum(dim=-1) / temperature
    neg_sim = (anchor * negative).sum(dim=-1) / temperature

    loss = -torch.log(
        torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))
    )
    return loss.mean()




def train_encoder(
    data_paths,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    temporal_distance: int,
    min_negative_distance: int,
    save_dir: str,
    save_every: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("-" * 70)
    print("TRAINING ATARI CNN ENCODER (MULTI-GAME)")
    print(f"\nDevice: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Build per-game datasets
    print(f"\nCreating per-game datasets from {len(data_paths)} replay buffers...")
    datasets = []
    total_samples = 0
    for i, path in enumerate(data_paths, 1):
        print(f"  [{i}/{len(data_paths)}] {path}")
        ds = TemporalContrastiveNPZ(
            path,
            temporal_distance=temporal_distance,
            min_negative_distance=min_negative_distance,
        )
        print(f"     → {len(ds):,} samples")
        datasets.append(ds)
        total_samples += len(ds)

    dataset = ConcatDataset(datasets)
    print(f"\nCombined dataset size: {len(dataset):,} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # safer for Colab
        pin_memory=False,
        drop_last=True,
    )

    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Model
    print("\nInitializing model...")
    model = AtariCNNEncoder(input_channels=4, embedding_dim=384).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.count_parameters() * 4 / 1024**2:.2f} MB (float32)")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Training loop
    print(f"\nSTARTING TRAINING ({num_epochs} epochs)")
    print("-" * 70)
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        for anchor, positive, negative in pbar:
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = contrastive_loss(
                emb_anchor, emb_positive, emb_negative, temperature=temperature
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            )

        avg_loss = epoch_loss / max(1, num_batches)
        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Time elapsed: {elapsed/60:.1f} min")
        print(
            f"  Est. remaining: {(elapsed/(epoch+1))*(num_epochs-epoch-1)/60:.1f} min"
        )

        scheduler.step()

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = Path(save_dir) / "encoder_best.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                best_path,
            )
            print(f"   Saved best model (loss: {best_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = Path(save_dir) / f"encoder_epoch{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            print(f"   Saved checkpoint: {ckpt_path}")

        print()

    final_path = Path(save_dir) / "encoder_final.pt"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        final_path,
    )

    total_time = time.time() - start_time
    print("TRAINING COMPLETE")
    print("-" * 70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print("\nSaved models:")
    print(f"   Best:  {Path(save_dir) / 'encoder_best.pt'}")
    print(f"   Final: {final_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return model



if __name__ == "__main__":
    config = load_config("configs/encoder.yaml")

    data_paths = config["data"]["replay_buffers"]
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    temperature = config["contrastive"]["temperature"]
    temporal_distance = config["data"]["temporal_distance"]
    min_negative_distance = config["contrastive"]["min_negative_distance"]
    save_dir = config["logging"]["save_dir"]
    save_every = config["logging"]["save_every"]

    model = train_encoder(
        data_paths=data_paths,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        temporal_distance=temporal_distance,
        min_negative_distance=min_negative_distance,
        save_dir=save_dir,
        save_every=save_every,
    )

    print("\nTRAINING FINISHED, MODEL READY FOR USE.")
