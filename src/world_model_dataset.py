""" DATASET CLASS FOR WORLD MODEL TRAINING

1. RESHAPES
2. RESPECT EPISODE BOUNDARIES
3. MULTI-GAME MIXING
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
import yaml

class WorldModelDataset(Dataset):
    def __init__(
        self,
        games : List[str],
        tokens_dir : str = "checkpoints/rsvq_tokens",
        replay_dir : str = "data",
        seq_len : int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        
        self.all_tokens = []        # (N, 3) arrays, 1x per game
        self.all_actions = []       # (N,) arrays, 1x per game
        self.valid_starts = []      # (game_idx, start_idx) tuples 
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        pass
    

def create_dataloders():
    pass

if __name__ == "__main__":
    pass