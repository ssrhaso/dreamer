""" SKELETON CODE FOR WORLD MODEL MODULE 

- WIP 1 : BASELINE IMPLEMENTATION - STORM(2023) INSPIRED
- WIP 2 : TWISTER(2025) / DREAMERv4(2025) INSPIRED IMPROVEMENTS

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import yaml
from pathlib import Path

@dataclass 
class WorldModelConfig:
    """ CONFIG MATCHING configs/worldmodel.yaml """
    
    d_model : int  = 384            # EMBEDDING DIMENSION (WIDTH OF NN)             - HRVQ EMBEDDING DIMENSION
    n_layers : int = 6              # NUMBER OF TRANSFORMER BLOCKS (DEPTH OF NN)    - (Kaplan et al. 2020)
    n_heads : int = 6               # NUMBER OF ATTENTION HEADS                     - (Vaswani et al. 2017) - standard rule of = d_model / 64
    d_ff : int    = 1536            # DIMENSION OF FEEDFORWARD NETWORK              - (Vaswani et al. 2017) - standard rule of = 4 * d_model
    dropout : float = 0.1           # DROPOUT RATE                                  - (Devlin et al. 2017)  - BERT, GPT, STORM use 0.1       
    max_seq_len : int = 256         # MAXIMUM SEQUENCE LENGTH                       - ~ 65k positions of memory, Fits T4 GPU safely
    num_codes : int = 256           # NUMBER OF CODEBOOK ENTRIES                    - HRVQ Codebook Size
    num_actions : int = 9           # NUMBER OF POSSIBLE ACTIONS                    - ATARI100K has 9 discrete actions
    
    # HIERARCHICAL LOSS (NOVELTY) 
    
    # e.g.: layer_weights = [1.0, 0.5, 0.1] for 3-layer HRVQ (L0, L1, L2)
    # HIGHER WEIGHTING FOR COARSE LAYER (L0) if desired; default set in __post_init__
    layer_weights: list = None
    
    
    def __post_init__(self):
        """ INITIALISE DERIVED PARAMETERS """
        if self.layer_weights is None:
            # DEFAULT FOR 3-LAYER HRVQ (L0, L1, L2)
            self.layer_weights = [1.0, 0.5, 0.1]    
        
        # VALIDATE HEAD DIMENSIONS
        assert self.d_model % self.n_heads == 0, f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        
        # VALIDATE SEQUENCE LENGTH
        assert self.max_seq_len % 4 == 0 , f"max_seq_len {self.max_seq_len} must be divisible by 4 (tokens per time step)"


    @classmethod
    def from_yaml(
        cls,
        path : str = "configs/worldmodel.yaml"
    ):
        """ LOAD CONFIG FROM YAML FILE  """
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        return cls(
            # ARCHITECTURE
            d_model = cfg['model']['d_model'],
            n_layers = cfg['model']['n_layers'],
            n_heads = cfg['model']['n_heads'],
            d_ff = cfg['model']['d_ff'],
            dropout = cfg['model']['dropout'],
            max_seq_len = cfg['model']['max_seq_len'],
            num_codes = cfg['model']['num_codes'],
            num_actions = cfg['model']['num_actions'],
            
            # HIERARCHICAL LOSS
            layer_weights = cfg['model']['layer_weights'],
        )
        
    
    def __repr__(self):
        return(
            f"WORLD MODEL CONFIG:\n"
            
            f"  d_model: {self.d_model}\n"
            f"  n_layers: {self.n_layers}\n"
            f"  n_heads: {self.n_heads}\n"
            f"  d_ff: {self.d_ff}\n"
            f"  dropout: {self.dropout}\n"
            f"  max_seq_len: {self.max_seq_len}\n"
            f"  num_codes: {self.num_codes}\n"
            f"  num_actions: {self.num_actions}\n"
            f"  layer_weights: {self.layer_weights}\n"
        )
    

    
    
    
class TokenEmbedding(nn.Module):
    """ EMBED HIERARCHICAL (HRVQ) TOKENS + ACTIONS into TRANSFORMER SEQUENCE"""


    def __init__(
        self,
        config : WorldModelConfig,
    ):
        super().__init__()
        self.config = config
    
    
        """ LOOKUPS  """
        
        # 1. TOKEN Lookups - (3 tables for L0, L1, L2)
        self.token_embeds = nn.ModuleList([
            nn.Embedding(
                num_embeddings = config.num_codes, 
                embedding_dim = config.d_model
            ) 
            for _ in range(3) # L0, L1, L2
        ])
        
        # 2. ACTION Lookup - (1 table for all actions)
        self.action_embed = nn.Embedding(
            num_embeddings = config.num_actions,
            embedding_dim = config.d_model
        )
        
        # 3. LEVEL Lookup - (1 table for L0, L1, L2 + ACTION)
        self.level_embed = nn.Embedding(
            num_embeddings = 4,   # LEVELS: L0, L1, L2 + ACTION
            embedding_dim = config.d_model
        )
        
        # 4. POSITION Lookup - (1 table for all positions)
        self.pos_embed = nn.Embedding(
            num_embeddings = config.max_seq_len,
            embedding_dim = config.d_model
        )
        
    def forward(
        self,
        tokens : torch.tensor,      # SHAPE: (B, T, 3) - 3 LAYERS
        actions : torch.tensor,     # SHAPE: (B, T) - ACTIONS
    ) -> torch.tensor:              # SHAPE: (B, T*4, d_model) - 4 TOKENS PER TIME STEP
        """ FORWARD PASS  """
        
        B, T, _ = tokens.shape
        device = tokens.device
        
        
        # 1. EMBED Each component
        emb_level0 = self.token_embeds[0](tokens[..., 0]) 
        emb_level1 = self.token_embeds[1](tokens[..., 1])
        emb_level2 = self.token_embeds[2](tokens[..., 2])
        emb_action = self.action_embed(actions)
        
        # 2. ADD LEVEL EMBEDDING
        level_ids = torch.arange(4, device = device)  
        level_embeds = self.level_embed(level_ids)
        
        emb_level0 = emb_level0 + level_embeds[0]
        emb_level1 = emb_level1 + level_embeds[1]
        emb_level2 = emb_level2 + level_embeds[2]
        emb_action = emb_action + level_embeds[3]
        
        # 3. INTERLEAVE TOKENS PER TIME STEP
        seq = torch.stack(
            [emb_level0, emb_level1, emb_level2, emb_action], 
            dim = 2
        )
        
        seq = seq.reshape(B, T * 4, self.config.d_model)  # (B, T*4, d_model)
        
        # 4. ADD POSITIONAL EMBEDDING
        positions = torch.arange(T * 4, device = device)
        seq = seq + self.pos_embed(positions)
        
        return seq
        
     
        
        
def hierarchical_causal_mask(
    seq_len : int,
    device : torch.device
):
    """ CAUSAL MASK TO ENSURE MODEL CANNOT 'SEE' FUTURE TOKENS (TRIANGULAR MASK SINCE TOKENS ARE INTERLEAVED SEQUENTIALLY) """
    
    mask = torch.triu(torch.ones(size= (seq_len, seq_len), device = device), diagonal = 1).bool()
    
    # 0 = ALLOW ATTENTION
    # -inf = BLOCK ATTENTION
    
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    return mask












class TransformerBlock(nn.Module):
    """ STANDARD TRANSFORMER BLOCK  """
    pass

class HierarchicalWorldModel(nn.Module):
    """ MAIN HIERARCHICAL WORLD MODEL FOR ATARI100K PREDICTION  """
    pass

def hierarchical_loss():
    """ HIERARCHICAL LOSS FUNCTION  """
    pass



if __name__ == "__main__":
    pass

    
   