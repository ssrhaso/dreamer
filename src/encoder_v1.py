""" SIMPLE TRAINABLE CNN ENCODER FOR ATARI FRAMES """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AtariCNNEncoder(nn.Module):
    """
    LIGHTWEIGHT CNN ENCODER FOR ATARI FRAMES (84x84)
    
    Architecture: 3 Conv layers + 2 FC layers
    Input:  (B, 4, 84, 84) - 4 stacked grayscale frames
    Output: (B, 384) L2-normalized embeddings
    
    Expected correlation: 0.75-0.85 on Atari PONGv5
    Parameters: ~ 1,881,888 (1.88M)
    """

    def __init__(
        self,
        input_channels: int = 4,
        embedding_dim: int = 384,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        # CONVOLUTIONAL FEATURE EXTRACTOR
        self.encoder = nn.Sequential(
            # LAYER 1: 84x84 -> 20x20
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0), # OUTPUT: (32, 20, 20)
            nn.ReLU(inplace=True),

            # LAYER 2: 20x20 -> 9x9
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # OUTPUT: (64, 9, 9)
            nn.ReLU(inplace=True),

            # LAYER 3: 9x9 -> 7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # OUTPUT: (64, 7, 7)
            nn.ReLU(inplace=True),

            nn.Flatten(),  # (64*7*7) = 3,136
        )

        # PROJECTION HEAD (feature compression)
        self.projection = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self._init_weights()
    

    def _init_weights(self):
        """Initialize weights to prevent vanishing/exploding gradients"""
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder
        
        Args:
            x: (B, 4, 84, 84) stacked grayscale frames
        
        Returns:
            z_norm: (B, embedding_dim) L2-normalized embeddings
        """
        
        y = self.encoder(x)       # (B, 3136)
        z = self.projection(y)    # (B, embedding_dim)
        z_norm = F.normalize(z, p=2, dim=1)  # L2 normalization
        
        return z_norm

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_batch(self, frames: np.ndarray) -> torch.Tensor:
        """
        Extract embeddings from numpy frames (for inference)
        
        Args:
            frames: (B, 4, 84, 84) uint8 numpy array [0-255]
        
        Returns:
            embeddings: (B, embedding_dim) normalized embeddings
        """
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor and move to device
        frames_tensor = torch.from_numpy(frames)
        device = next(self.parameters()).device
        frames_tensor = frames_tensor.to(device)
        
        # Extract embeddings
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(frames_tensor)
        
        return embeddings


""" TEST PIPELINE """
if __name__ == "__main__":

    print("TESTING ATARI CNN ENCODER")
   
    
    model = AtariCNNEncoder(input_channels=4, embedding_dim=384)
    print(f"\n Model created")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    print("\nTest: Forward pass (B, 4, 84, 84)")
    x = torch.randn(8, 4, 84, 84)
    out = model(x)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(out.shape)}")
    print(f"  L2 norm: {torch.norm(out, dim=1).mean():.3f} (should be 1.0)")
    assert out.shape == (8, 384)
    print(f"  PASSED")
    
    print("\n" + "="*60)
    print("ENCODER READY")
    print("="*60)