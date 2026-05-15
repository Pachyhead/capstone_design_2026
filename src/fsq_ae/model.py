import torch
import torch.nn as nn
 
from .config import FSQAEConfig
from .quantizer import FSQ
 
 
class MLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        
    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + h
 
class FSQAutoEncoder(nn.Module):
    """
    학습/추론 양쪽에서 사용하는 메인 모델
 
    학습:    forward(x) -> (x_recon, z, z_q)
    추론:    encoder(x) -> z, quantizer(z) -> z_q, decoder(z_q) -> x_recon
    """
 
    def __init__(self, config: FSQAEConfig):
        super().__init__()
        self.config = config
        bottleneck_dim = len(config.levels)
        in_dim = config.input_dim
        h = config.hidden_dims_skip[0]
        n_blocks = len(config.hidden_dims_skip)
 
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h),
            *[MLPBlock(h) for _ in range(n_blocks)],
            nn.LayerNorm(h),
            nn.Linear(h, bottleneck_dim),
        )
 
        # FSQ
        self.quantizer = FSQ(config.levels)
 
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h),
            *[MLPBlock(h) for _ in range(n_blocks)],
            nn.LayerNorm(h),
            nn.Linear(h, in_dim),
        )
 
    def forward(self, x):
        z = self.encoder(x)
        z_q = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q
    
    @torch.no_grad()
    def encode_to_indices(self, x):
        self.eval()
        z = self.encoder(x)
        z_q = self.quantizer(z)
        return self.quantizer.code_to_indices(z_q)
    
    @torch.no_grad()
    def decode_from_indices(self, indices):
        self.eval()
        z_q = self.quantizer.indices_to_codes(indices)
        x_recon = self.decoder(z_q)
        return x_recon