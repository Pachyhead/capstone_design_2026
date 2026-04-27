import torch.nn as nn
 
from .config import FSQAEConfig
from .quantizer import FSQ
 
 
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
        h1, h2 = config.hidden_dims
        in_dim = config.input_dim
 
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Linear(h2, bottleneck_dim),
        )
 
        # FSQ
        self.quantizer = FSQ(config.levels)
 
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Linear(h2, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Linear(h1, in_dim),
        )
 
    def forward(self, x):
        z = self.encoder(x)
        z_q = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q