import torch

from fsq_ae.load import load_fsq_ae

from .config import ReceiverConfig

class ReceiverDecode:
    def __init__(self, fsq_model, fsq_norm, device):
        self.fsq = fsq_model
        self.norm = fsq_norm
        self.device = device
    
    @classmethod
    def from_config(cls, cfg: ReceiverConfig) -> "ReceiverDecode":
        fsq_model, _, fsq_norm = load_fsq_ae(cfg.fsq_ckpt)
        return cls(fsq_model, fsq_norm, cfg.device)
    
    @torch.inference_mode()
    def decode(self, emotion_indices: tuple):
        indices = torch.tensor(
            emotion_indices,
            dtype=torch.long,
            device=self.device
        )
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        x_recon_norm = self.fsq.decode_from_indices(indices)
        emo_vec_1024d = x_recon_norm * self.norm["std"] + self.norm["mean"]
        
        return emo_vec_1024d.squeeze(0).cpu().numpy()