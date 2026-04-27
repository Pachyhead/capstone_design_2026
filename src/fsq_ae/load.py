import torch
 
from .config import FSQAEConfig
from .model import FSQAutoEncoder
 
 
def load_fsq_ae(ckpt_path: str, device: str = "cuda"):
    """저장된 best.pt에서 모델 + 정규화 통계 복원.
 
    Returns:
        model: FSQAutoEncoder (eval 모드)
        cfg:   학습 시 사용한 FSQAEConfig
        norm:  {"mean": Tensor, "std": Tensor} - 송수신 양쪽에서 동일하게 적용
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
 
    cfg = FSQAEConfig(**checkpoint["config"])
    model = FSQAutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
 
    norm = {
        "mean": checkpoint["norm_mean"].to(device),
        "std": checkpoint["norm_std"].to(device),
    }
    return model, cfg, norm
 