import os
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
 
from .config import FSQAEConfig
from .dataset import EmotionEmbedDataset
from .model import FSQAutoEncoder
 
 
def train(config: FSQAEConfig):
    torch.manual_seed(config.seed)
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
 
    # 데이터
    full_dataset = EmotionEmbedDataset(config.data_path)
    train_set, val_set = full_dataset.speaker_split(
        config.val_speaker_ratio, config.seed
    )
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
 
    # 모델
    model = FSQAutoEncoder(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
 
    n_params = sum(p.numel() for p in model.parameters())
    cb_size = model.quantizer.codebook_size
    bits = torch.log2(torch.tensor(float(cb_size))).item()
    print(
        f"[Model] 파라미터 {n_params:,}개 | "
        f"코드북 {cb_size:,}개 (~{bits:.2f} 비트)"
    )
 
    # STE 동작 확인
    model.train()
    dummy = torch.randn(2, config.input_dim, device=device)
    dummy_recon, _, _ = model(dummy)
    F.mse_loss(dummy_recon, dummy).backward()
    encoder_grad = model.encoder[0].weight.grad                # 첫 Linear의 weight
    assert encoder_grad is not None and encoder_grad.norm() > 0, (
        "STE가 작동하지 않음 - quantizer.py의 round_with_ste 확인 필요"
    )
    print(f"[STE 체크] 인코더 그래디언트 norm = {encoder_grad.norm():.4f} OK")
    model.zero_grad()
 
    # 학습 루프
    os.makedirs(config.save_dir, exist_ok=True)
    best_val_loss = float("inf")
 
    for epoch in range(1, config.num_epochs + 1):
 
        # train
        model.train()
        train_loss = 0.0
        for step, x in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
 
            x_recon, _, _ = model(x)
            loss = F.mse_loss(x_recon, x)
 
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
 
            train_loss += loss.item()
            if step % config.log_interval == 0:
                print(f"  ep{epoch} step{step:>4d} | mse {loss.item():.5f}")
 
        train_loss /= len(train_loader)
 
        # validate
        model.eval()
        val_loss_sum, val_cos_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                x_recon, _, _ = model(x)
                val_loss_sum += F.mse_loss(x_recon, x).item()
                val_cos_sum += F.cosine_similarity(x, x_recon, dim=-1).sum().item()
                n_val += x.size(0)
 
        val_loss = val_loss_sum / len(val_loader)
        val_cos = val_cos_sum / n_val
        print(
            f"[ep {epoch:>3d}] train {train_loss:.5f} | "
            f"val {val_loss:.5f} | cos {val_cos:.4f}"
        )
 
        # 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "config": config.__dict__,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_cos": val_cos,
                # 송수신측에서 같은 정규화를 쓸 수 있도록 통계도 함께 저장
                "norm_mean": full_dataset.mean,
                "norm_std": full_dataset.std,
            }, os.path.join(config.save_dir, "best.pt"))
 
    print(f"[학습 완료] best val MSE = {best_val_loss:.5f}")