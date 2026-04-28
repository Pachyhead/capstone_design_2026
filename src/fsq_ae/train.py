import os
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
 
from .config import FSQAEConfig
from .dataset import EmotionEmbedDataset
from .model import FSQAutoEncoder
 
def load_emotion_head(device) :
    from funasr import AutoModel
    emo_model = AutoModel(
        model="iic/emotion2vec_plus_large",
        disable_update=True,
    )
    head = emo_model.model.proj.to(device).eval()
    for p in head.parameters():
        p.requires_grad = False
    print(f"[Loss] head: Linear({head.in_features} -> {head.out_features}), frozen")
    return head
 
def train(config: FSQAEConfig):
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
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
    
    use_kl = config.emotion_kl_weight > 0
    if use_kl:
        emotion_head = load_emotion_head(device)
        norm_mean = full_dataset.mean.to(device)
        norm_std = full_dataset.std.to(device)
        print(f"[Loss] MSE + {config.emotion_kl_weight} * KL(emotion2vec+)")
    else:
        emotion_head = None
        print("[Loss] MSE only")
 
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
        train_mse_sum = 0.0
        train_kl_sum = 0.0
        for step, x in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
 
            x_recon, _, _ = model(x)
            mse = F.mse_loss(x_recon, x)
            loss = mse

            if use_kl:
                x_orig = x * norm_std + norm_mean
                x_recon_orig = x_recon * norm_std + norm_mean
            
                with torch.no_grad():
                    p_orig = F.softmax(emotion_head(x_orig), dim=-1)
                    
                log_p_recon = F.log_softmax(emotion_head(x_recon_orig), dim=-1)
                
                kl = F.kl_div(log_p_recon, p_orig, reduction="batchmean")
                loss = mse + config.emotion_kl_weight * kl
                train_kl_sum += kl.item()
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
 
            train_mse_sum += mse.item()
            if step % config.log_interval == 0:
                if use_kl:
                    print(f"  ep{epoch} step{step:>4d} | "
                          f"mse {mse.item():.5f} | kl {kl.item():.5f}")
                else:
                    print(f"  ep{epoch} step{step:>4d} | mse {mse.item():.5f}")
 
        train_mse = train_mse_sum / len(train_loader)
        train_kl = train_kl_sum / len(train_loader) if use_kl else 0.0
 
        # validate
        model.eval()
        val_mse_sum, val_kl_sum, val_cos_sum, n_val = 0.0, 0.0, 0.0, 0
        val_match_sum = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                x_recon, _, _ = model(x)
                
                val_mse_sum += F.mse_loss(x_recon, x).item()
                val_cos_sum += F.cosine_similarity(x, x_recon, dim=-1).sum().item()
                
                if use_kl:
                    x_orig = x * norm_std + norm_mean
                    x_recon_orig = x_recon * norm_std + norm_mean
 
                    p_orig = F.softmax(emotion_head(x_orig), dim=-1)
                    log_p_recon = F.log_softmax(emotion_head(x_recon_orig), dim=-1)
                    val_kl_sum += F.kl_div(log_p_recon, p_orig, reduction="batchmean").item()
 
                    # 분류 일치율
                    pred_orig = emotion_head(x_orig).argmax(-1)
                    pred_recon = emotion_head(x_recon_orig).argmax(-1)
                    val_match_sum += (pred_orig == pred_recon).sum().item()
                
                n_val += x.size(0)
 
        val_mse = val_mse_sum / len(val_loader)
        val_cos = val_cos_sum / n_val
        
        if use_kl:
            val_kl = val_kl_sum / len(val_loader)
            val_match = val_match_sum / n_val * 100
            print(
                f"[ep {epoch:>3d}] train mse {train_mse:.5f} kl {train_kl:.5f} | "
                f"val mse {val_mse:.5f} cos {val_cos:.4f} kl {val_kl:.5f} "
                f"match {val_match:.2f}%"
            )
        else:
            print(
                f"[ep {epoch:>3d}] train {train_mse:.5f} | "
                f"val {val_mse:.5f} | cos {val_cos:.4f}"
            )
 
        # 체크포인트 저장
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            ckpt = {
                "model": model.state_dict(),
                "config": config.__dict__,
                "epoch": epoch,
                "val_mse": val_mse,
                "val_cos": val_cos,
                "norm_mean": full_dataset.mean,
                "norm_std": full_dataset.std,
            }
            if use_kl:
                ckpt["val_kl"] = val_kl
                ckpt["val_match_rate"] = val_match
            torch.save(ckpt, os.path.join(config.save_dir, "best.pt"))
 
    print(f"[학습 완료] best val MSE = {best_val_loss:.5f}")