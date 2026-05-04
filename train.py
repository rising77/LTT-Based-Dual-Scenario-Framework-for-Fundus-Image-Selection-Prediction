"""
src/train.py
============

EfficientNet-B3 的两阶段微调训练循环（论文 §A.3）。

Stage 1 (10 ep)：冻结 backbone，只训练 Dropout+Linear 分类头，lr = 1e-4。
Stage 2 (30 ep)：解冻全网络，分类头 lr = 1e-4，主干 lr = 1e-5（1/10）。
优化器：Adam；损失：Focal Loss γ=2.0；自动混合精度 (AMP)。

按验证集 QWK 选择最佳 checkpoint 保存到 CKPT_BEST。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score

from src.models import FocalLoss


# ============================ 评估 ============================
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    losses, ys, ps = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item() * x.size(0))
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(-1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "loss": float(np.sum(losses) / len(loader.dataset)),
        "acc":  float(accuracy_score(y_true, y_pred)),
        "qwk":  float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================ 训练单个 epoch ============================
def _train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, _ = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        n += x.size(0)
    return running_loss / max(n, 1)


# ============================ 两阶段训练主循环 ============================
def train_two_stage(model, train_loader, val_loader,
                    *,
                    stage1_epochs: int,
                    stage2_epochs: int,
                    head_lr: float,
                    backbone_lr_ratio: float,
                    weight_decay: float,
                    focal_gamma: float,
                    device,
                    ckpt_best_path: Path,
                    ckpt_last_path: Path):
    criterion = FocalLoss(gamma=focal_gamma)
    history = {"stage1": [], "stage2": []}
    best_qwk, best_ep = -1.0, -1

    # ============================ Stage 1：冻结主干 ============================
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=head_lr, weight_decay=weight_decay)
    scaler = GradScaler()

    print(f"\n=== Stage 1: head only, {stage1_epochs} epochs, lr={head_lr} ===")
    for ep in range(1, stage1_epochs + 1):
        tr_loss = _train_one_epoch(model, train_loader, criterion,
                                   optimizer, scaler, device)
        val = evaluate(model, val_loader, criterion, device)
        print(f"  [S1 ep {ep:02d}] train_loss={tr_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  val_acc={val['acc']:.4f}  "
              f"val_qwk={val['qwk']:.4f}")
        history["stage1"].append({"epoch": ep, "train_loss": tr_loss,
                                  "val_loss": val["loss"],
                                  "val_acc": val["acc"],
                                  "val_qwk": val["qwk"]})
        if val["qwk"] > best_qwk:
            best_qwk, best_ep = val["qwk"], ("S1", ep)
            torch.save(model.state_dict(), ckpt_best_path)

    # ============================ Stage 2：解冻全网络 ============================
    for p in model.backbone.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": head_lr * backbone_lr_ratio},
        {"params": model.head.parameters(),     "lr": head_lr},
        {"params": model.dropout.parameters(),  "lr": head_lr},  # 通常空
    ], weight_decay=weight_decay)
    scaler = GradScaler()

    print(f"\n=== Stage 2: full fine-tune, {stage2_epochs} epochs, "
          f"head_lr={head_lr}, bb_lr={head_lr*backbone_lr_ratio} ===")
    for ep in range(1, stage2_epochs + 1):
        tr_loss = _train_one_epoch(model, train_loader, criterion,
                                   optimizer, scaler, device)
        val = evaluate(model, val_loader, criterion, device)
        print(f"  [S2 ep {ep:02d}] train_loss={tr_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  val_acc={val['acc']:.4f}  "
              f"val_qwk={val['qwk']:.4f}")
        history["stage2"].append({"epoch": ep, "train_loss": tr_loss,
                                  "val_loss": val["loss"],
                                  "val_acc": val["acc"],
                                  "val_qwk": val["qwk"]})
        if val["qwk"] > best_qwk:
            best_qwk, best_ep = val["qwk"], ("S2", ep)
            torch.save(model.state_dict(), ckpt_best_path)

    torch.save(model.state_dict(), ckpt_last_path)
    print(f"\nBest val QWK = {best_qwk:.4f} at {best_ep}")
    return history, best_qwk
