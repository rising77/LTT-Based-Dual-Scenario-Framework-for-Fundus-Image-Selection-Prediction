"""
scripts/01_train_baseline.py
============================

训练 EfficientNet-B3 基线模型（论文 §A.3）。

策略：两阶段微调
  Stage 1 (10 ep) ：冻结主干，训练分类头，lr=1e-4
  Stage 2 (30 ep) ：解冻全网络，head_lr=1e-4，backbone_lr=1e-5
优化器 Adam；损失 Focal Loss γ=2.0；混合精度 AMP；按验证 QWK 选 best。

输出：
  outputs/efficientnet_b3_best.pth
  outputs/efficientnet_b3_last.pth
  outputs/baseline_metrics.json     # 测试集 acc / qwk + 训练 history
  outputs/fig_confusion_matrix.png
"""
import sys, json, random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import config as C
from src.data import load_dataframes, make_loaders
from src.models import build_model, FocalLoss
from src.train import train_two_stage, evaluate


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False         # 训练时让 cudnn 自由优化
    torch.backends.cudnn.benchmark     = True


def main():
    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # —— 数据 ——
    train_df, val_df, test_df = load_dataframes(C.DATA_ROOT)
    print(f"Sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Train class counts:\n{train_df['diagnosis'].value_counts().to_dict()}")
    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df,
        img_size=C.IMG_SIZE, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS,
    )

    # —— 模型 ——
    model = build_model(num_classes=C.NUM_CLASSES, pretrained=True, device=device)
    print(f"Model loaded: {C.MODEL_NAME} (feat_dim={model.feat_dim})")

    # —— 两阶段训练 ——
    history, best_qwk = train_two_stage(
        model, train_loader, val_loader,
        stage1_epochs=C.STAGE1_EPOCHS,
        stage2_epochs=C.STAGE2_EPOCHS,
        head_lr=C.HEAD_LR,
        backbone_lr_ratio=C.BACKBONE_LR_RATIO,
        weight_decay=C.WEIGHT_DECAY,
        focal_gamma=C.FOCAL_GAMMA,
        device=device,
        ckpt_best_path=C.CKPT_BEST,
        ckpt_last_path=C.CKPT_LAST,
    )

    # —— 加载最佳权重，在测试集上评估 ——
    model.load_state_dict(torch.load(C.CKPT_BEST, map_location=device))
    res = evaluate(model, test_loader, FocalLoss(gamma=C.FOCAL_GAMMA), device)
    print("\n" + "=" * 60)
    print(f"  TEST  loss={res['loss']:.4f}  acc={res['acc']:.4f}  qwk={res['qwk']:.4f}")
    print("=" * 60)
    print("\nClassification report:")
    print(classification_report(res["y_true"], res["y_pred"], digits=4))

    # —— 混淆矩阵 ——
    cm = confusion_matrix(res["y_true"], res["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix  (Acc={res['acc']:.4f}, QWK={res['qwk']:.4f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(C.OUTPUT_DIR / "fig_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    # —— 保存指标 ——
    metrics = {
        "test_loss": res["loss"],
        "test_acc":  res["acc"],
        "test_qwk":  res["qwk"],
        "best_val_qwk": best_qwk,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "config": {
            "model": C.MODEL_NAME,
            "img_size": C.IMG_SIZE,
            "batch_size": C.BATCH_SIZE,
            "stage1_epochs": C.STAGE1_EPOCHS,
            "stage2_epochs": C.STAGE2_EPOCHS,
            "head_lr": C.HEAD_LR,
            "backbone_lr_ratio": C.BACKBONE_LR_RATIO,
            "focal_gamma": C.FOCAL_GAMMA,
        }
    }
    with open(C.OUTPUT_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {C.OUTPUT_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
