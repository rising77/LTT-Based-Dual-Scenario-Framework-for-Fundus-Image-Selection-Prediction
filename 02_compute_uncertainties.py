"""
scripts/02_compute_uncertainties.py
====================================

提取所有不确定性分数（u₁ + u₂ 候选）。

加载训练好的 EfficientNet-B3，对校准集和测试集计算：
  u₁              = 1 − max softmax
  u₂_knn{5,10,20} = 1 − KNN-K 一致性
  u₂_maha         = Mahalanobis 距离 (rank-percentile 归一化)
  u₂_mcd          = MC Dropout 预测熵 / log C

同时记录每种 u₂ 的计算耗时（论文 §A.5 表）。

输出：
  outputs/uncertainties_cal.npz
  outputs/uncertainties_test.npz
  outputs/uncertainty_timings.json
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import config as C
from src.data import load_dataframes, build_eval_transforms, APTOSDataset
from src.models import build_model
from src.uncertainty import compute_all_uncertainties


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # —— 数据 ——
    train_df, val_df, test_df = load_dataframes(C.DATA_ROOT)
    eval_tf  = build_eval_transforms(C.IMG_SIZE)
    train_ds = APTOSDataset(train_df, transforms=eval_tf, img_size=C.IMG_SIZE)
    cal_ds   = APTOSDataset(val_df,   transforms=eval_tf, img_size=C.IMG_SIZE)
    test_ds  = APTOSDataset(test_df,  transforms=eval_tf, img_size=C.IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=True)
    cal_loader   = DataLoader(cal_ds,   batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=C.NUM_WORKERS, pin_memory=True)

    # —— 模型（载入 best checkpoint） ——
    if not C.CKPT_BEST.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {C.CKPT_BEST}. 请先运行 scripts/01_train_baseline.py")
    model = build_model(num_classes=C.NUM_CLASSES, pretrained=False, device=device)
    state = torch.load(C.CKPT_BEST, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint → {C.CKPT_BEST.name}")

    # —— 校准集 ——
    print("\n" + "=" * 60)
    print("  Calibration set")
    print("=" * 60)
    res_cal = compute_all_uncertainties(model, train_loader, cal_loader,
                                        device=device,
                                        num_classes=C.NUM_CLASSES,
                                        mcd_T=20)

    # —— 测试集 ——
    print("\n" + "=" * 60)
    print("  Test set")
    print("=" * 60)
    res_test = compute_all_uncertainties(model, train_loader, test_loader,
                                         device=device,
                                         num_classes=C.NUM_CLASSES,
                                         mcd_T=20)

    # —— 保存 ——
    keys_to_save = ["logits", "preds", "labels", "features",
                    "u1", "u2_knn5", "u2_knn10", "u2_knn20",
                    "u2_maha", "u2_mcd"]
    np.savez_compressed(C.UNC_CAL_NPZ,  **{k: res_cal[k]  for k in keys_to_save})
    np.savez_compressed(C.UNC_TEST_NPZ, **{k: res_test[k] for k in keys_to_save})
    print(f"\nSaved cal  → {C.UNC_CAL_NPZ.name}")
    print(f"Saved test → {C.UNC_TEST_NPZ.name}")

    # —— Timings (per-sample seconds) ——
    timings = {
        "calibration":   res_cal["timings"],
        "test":          res_test["timings"],
        "calibration_n": int(len(res_cal["preds"])),
        "test_n":        int(len(res_test["preds"])),
    }
    for split, n in [("calibration", timings["calibration_n"]),
                     ("test",        timings["test_n"])]:
        timings[f"{split}_per_sample_ms"] = {
            k: 1000.0 * v / max(n, 1)
            for k, v in timings[split].items()
        }
    with open(C.OUTPUT_DIR / "uncertainty_timings.json", "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)
    print(f"Saved timings → {(C.OUTPUT_DIR / 'uncertainty_timings.json').name}")


if __name__ == "__main__":
    main()
