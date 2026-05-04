"""
scripts/03_eda.py
==================

第二章的三个发现（EDA）。

发现一：Softmax 概率不可直接信任 → 可靠性图 + ECE
发现二：u₁ 与 u₂ 捕捉不同信息 → Spearman 相关 + u₁-u₂ 联合散点 + 四象限统计
发现三：单分数没有全局赢家 → 单分数的覆盖率-风险曲线交叉

输出：
  outputs/fig_reliability.png            (图 3)
  outputs/fig_scatter_u1u2.png           (图 4)
  outputs/fig_coverage_risk_single.png   (图 5)
  outputs/eda_metrics.json               (ECE、Spearman 表、四象限表)
  outputs/spearman_table.csv             (论文表 3a)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import config as C
from src.uncertainty import softmax_np
from src.eval import (
    plot_reliability_diagram,
    plot_u1_u2_scatter,
    coverage_risk_curve_single,
    plot_coverage_risk_single,
    spearman_ci,
)


def main():
    if not C.UNC_CAL_NPZ.exists():
        raise FileNotFoundError(
            f"Missing {C.UNC_CAL_NPZ}. 请先运行 scripts/02_compute_uncertainties.py")

    cal  = np.load(C.UNC_CAL_NPZ)

    # 用校准集做 EDA（论文 §2.3 都在校准集上做）
    logits = cal["logits"]
    labels = cal["labels"]
    preds  = cal["preds"]
    u1     = cal["u1"]
    errors = (preds != labels).astype(int)
    probs  = softmax_np(logits)

    # ---------- 发现一：可靠性图 + ECE ----------
    print("\n[Finding 1] Reliability diagram + ECE")
    ece, bin_stats = plot_reliability_diagram(
        probs, labels, n_bins=15,
        save_path=C.OUTPUT_DIR / "fig_reliability.png",
        title="Reliability Diagram (calibration set)",
    )
    print(f"  ECE = {ece:.4f}")
    mask_low = u1 <= 0.10
    if mask_low.sum() > 0:
        low_acc = float((preds[mask_low] == labels[mask_low]).mean())
        print(f"  acc on (u1 ≤ 0.10) bin: {low_acc:.4f} "
              f"(理想值 ≈ 0.95 if calibrated)")
    else:
        low_acc = float("nan")

    # ---------- 发现二：u₂ 候选的 Spearman 相关 + 散点 ----------
    print("\n[Finding 2] Spearman with u1 across u2 candidates")
    candidates = ["u2_knn5", "u2_knn10", "u2_knn20", "u2_maha", "u2_mcd"]
    rows = []
    for c in candidates:
        rho, lo, hi = spearman_ci(u1, cal[c], n_boot=1000, seed=0)
        rows.append({"candidate": c, "spearman_rho": rho,
                     "ci_lower": lo, "ci_upper": hi})
    spear_df = pd.DataFrame(rows)
    spear_df.to_csv(C.OUTPUT_DIR / "spearman_table.csv", index=False)
    print(spear_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 选定 u₂ = u2_knn5（论文最终选择）
    u2 = cal["u2_knn5"]
    quad = plot_u1_u2_scatter(
        u1, u2, errors,
        save_path=C.OUTPUT_DIR / "fig_scatter_u1u2.png",
        title="u₁ vs u₂ joint scatter (calibration set)",
    )
    print(f"\n  Quadrant stats (median split):")
    print(json.dumps(quad, indent=2))

    # ---------- 发现三：单分数覆盖率-风险曲线 ----------
    print("\n[Finding 3] Coverage–risk curves of single scores")
    cov1, risk1 = coverage_risk_curve_single(u1, errors)
    cov2, risk2 = coverage_risk_curve_single(u2, errors)
    plot_coverage_risk_single(
        {"u1": (cov1, risk1), "u2 (KNN-5)": (cov2, risk2)},
        save_path=C.OUTPUT_DIR / "fig_coverage_risk_single.png",
        title="Coverage–Risk curves (single scores, calibration set)",
    )
    cov_at_alpha = {}
    for label, (cov, risk) in [("u1", (cov1, risk1)),
                                ("u2", (cov2, risk2))]:
        valid = risk <= 0.10
        cov_at_alpha[label] = float(cov[valid].max()) if valid.any() else 0.0
    print(f"  Max coverage at R_cond ≤ 0.10: {cov_at_alpha}")

    # ---------- 保存 ----------
    out = {
        "ece": ece,
        "low_u1_bin_accuracy": low_acc,
        "spearman_with_u1": rows,
        "u2_chosen": "u2_knn5",
        "quadrants": quad,
        "max_coverage_at_alpha_0.10": cov_at_alpha,
        "bin_stats_for_reliability": bin_stats,
    }
    with open(C.OUTPUT_DIR / "eda_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved EDA metrics → {(C.OUTPUT_DIR / 'eda_metrics.json').name}")


if __name__ == "__main__":
    main()
