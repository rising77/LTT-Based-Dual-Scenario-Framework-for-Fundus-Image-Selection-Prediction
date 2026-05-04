"""
scripts/05_make_results.py
===========================

整合 100 次重复的 LTT 实验结果，生成论文第四章所有表格与图表。

输入：outputs/ltt_repeats.npz  （由 04_run_ltt_experiments.py 产生）

输出：
  outputs/table41_violations.csv   （表 4-1：违反率 P(R_cond > α)）
  outputs/table42_coverage.csv     （表 4-2：覆盖率均值 + 并联提升 pp）
  outputs/table43_alpha.csv        （表 4-3：达目标覆盖率所需 α + 串联收紧 pp）
  outputs/fig41_coverage_risk.png  （图 4-1：四方法覆盖率-α 曲线）
  outputs/fig42_alpha_coverage.png （图 4-2：四方法 α-覆盖率曲线）
  outputs/wilcoxon_results.json    （配对显著性检验结果）
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import config as C
from src.eval import (
    make_table_violation_rate,
    make_table_coverage,
    make_table_alpha_at_coverage,
    plot_fig41_coverage_vs_alpha,
    plot_fig42_alpha_vs_coverage,
)


METHODS = ["u1", "u2", "Parallel", "Series"]


# ============================ 反向：α at coverage ============================
def alpha_at_coverage(coverage_per_repeat: np.ndarray,
                      alpha_grid: np.ndarray,
                      coverage_levels) -> np.ndarray:
    """
    对每次重复：沿 α 升序构造单调阶梯（cummax），找出使 coverage ≥ c 的最小 α。
    返回 (n_repeats, n_coverage)，找不到时记 np.nan。
    """
    R, _ = coverage_per_repeat.shape
    n_c = len(coverage_levels)
    out = np.full((R, n_c), np.nan)
    sort_idx = np.argsort(alpha_grid)
    a_sorted = alpha_grid[sort_idx]
    for r in range(R):
        c_mono = np.maximum.accumulate(coverage_per_repeat[r][sort_idx])
        for j, c in enumerate(coverage_levels):
            ok = np.where(c_mono >= c)[0]
            if len(ok) > 0:
                out[r, j] = a_sorted[ok[0]]
    return out


# ============================ Wilcoxon 检验工具 ============================
def safe_wilcoxon_greater(diff: np.ndarray):
    """
    单侧 Wilcoxon signed-rank：H1: median(diff) > 0。
    diff 全 0 或样本不足时返回 (None, 1.0)。
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or np.allclose(diff, 0):
        return None, 1.0
    try:
        stat, p = wilcoxon(diff, alternative="greater", zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return None, 1.0


# ============================ 主流程 ============================
def main():
    npz_path = C.OUTPUT_DIR / "ltt_repeats.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path}. 请先运行 scripts/04_run_ltt_experiments.py")
    ltt = np.load(npz_path)
    alpha_grid = ltt["alpha_grid"]
    coverage = {
        "u1":       ltt["coverage_u1"],
        "u2":       ltt["coverage_u2"],
        "Parallel": ltt["coverage_parallel"],
        "Series":   ltt["coverage_series"],
    }
    rcond = {
        "u1":       ltt["rcond_u1"],
        "u2":       ltt["rcond_u2"],
        "Parallel": ltt["rcond_parallel"],
        "Series":   ltt["rcond_series"],
    }
    R = coverage["u1"].shape[0]
    print(f"Loaded {R} repeats × {len(alpha_grid)} α × 4 methods")

    # ---------- 报告 α 子集 ----------
    report_alphas = C.ALPHA_LIST                                # [0.10, 0.125, 0.15, 0.175]
    sel = np.array([int(np.argmin(np.abs(alpha_grid - a))) for a in report_alphas])
    coverage_sel = {m: coverage[m][:, sel] for m in METHODS}
    rcond_sel    = {m: rcond[m][:, sel]    for m in METHODS}

    # ---------- 表 4-1：违反率 ----------
    df41 = make_table_violation_rate(rcond_sel, report_alphas, METHODS)
    df41.to_csv(C.OUTPUT_DIR / "table41_violations.csv", index=False, encoding="utf-8-sig")
    print("\n=== Table 4-1：违反率 P(R_cond > α) (%) ===")
    print(df41.to_string(index=False))

    # ---------- 表 4-2：覆盖率 ----------
    df42 = make_table_coverage(coverage_sel, report_alphas, METHODS)
    df42.to_csv(C.OUTPUT_DIR / "table42_coverage.csv", index=False, encoding="utf-8-sig")
    print("\n=== Table 4-2：覆盖率均值 (%) 与并联提升 (pp) ===")
    print(df42.to_string(index=False))

    # ---------- 表 4-3：达目标覆盖率所需 α ----------
    coverage_levels = C.COVERAGE_LEVELS                          # [0.50, 0.60, 0.70]
    alpha_at_cov = {m: alpha_at_coverage(coverage[m], alpha_grid, coverage_levels)
                    for m in METHODS}
    df43 = make_table_alpha_at_coverage(alpha_at_cov, coverage_levels, METHODS)
    df43.to_csv(C.OUTPUT_DIR / "table43_alpha.csv", index=False, encoding="utf-8-sig")
    print("\n=== Table 4-3：达目标覆盖率所需 α (%) 与串联收紧 (pp) ===")
    print(df43.to_string(index=False))

    # ---------- 图 4-1 ----------
    cov_dict = {m: (coverage[m].mean(axis=0), coverage[m].std(axis=0))
                for m in METHODS}
    plot_fig41_coverage_vs_alpha(
        alpha_grid, cov_dict,
        save_path=C.OUTPUT_DIR / "fig41_coverage_risk.png",
        title=f"Coverage on test set (mean ± 1 std over {R} repeats)",
    )
    print(f"\nSaved → fig41_coverage_risk.png")

    # ---------- 图 4-2 ----------
    fine_cov_levels = np.round(np.linspace(0.30, 0.95, 14), 3)
    alpha_at_cov_fine = {m: alpha_at_coverage(coverage[m], alpha_grid, fine_cov_levels)
                         for m in METHODS}
    alpha_dict = {}
    for m in METHODS:
        a_arr = alpha_at_cov_fine[m]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(a_arr, axis=0)
            sd = np.nanstd(a_arr, axis=0)
        alpha_dict[m] = (mu, sd)
    plot_fig42_alpha_vs_coverage(
        fine_cov_levels, alpha_dict,
        save_path=C.OUTPUT_DIR / "fig42_alpha_coverage.png",
        title=f"Risk α at target coverage (mean ± 1 std over {R} repeats)",
    )
    print(f"Saved → fig42_alpha_coverage.png")

    # ---------- Wilcoxon 配对检验 ----------
    print("\n=== Wilcoxon signed-rank tests ===")
    wilcoxon_results = {
        "coverage_parallel_vs_best_single": {},
        "alpha_series_vs_best_single": {},
    }

    # H1: Parallel 覆盖率 > max(u1, u2) 覆盖率
    for a in report_alphas:
        k = int(np.argmin(np.abs(alpha_grid - a)))
        best_single = np.maximum(coverage["u1"][:, k], coverage["u2"][:, k])
        diff = coverage["Parallel"][:, k] - best_single
        stat, p = safe_wilcoxon_greater(diff)
        wilcoxon_results["coverage_parallel_vs_best_single"][f"alpha={a}"] = {
            "statistic":    stat,
            "p_value":      p,
            "mean_diff_pp": float(np.nanmean(diff) * 100),
            "n":            int(np.sum(~np.isnan(diff))),
        }
        print(f"  Parallel > max(u1,u2) at α={a:.3f}: "
              f"p = {p:.4g}  (Δcoverage = {np.nanmean(diff)*100:+.2f} pp)")

    # H1: Series 所需 α < min(u1, u2) 所需 α
    for c_idx, c in enumerate(coverage_levels):
        a_u1  = alpha_at_cov["u1"][:, c_idx]
        a_u2  = alpha_at_cov["u2"][:, c_idx]
        a_ser = alpha_at_cov["Series"][:, c_idx]
        # 缺失对齐：对应 repeat 上若任一方法找不到 α，就跳过该 repeat
        mask = (~np.isnan(a_u1)) & (~np.isnan(a_u2)) & (~np.isnan(a_ser))
        diff = np.where(mask, np.minimum(a_u1, a_u2) - a_ser, np.nan)
        stat, p = safe_wilcoxon_greater(diff)
        wilcoxon_results["alpha_series_vs_best_single"][f"coverage={c}"] = {
            "statistic":    stat,
            "p_value":      p,
            "mean_diff_pp": float(np.nanmean(diff) * 100) if mask.any() else None,
            "n_valid":      int(mask.sum()),
        }
        diff_pp = np.nanmean(diff) * 100 if mask.any() else float("nan")
        print(f"  Series < min(u1,u2) at coverage={c:.2f}: "
              f"p = {p:.4g}  (Δα = {diff_pp:+.2f} pp, n = {mask.sum()})")

    with open(C.OUTPUT_DIR / "wilcoxon_results.json", "w", encoding="utf-8") as f:
        json.dump(wilcoxon_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → wilcoxon_results.json")

    print("\n" + "=" * 60)
    print(f"All Section-4 outputs saved to {C.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
