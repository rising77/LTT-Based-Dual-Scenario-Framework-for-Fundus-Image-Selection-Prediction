"""
src/eval.py
===========

评估与可视化工具。

包括：
  - 基础指标：Accuracy, QWK, ECE
  - 第二章 EDA 图：可靠性图、u₁-u₂ 联合散点图、单分数覆盖率-风险曲线
  - 第四章结果图：覆盖率-风险曲线、α-覆盖率曲线
  - 第四章结果表：表 4-1 / 4-2 / 4-3
  - Spearman 相关 + bootstrap 95% CI
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score


# ============================ 基础指标 ============================
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def expected_calibration_error(probs, labels, n_bins: int = 15):
    """
    期望校准误差 ECE。
    probs : (N, C) softmax 概率
    labels: (N,) 真实标签
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, bins) - 1, 0, n_bins - 1)
    n = len(probs)
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            bin_stats.append({"bin": b, "lo": float(bins[b]), "hi": float(bins[b+1]),
                              "n": 0, "acc": float("nan"), "conf": float("nan")})
            continue
        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        weight   = mask.sum() / n
        ece += weight * abs(bin_acc - bin_conf)
        bin_stats.append({"bin": b, "lo": float(bins[b]), "hi": float(bins[b+1]),
                          "n": int(mask.sum()),
                          "acc": float(bin_acc), "conf": float(bin_conf)})
    return float(ece), bin_stats


# ============================ 第二章图：可靠性图 ============================
def plot_reliability_diagram(probs, labels, n_bins: int = 15,
                             save_path=None, title: str = "Reliability Diagram"):
    ece, stats = expected_calibration_error(probs, labels, n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    confs = [s["conf"] for s in stats if s["n"] > 0]
    accs  = [s["acc"]  for s in stats if s["n"] > 0]
    ns    = [s["n"]    for s in stats if s["n"] > 0]
    ax.scatter(confs, accs, s=[20 + 0.5 * x for x in ns],
               c="C0", alpha=0.85, label="Empirical")
    ax.set_xlabel("Confidence (mean)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title(f"{title}  (ECE = {ece:.4f})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return ece, stats


# ============================ 第二章图：u₁-u₂ 联合散点图 ============================
def plot_u1_u2_scatter(u1, u2, errors, save_path=None,
                       title: str = "u₁ vs u₂ joint scatter"):
    fig, ax = plt.subplots(figsize=(6, 6))
    correct = errors == 0
    wrong   = errors == 1
    ax.scatter(u1[correct], u2[correct], s=8,  c="C2", alpha=0.4, label="Correct")
    ax.scatter(u1[wrong],   u2[wrong],   s=18, c="C3", alpha=0.7, label="Misclassified")
    m1 = float(np.median(u1)); m2 = float(np.median(u2))
    ax.axvline(m1, color="k", lw=0.7, ls="--")
    ax.axhline(m2, color="k", lw=0.7, ls="--")
    ax.set_xlabel(r"$u_1$  (1 − max softmax)")
    ax.set_ylabel(r"$u_2$  (1 − KNN-5 consistency)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    quad = {
        "Q1_u1lo_u2lo": int(((u1 <= m1) & (u2 <= m2)).sum()),
        "Q2_u1hi_u2lo": int(((u1 >  m1) & (u2 <= m2)).sum()),
        "Q3_u1lo_u2hi": int(((u1 <= m1) & (u2 >  m2)).sum()),
        "Q4_u1hi_u2hi": int(((u1 >  m1) & (u2 >  m2)).sum()),
        "Q1_err": int(((u1 <= m1) & (u2 <= m2) & (errors == 1)).sum()),
        "Q2_err": int(((u1 >  m1) & (u2 <= m2) & (errors == 1)).sum()),
        "Q3_err": int(((u1 <= m1) & (u2 >  m2) & (errors == 1)).sum()),
        "Q4_err": int(((u1 >  m1) & (u2 >  m2) & (errors == 1)).sum()),
        "median_u1": m1, "median_u2": m2,
    }
    return quad


# ============================ 第二章图：单分数覆盖率-风险曲线 ============================
def coverage_risk_curve_single(scores, errors, n_grid: int = 200):
    """
    沿 score 升序构造 (coverage, R_cond) 曲线。
    """
    order = np.argsort(scores)
    err_sorted = errors[order]
    cum_err = np.cumsum(err_sorted)
    n = len(scores)
    cov  = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    if n_grid is not None and n > n_grid:
        idx = np.linspace(0, n - 1, n_grid).astype(int)
        cov, risk = cov[idx], risk[idx]
    return cov, risk


def plot_coverage_risk_single(curves: dict, save_path=None,
                              title: str = "Coverage–Risk Curves (single scores)"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (cov, risk) in curves.items():
        ax.plot(risk, cov, label=label, lw=1.5)
    ax.set_xlabel(r"Empirical $R_{\mathrm{cond}}$")
    ax.set_ylabel("Coverage")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================ 第四章图：4-1 覆盖率–α 曲线 ============================
def plot_fig41_coverage_vs_alpha(alpha_grid, coverage_dict, save_path=None,
                                 title="Coverage at risk budget α  (mean ± 1 std)"):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"u1": "C0", "u2": "C1", "Parallel": "C2", "Series": "C3"}
    for label, (mu, sd) in coverage_dict.items():
        c = colors.get(label, None)
        ax.plot(alpha_grid, mu, label=label, lw=1.8, color=c)
        ax.fill_between(alpha_grid, mu - sd, mu + sd, alpha=0.18, color=c)
    ax.set_xlabel(r"Risk budget $\alpha$")
    ax.set_ylabel("Coverage on test set")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================ 第四章图：4-2 α–覆盖率曲线 ============================
def plot_fig42_alpha_vs_coverage(coverage_grid, alpha_dict, save_path=None,
                                 title="Risk α at coverage level (mean ± 1 std)"):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"u1": "C0", "u2": "C1", "Parallel": "C2", "Series": "C3"}
    for label, (mu, sd) in alpha_dict.items():
        c = colors.get(label, None)
        ax.plot(coverage_grid, mu, label=label, lw=1.8, color=c)
        ax.fill_between(coverage_grid, mu - sd, mu + sd, alpha=0.18, color=c)
    ax.set_xlabel("Coverage on test set")
    ax.set_ylabel(r"Risk $\alpha$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================ Section 4 表格 ============================
def make_table_violation_rate(rcond_per_repeat, alpha_list, methods):
    """每种方法在每个 α 下的违反率（%）。"""
    rows = []
    for a_idx, a in enumerate(alpha_list):
        row = {"alpha": a}
        for m in methods:
            r = rcond_per_repeat[m][:, a_idx]
            row[m] = float((r > a).mean() * 100)
        rows.append(row)
    rows.append({"alpha": "理论上界 (δ=0.10)",
                 **{m: 10.0 for m in methods}})
    return pd.DataFrame(rows)


def make_table_coverage(coverage_per_repeat, alpha_list, methods):
    """每个 α 下四种方法的覆盖率均值 (%) 与并联提升 (pp)。"""
    rows = []
    for a_idx, a in enumerate(alpha_list):
        row = {"alpha": a}
        for m in methods:
            row[m] = float(coverage_per_repeat[m][:, a_idx].mean() * 100)
        if "Parallel" in methods and "u1" in methods and "u2" in methods:
            best_single = max(row["u1"], row["u2"])
            row["Parallel_gain_pp"] = row["Parallel"] - best_single
        rows.append(row)
    return pd.DataFrame(rows)


def make_table_alpha_at_coverage(alpha_at_cov, coverage_levels, methods):
    """每个目标覆盖率下四种方法的 α 均值 (%) 与串联收紧 (pp)。"""
    rows = []
    for c_idx, c in enumerate(coverage_levels):
        row = {"coverage": c}
        for m in methods:
            vals = alpha_at_cov[m][:, c_idx]
            vals = vals[~np.isnan(vals)]
            row[m] = float(vals.mean() * 100) if len(vals) > 0 else float("nan")
        if "Series" in methods and "u1" in methods and "u2" in methods:
            best_single = min(row["u1"], row["u2"])
            row["Series_tighten_pp"] = best_single - row["Series"]
        rows.append(row)
    return pd.DataFrame(rows)


# ============================ Spearman 相关 + 95% CI（自助法） ============================
def spearman_ci(x, y, n_boot: int = 1000, ci: float = 0.95, seed: int = 0):
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rb, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(rb):
            boots.append(rb)
    boots = np.array(boots)
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return float(rho), lo, hi
