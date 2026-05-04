"""
src/ltt.py
==========

Learn-Then-Test (LTT) 风险控制框架的实现。

核心思路 (Angelopoulos & Bates, 2021)
-------------------------------------
对每个候选阈值 λ，把"当 λ 触发选择时，选择集上的条件错误率"当作一个假设
H_λ : R_cond(λ) ≤ α 来检验。我们用单边 Hoeffding-Bentkus (HB) p 值；
为了避免逐 λ 校正太严，用 Fixed-Sequence Testing (FST)：先用一小段
保留校准数据决定 λ 的考察顺序，再沿这个顺序扫到第一个被拒绝的 λ 处停止
（FWER ≤ δ 直接成立，无需 Bonferroni）。

四种风控模式：
  single_score_ltt(scores, errors, α, δ, grid)            → 单分数
  two_score_ltt(u1, u2, errors, α, δ, ..., mode)          → 并联 (OR) / 串联 (AND)

应用层（在 test 集上根据返回的 λ 做选择）：
  apply_single, apply_parallel, apply_series, empirical_metrics
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.stats import binom


# ============================ HB p 值 ============================
def hb_pvalue(n_sel: int, n_err: int, alpha: float) -> float:
    """
    单边 Hoeffding-Bentkus p 值，用于检验 H : E[R_cond] ≤ α，损失 ∈ [0, 1]。

    HB 公式（针对 0/1 损失即 Bernoulli）退化为 Binomial 尾概率：
        p_HB = e * P(Bin(n_sel, α) ≤ n_err)
    其中 e = exp(1)。我们再与 Hoeffding 上界取 min（更紧）。
    返回的是"R_cond ≤ α 假设下"的 p 值；越小越显著。

    若 n_sel == 0：返回 1.0（无样本 → 无法拒绝原假设 → 不接受该 λ）。
    """
    if n_sel <= 0:
        return 1.0
    rhat = n_err / n_sel
    # Hoeffding 一侧
    if rhat >= alpha:
        p_hoef = 1.0
    else:
        p_hoef = float(np.exp(-2.0 * n_sel * (alpha - rhat) ** 2))
    # Bentkus 一侧（n_err 越多越接近 1）
    p_b = float(np.e * binom.cdf(n_err, n_sel, alpha))
    return min(1.0, p_hoef, p_b)


# ============================ 选择算子 ============================
def apply_single(scores: np.ndarray, lam: float) -> np.ndarray:
    """单分数：u ≤ λ 即选择。"""
    return scores <= lam


def apply_parallel(u1: np.ndarray, u2: np.ndarray,
                   lam: Tuple[float, float]) -> np.ndarray:
    """并联 (OR)：u1 ≤ λ1 或 u2 ≤ λ2 即选择。覆盖率天然 ≥ 单分数。"""
    return (u1 <= lam[0]) | (u2 <= lam[1])


def apply_series(u1: np.ndarray, u2: np.ndarray,
                 lam: Tuple[float, float]) -> np.ndarray:
    """串联 (AND)：u1 ≤ λ1 且 u2 ≤ λ2 才选择。风险天然 ≤ 单分数。"""
    return (u1 <= lam[0]) & (u2 <= lam[1])


def empirical_metrics(selected: np.ndarray, errors: np.ndarray
                      ) -> Tuple[float, float]:
    """返回 (coverage, R_cond) = (|S|/n, err 数 / |S|)。"""
    n = len(selected)
    n_sel = int(selected.sum())
    if n_sel == 0:
        return 0.0, 0.0
    return n_sel / n, float(errors[selected].sum()) / n_sel


# ============================ 单分数 LTT ============================
def single_score_ltt(scores: np.ndarray, errors: np.ndarray,
                     alpha: float, delta: float,
                     grid: np.ndarray) -> Optional[float]:
    """
    在 1-D 网格 `grid` 上，按 λ 从大到小搜索（先放宽再收紧）。
    Fixed-Sequence Testing：返回沿该顺序首次失败之前的最大 λ；若首点就失败
    则返回 None。FST 下 FWER ≤ δ 直接成立，不需要再做 Bonferroni。
    """
    grid_sorted = np.sort(grid)[::-1]                  # 大 → 小
    best_lam = None
    for lam in grid_sorted:
        sel = apply_single(scores, lam)
        n_sel = int(sel.sum())
        n_err = int(errors[sel].sum())
        p = hb_pvalue(n_sel, n_err, alpha)
        if p <= delta:
            best_lam = float(lam)                      # 接受这个 λ，继续往更紧方向走
        else:
            break                                      # FST：第一次失败即停
    return best_lam


# ============================ 双分数 LTT ============================
def _fst_order_two_score(u1_split: np.ndarray, u2_split: np.ndarray,
                         err_split: np.ndarray,
                         alpha: float,
                         grid_u1: np.ndarray, grid_u2: np.ndarray,
                         mode: str) -> np.ndarray:
    """
    在 9:1 拆分的 10% 数据上，给二维网格 (λ1, λ2) 排序。
    排序键：经验风险 R̂(λ) 从小到大；R̂ 相等时按"覆盖率从大到小"作 tie-break，
    这样能优先考察更宽松、覆盖率更高的 (λ1, λ2)。
    """
    apply_fn = apply_parallel if mode == "or" else apply_series
    pairs = []
    for l1 in grid_u1:
        for l2 in grid_u2:
            sel = apply_fn(u1_split, u2_split, (l1, l2))
            n_sel = int(sel.sum())
            if n_sel == 0:
                rhat = 1.0
                cov = 0.0
            else:
                rhat = err_split[sel].mean()
                cov  = n_sel / len(sel)
            pairs.append((rhat, -cov, float(l1), float(l2)))
    # 升序：风险小 → 覆盖率大 → λ
    pairs.sort()
    return np.array([(p[2], p[3]) for p in pairs], dtype=np.float64)


def two_score_ltt(u1_cal: np.ndarray, u2_cal: np.ndarray,
                  err_cal: np.ndarray,
                  alpha: float, delta: float,
                  grid_u1: np.ndarray, grid_u2: np.ndarray,
                  mode: str = "or",
                  split_idx: Optional[np.ndarray] = None
                  ) -> Optional[Tuple[float, float]]:
    """
    双分数 LTT。

    Parameters
    ----------
    u1_cal, u2_cal, err_cal : 校准集上的 u₁/u₂/0-1 错误指示
    alpha, delta            : 风险预算 / 显著性
    grid_u1, grid_u2        : 一维 λ 网格
    mode                    : "or"（并联）或 "and"（串联）
    split_idx               : bool 数组，True=10% 用于决定 FST 顺序，False=90% 用于 p 值
                              如为 None，则平均切

    Returns
    -------
    (λ1*, λ2*) 或 None.  None 表示在显著性 δ 下没有 (λ1, λ2) 满足风控。

    数据分割：10% 决定考察顺序；剩下 90% 计算 p 值（保证独立性，FWER ≤ δ）。
    """
    assert mode in {"or", "and"}
    apply_fn = apply_parallel if mode == "or" else apply_series

    if split_idx is None:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(u1_cal))
        split_idx = np.zeros(len(u1_cal), dtype=bool)
        split_idx[perm[:max(1, int(0.1 * len(u1_cal)))]] = True

    u1_o, u2_o, e_o = u1_cal[split_idx],  u2_cal[split_idx],  err_cal[split_idx]
    u1_p, u2_p, e_p = u1_cal[~split_idx], u2_cal[~split_idx], err_cal[~split_idx]

    # 在 10% 子集上得到 FST 顺序
    order = _fst_order_two_score(u1_o, u2_o, e_o, alpha,
                                 grid_u1, grid_u2, mode)
    # 在 90% 子集上沿该顺序检验
    best_lam = None
    for l1, l2 in order:
        sel = apply_fn(u1_p, u2_p, (l1, l2))
        n_sel = int(sel.sum())
        n_err = int(e_p[sel].sum())
        p = hb_pvalue(n_sel, n_err, alpha)
        if p <= delta:
            best_lam = (float(l1), float(l2))          # 接受，继续
        else:
            break                                      # FST：第一次失败即停
    return best_lam
