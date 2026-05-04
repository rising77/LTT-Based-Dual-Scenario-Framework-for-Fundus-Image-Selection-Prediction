"""
scripts/04_run_ltt_experiments.py
==================================

100 次重复的 LTT 风险控制实验。

把校准集 (732) 与测试集 (733) 合并为 1465 个样本，做 100 次分层重抽样：
  每次都重新抽出 cal=732, test=733；
  在 cal 内再做 9:1 拆分（10% 用于决定 FST 顺序，90% 用于 p 值）；
  对每个 α ∈ ALPHA_GRID 跑 4 种方法：单 u₁、单 u₂、并联 (OR)、串联 (AND)；
  在 test 上记录 λ*、覆盖率 cov 与条件风险 R_cond。

ALPHA_GRID 同时覆盖：
  ① 论文报告的 4 个 α：0.10 / 0.125 / 0.15 / 0.175
  ② 反向视角（Table 4-3 / Fig 4-2）所需的密网格

输出：
  outputs/ltt_repeats.npz
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import config as C
from src.ltt import (
    single_score_ltt,
    two_score_ltt,
    apply_single,
    apply_parallel,
    apply_series,
    empirical_metrics,
)


# 论文使用 KNN-5 作为 u₂（在 §3.4 候选比较中胜出）
U2_KEY = "u2_knn5"

# α 网格：32 个点，步长 0.005，覆盖 [0.05, 0.205]
# 包含报告的四个 α（0.10 / 0.125 / 0.15 / 0.175），同时为反向视角提供分辨率
ALPHA_GRID = np.round(np.arange(0.05, 0.21 + 1e-9, 0.005), 4)


def load_combined():
    """加载并拼接 (cal+test) 的 u₁ / u₂ / errors / labels。"""
    cal = np.load(C.UNC_CAL_NPZ)
    tst = np.load(C.UNC_TEST_NPZ)
    u1     = np.concatenate([cal["u1"],     tst["u1"]])
    u2     = np.concatenate([cal[U2_KEY],   tst[U2_KEY]])
    preds  = np.concatenate([cal["preds"],  tst["preds"]])
    labels = np.concatenate([cal["labels"], tst["labels"]])
    errors = (preds != labels).astype(np.int64)
    return u1, u2, errors, labels


def make_split(n_total, labels, n_cal, seed):
    """分层拆分：从 [0, n_total) 中抽 n_cal 个做 cal，剩下做 test。"""
    idx = np.arange(n_total)
    cal_idx, test_idx = train_test_split(
        idx, train_size=n_cal, stratify=labels, random_state=seed)
    return np.sort(cal_idx), np.sort(test_idx)


def make_internal_split(n_cal, ratio, seed):
    """校准集内 9:1 拆分；返回 split_idx (bool, len=n_cal)，True=10% 部分。"""
    rng = np.random.default_rng(seed)
    n_order = max(1, int(round(n_cal * ratio)))
    perm = rng.permutation(n_cal)
    split_idx = np.zeros(n_cal, dtype=bool)
    split_idx[perm[:n_order]] = True
    return split_idx


def run_one_repeat(u1, u2, errors, labels, n_cal, repeat_seed,
                   grid_u1, grid_u2):
    """跑单次重复实验。"""
    cal_idx, test_idx = make_split(len(u1), labels, n_cal, repeat_seed)
    split_idx = make_internal_split(len(cal_idx), C.CAL_SPLIT_RATIO,
                                    repeat_seed + 99991)

    u1_cal, u2_cal, err_cal = u1[cal_idx], u2[cal_idx], errors[cal_idx]
    u1_tst, u2_tst, err_tst = u1[test_idx], u2[test_idx], errors[test_idx]

    K = len(ALPHA_GRID)
    out = {
        "u1":       {"lam": np.full(K, np.nan),       "cov": np.zeros(K), "rc": np.zeros(K)},
        "u2":       {"lam": np.full(K, np.nan),       "cov": np.zeros(K), "rc": np.zeros(K)},
        "Parallel": {"lam": np.full((K, 2), np.nan),  "cov": np.zeros(K), "rc": np.zeros(K)},
        "Series":   {"lam": np.full((K, 2), np.nan),  "cov": np.zeros(K), "rc": np.zeros(K)},
    }

    for k, alpha in enumerate(ALPHA_GRID):
        # —— 单分数 u₁ ——
        lam = single_score_ltt(u1_cal, err_cal, alpha, C.DELTA, grid_u1)
        if lam is not None:
            out["u1"]["lam"][k] = lam
            cov, rc = empirical_metrics(apply_single(u1_tst, lam), err_tst)
            out["u1"]["cov"][k], out["u1"]["rc"][k] = cov, rc

        # —— 单分数 u₂ ——
        lam = single_score_ltt(u2_cal, err_cal, alpha, C.DELTA, grid_u2)
        if lam is not None:
            out["u2"]["lam"][k] = lam
            cov, rc = empirical_metrics(apply_single(u2_tst, lam), err_tst)
            out["u2"]["cov"][k], out["u2"]["rc"][k] = cov, rc

        # —— 并联 (OR) ——
        lam = two_score_ltt(u1_cal, u2_cal, err_cal, alpha, C.DELTA,
                            grid_u1, grid_u2,
                            mode="or", split_idx=split_idx)
        if lam is not None:
            out["Parallel"]["lam"][k] = lam
            cov, rc = empirical_metrics(apply_parallel(u1_tst, u2_tst, lam), err_tst)
            out["Parallel"]["cov"][k], out["Parallel"]["rc"][k] = cov, rc

        # —— 串联 (AND) ——
        lam = two_score_ltt(u1_cal, u2_cal, err_cal, alpha, C.DELTA,
                            grid_u1, grid_u2,
                            mode="and", split_idx=split_idx)
        if lam is not None:
            out["Series"]["lam"][k] = lam
            cov, rc = empirical_metrics(apply_series(u1_tst, u2_tst, lam), err_tst)
            out["Series"]["cov"][k], out["Series"]["rc"][k] = cov, rc

    return out


def main():
    print(f"=== Loading uncertainties (u₂ = {U2_KEY}) ===")
    u1, u2, errors, labels = load_combined()
    n_total = len(u1)
    print(f"Total combined samples: {n_total}")

    n_cal  = 732
    n_test = n_total - n_cal
    print(f"Each repeat: cal = {n_cal}, test = {n_test}")
    print(f"α grid ({len(ALPHA_GRID)} points): "
          f"{ALPHA_GRID[0]:.3f} → {ALPHA_GRID[-1]:.3f}, step = 0.005")

    grid_u1 = np.linspace(0.01, 0.99, C.GRID_NUM)
    grid_u2 = np.linspace(0.01, 0.99, C.GRID_NUM)

    methods = ["u1", "u2", "Parallel", "Series"]
    K = len(ALPHA_GRID)
    R = C.N_REPEATS

    coverage = {m: np.zeros((R, K)) for m in methods}
    rcond    = {m: np.zeros((R, K)) for m in methods}
    lam_u1   = np.full((R, K), np.nan)
    lam_u2   = np.full((R, K), np.nan)
    lam_par  = np.full((R, K, 2), np.nan)
    lam_ser  = np.full((R, K, 2), np.nan)

    print(f"\n=== Running {R} repeats × {K} α × 4 methods ===")
    for r in tqdm(range(R)):
        out = run_one_repeat(u1, u2, errors, labels, n_cal,
                             repeat_seed=C.SEED + r,
                             grid_u1=grid_u1, grid_u2=grid_u2)
        for m in methods:
            coverage[m][r] = out[m]["cov"]
            rcond[m][r]    = out[m]["rc"]
        lam_u1[r]  = out["u1"]["lam"]
        lam_u2[r]  = out["u2"]["lam"]
        lam_par[r] = out["Parallel"]["lam"]
        lam_ser[r] = out["Series"]["lam"]

    # —— 保存 ——
    save_path = C.OUTPUT_DIR / "ltt_repeats.npz"
    np.savez_compressed(
        save_path,
        alpha_grid=ALPHA_GRID,
        coverage_u1=coverage["u1"],
        coverage_u2=coverage["u2"],
        coverage_parallel=coverage["Parallel"],
        coverage_series=coverage["Series"],
        rcond_u1=rcond["u1"],
        rcond_u2=rcond["u2"],
        rcond_parallel=rcond["Parallel"],
        rcond_series=rcond["Series"],
        lam_u1=lam_u1,
        lam_u2=lam_u2,
        lam_parallel=lam_par,
        lam_series=lam_ser,
    )
    print(f"\nSaved → {save_path.name}")

    # —— 终端摘要：报告的 4 个 α ——
    print("\n=== Quick summary at reported α (mean over repeats) ===")
    print(f"{'method':<10s}{'α':>8s}{'cov %':>10s}{'R_cond %':>12s}{'violate %':>12s}")
    for a in [0.10, 0.125, 0.15, 0.175]:
        k = int(np.argmin(np.abs(ALPHA_GRID - a)))
        for m in methods:
            cov_mean = coverage[m][:, k].mean() * 100
            rc_mean  = rcond[m][:, k].mean() * 100
            viol     = (rcond[m][:, k] > a).mean() * 100
            print(f"{m:<10s}{a:>8.3f}{cov_mean:>10.2f}{rc_mean:>12.2f}{viol:>12.1f}")
        print("-" * 52)


if __name__ == "__main__":
    main()
