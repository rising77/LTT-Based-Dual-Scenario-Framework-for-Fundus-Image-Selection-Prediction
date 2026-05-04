"""
src/uncertainty.py
==================

提取 EfficientNet-B3 在校准/测试集上的 logits 与特征，并计算论文中所有
u₁ / u₂ 候选不确定性分数（统一归一化到 [0, 1]）：

  u₁          = 1 − max softmax  (predictive entropy 的简化形式)
  u₂_knn5/10/20 = 1 − KNN 一致率  (基于训练集特征库)
  u₂_maha     = Mahalanobis 距离的 rank-percentile（按训练集分布）
  u₂_mcd      = MC Dropout 预测熵 / log C
"""
from __future__ import annotations
import time
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


# ============================ 工具 ============================
def softmax_np(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax (numpy)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@torch.no_grad()
def _extract_logits_features(model, loader, device):
    """在 loader 上前向一遍，返回 (logits, features, preds, labels)。"""
    model.eval()
    L, F_, Y = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, feats = model(x)
        L.append(logits.cpu().numpy())
        F_.append(feats.cpu().numpy())
        Y.append(y.cpu().numpy())
    L = np.concatenate(L, axis=0).astype(np.float32)
    F_ = np.concatenate(F_, axis=0).astype(np.float32)
    Y = np.concatenate(Y, axis=0).astype(np.int64)
    return L, F_, Y


# ============================ u₁ ============================
def compute_u1(logits: np.ndarray) -> np.ndarray:
    return 1.0 - softmax_np(logits).max(axis=-1)


# ============================ u₂_knn ============================
def compute_u2_knn(features_eval: np.ndarray,
                   features_train: np.ndarray,
                   labels_train: np.ndarray,
                   preds_eval: np.ndarray,
                   k: int = 5) -> np.ndarray:
    """
    1 − KNN 一致性：在训练集特征库中查 K 近邻，计算这 K 个邻居中
    与模型预测同标签的比例，用 1 − 该比例作为 u₂。
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(features_train)
    _, idx = nn.kneighbors(features_eval)              # (N_eval, k)
    neigh_labels = labels_train[idx]                    # (N_eval, k)
    consistency = (neigh_labels == preds_eval[:, None]).mean(axis=1)
    return 1.0 - consistency


# ============================ u₂_maha ============================
def compute_u2_mahalanobis(features_eval: np.ndarray,
                           features_train: np.ndarray,
                           labels_train: np.ndarray,
                           preds_eval: np.ndarray,
                           num_classes: int) -> np.ndarray:
    """
    类条件 Mahalanobis 距离：对每个评估样本，取它的预测类的均值与共享协方差，
    计算 (x − μ_c)^T Σ^{-1} (x − μ_c)，再做 rank-percentile 归一化到 [0,1]。
    （共享协方差是为了稳健，论文 §A.4 同时报告了类内协方差版本，趋势一致。）
    """
    means = np.zeros((num_classes, features_train.shape[1]), dtype=np.float64)
    centered = []
    for c in range(num_classes):
        mask = labels_train == c
        if mask.sum() == 0:
            means[c] = features_train.mean(axis=0)
            continue
        means[c] = features_train[mask].mean(axis=0)
        centered.append(features_train[mask] - means[c])
    centered = np.concatenate(centered, axis=0)
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    cov += np.eye(cov.shape[0]) * 1e-4                  # 数值稳定
    cov_inv = np.linalg.inv(cov)

    # 用预测类的均值
    diff = features_eval - means[preds_eval]
    # x^T Σ^{-1} x  按行
    md2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    md = np.sqrt(np.maximum(md2, 0.0))

    # rank-percentile 归一化（按训练集 MD 的分布）
    # 训练集自身的 MD（用真实标签的均值）
    diff_tr = features_train - means[labels_train]
    md2_tr = np.einsum("ij,jk,ik->i", diff_tr, cov_inv, diff_tr)
    md_tr  = np.sqrt(np.maximum(md2_tr, 0.0))
    sorted_tr = np.sort(md_tr)
    pct = np.searchsorted(sorted_tr, md) / max(len(sorted_tr), 1)
    return np.clip(pct, 0.0, 1.0)


# ============================ u₂_mcd ============================
@torch.no_grad()
def compute_u2_mcd(model, loader, device, num_classes: int,
                   T: int = 20) -> np.ndarray:
    """
    MC Dropout 预测熵：T 次随机前向取概率均值后的熵，除以 log(num_classes) 归一化。
    """
    model.eval()
    model.enable_dropout()                              # 仅打开 Dropout
    eps = 1e-12
    entropies = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        prob_acc = torch.zeros(x.size(0), num_classes, device=device)
        for _t in range(T):
            logits, _ = model(x)
            prob_acc += F.softmax(logits, dim=-1)
        p_mean = prob_acc / T
        H = -(p_mean * (p_mean + eps).log()).sum(dim=-1)
        entropies.append(H.cpu().numpy())
    model.eval()                                        # 恢复
    H = np.concatenate(entropies, axis=0)
    return (H / np.log(num_classes)).clip(0.0, 1.0)


# ============================ 一站式 ============================
def compute_all_uncertainties(model, train_loader, eval_loader, *,
                              device, num_classes: int = 5,
                              mcd_T: int = 20) -> Dict[str, np.ndarray]:
    """
    在 eval_loader 上计算 u₁ 与所有 u₂ 候选，并记录耗时（秒）。
    train_loader 仅用于提取训练集特征库（KNN / Mahalanobis 都要用）。
    """
    timings = {}

    # —— 训练集特征库（一次提取，多次复用） ——
    t0 = time.time()
    _, feats_train, labels_train = _extract_logits_features(
        model, train_loader, device)
    timings["build_train_features_sec"] = time.time() - t0

    # —— 评估集 logits + features ——
    t0 = time.time()
    logits_eval, feats_eval, labels_eval = _extract_logits_features(
        model, eval_loader, device)
    preds_eval = logits_eval.argmax(axis=-1)
    timings["forward_eval_sec"] = time.time() - t0

    # —— u₁ ——
    t0 = time.time()
    u1 = compute_u1(logits_eval)
    timings["u1_sec"] = time.time() - t0

    # —— u₂_knn ——
    u2_knn = {}
    for k in [5, 10, 20]:
        t0 = time.time()
        u2_knn[k] = compute_u2_knn(feats_eval, feats_train,
                                   labels_train, preds_eval, k=k)
        timings[f"u2_knn{k}_sec"] = time.time() - t0

    # —— u₂_maha ——
    t0 = time.time()
    u2_maha = compute_u2_mahalanobis(
        feats_eval, feats_train, labels_train, preds_eval,
        num_classes=num_classes)
    timings["u2_maha_sec"] = time.time() - t0

    # —— u₂_mcd ——
    t0 = time.time()
    u2_mcd = compute_u2_mcd(model, eval_loader, device,
                            num_classes=num_classes, T=mcd_T)
    timings["u2_mcd_sec"] = time.time() - t0

    return {
        "logits":  logits_eval,
        "features": feats_eval,
        "preds":   preds_eval,
        "labels":  labels_eval,
        "u1":      u1.astype(np.float32),
        "u2_knn5":  u2_knn[5].astype(np.float32),
        "u2_knn10": u2_knn[10].astype(np.float32),
        "u2_knn20": u2_knn[20].astype(np.float32),
        "u2_maha":  u2_maha.astype(np.float32),
        "u2_mcd":   u2_mcd.astype(np.float32),
        "timings":  timings,
    }
