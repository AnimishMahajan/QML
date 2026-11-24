"""
Unified Evaluation Pipeline (Section 9):
One evaluation path for all arms (baseline, ablation, proposed methods).
Computes cross_split, leave_one_out, Recall@K, mAP, mAP@R, kNN, few-shot,
cross-condition gap, open-set AUROC, and embedding geometry.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from ..quantum.distances import pairwise_quantum_similarity_matrix


def compute_retrieval_metrics(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_sources: Optional[np.ndarray] = None,
    gallery_sources: Optional[np.ndarray] = None,
    exclude_same_source: bool = False,
    top_k_list: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Computes Recall@K, mAP (full ranking), and mAP@R (R-precision truncation).
    similarity_matrix: (N_query, N_gallery) higher is closer.
    """
    N_q, N_g = similarity_matrix.shape
    r_at_k = {f"Recall@{k}": [] for k in top_k_list}
    ap_list = []
    ap_r_list = []

    for i in range(N_q):
        q_label = query_labels[i]
        sims = similarity_matrix[i].copy()

        # Mask same source frame if requested (leave_one_out F13)
        if exclude_same_source and query_sources is not None and gallery_sources is not None:
            same_frame = (gallery_sources == query_sources[i])
            sims[same_frame] = -np.inf

        # Sorted gallery indices in descending similarity
        ranked_idx = np.argsort(-sims)
        ranked_labels = gallery_labels[ranked_idx]
        ranked_sims = sims[ranked_idx]

        # Valid items (exclude -inf)
        valid = ranked_sims > -np.inf
        ranked_labels = ranked_labels[valid]
        if len(ranked_labels) == 0:
            continue

        binary_hits = (ranked_labels == q_label).astype(np.int32)
        n_relevant = int(np.sum(binary_hits))
        if n_relevant == 0:
            continue

        # Recall@K
        for k in top_k_list:
            hit = int(np.any(binary_hits[:k]))
            r_at_k[f"Recall@{k}"].append(hit)

        # Average Precision (full ranking)
        cum_hits = np.cumsum(binary_hits)
        precisions = cum_hits / (np.arange(len(binary_hits)) + 1)
        ap = np.sum(precisions * binary_hits) / float(n_relevant)
        ap_list.append(ap)

        # mAP@R (truncated at R = n_relevant)
        r_cutoff = min(n_relevant, len(binary_hits))
        prec_r = cum_hits[:r_cutoff] / (np.arange(r_cutoff) + 1)
        ap_r = np.sum(prec_r * binary_hits[:r_cutoff]) / float(n_relevant)
        ap_r_list.append(ap_r)

    metrics = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in r_at_k.items()}
    metrics["mAP"] = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0
    metrics["mAP@R"] = float(np.mean(ap_r_list)) if len(ap_r_list) > 0 else 0.0
    return metrics


def evaluate_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k_list: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """Evaluates nearest neighbour recognition accuracy."""
    results = {}
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_features, train_labels)
        acc = float(knn.score(test_features, test_labels))
        results[f"knn_acc_k{k}"] = acc
    return results


def evaluate_few_shot(
    features: np.ndarray,
    labels: np.ndarray,
    n_way: int = 5,
    k_shots: Tuple[int, ...] = (1, 3, 5),
    n_episodes: int = 20,
    n_queries: int = 15,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluates few-shot recognition episodes."""
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(labels)
    results = {}

    for k in k_shots:
        episode_accs = []
        for _ in range(n_episodes):
            support_feats = []
            support_lbls = []
            query_feats = []
            query_lbls = []

            for c in unique_classes:
                c_idx = np.where(labels == c)[0]
                if len(c_idx) < (k + 1):
                    continue
                chosen = rng.choice(c_idx, size=min(len(c_idx), k + n_queries), replace=False)
                sup = chosen[:k]
                qry = chosen[k:]
                support_feats.append(features[sup])
                support_lbls.append(labels[sup])
                query_feats.append(features[qry])
                query_lbls.append(labels[qry])

            if len(support_feats) < 2 or len(query_feats) < 2:
                continue

            sup_X = np.concatenate(support_feats, axis=0)
            sup_y = np.concatenate(support_lbls, axis=0)
            qry_X = np.concatenate(query_feats, axis=0)
            qry_y = np.concatenate(query_lbls, axis=0)

            # Class prototype nearest neighbour
            # normalize prototypes
            prototypes = []
            for c in unique_classes:
                c_mask = (sup_y == c)
                if c_mask.sum() > 0:
                    proto = np.mean(sup_X[c_mask], axis=0)
                    proto = proto / (np.linalg.norm(proto) + 1e-12)
                    prototypes.append(proto)
            proto_mat = np.stack(prototypes)
            qry_norm = qry_X / (np.linalg.norm(qry_X, axis=1, keepdims=True) + 1e-12)

            sims = np.dot(qry_norm, proto_mat.T)
            preds = np.argmax(sims, axis=1)
            # map pred index back to class
            acc = float(np.mean(preds == qry_y))
            episode_accs.append(acc)

        results[f"few_shot_{k}shot"] = {
            "mean": float(np.mean(episode_accs)) if episode_accs else 0.0,
            "std": float(np.std(episode_accs)) if episode_accs else 0.0,
        }
    return results


def evaluate_embedding_geometry(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Computes silhouette, intra-class and inter-class distance."""
    # features: (N, d) L2 normalized
    dist = 1.0 - np.dot(features, features.T)
    N = len(labels)
    same_class = (labels[:, None] == labels[None, :])
    diag = np.eye(N, dtype=bool)

    intra_dists = dist[same_class & ~diag]
    inter_dists = dist[~same_class]

    mean_intra = float(np.mean(intra_dists)) if len(intra_dists) > 0 else 0.0
    mean_inter = float(np.mean(inter_dists)) if len(inter_dists) > 0 else 0.0
    sep_ratio = float(mean_inter / (mean_intra + 1e-12))

    try:
        sil = float(silhouette_score(features, labels, metric="cosine"))
    except Exception:
        sil = 0.0

    return {
        "intra_dist": mean_intra,
        "inter_dist": mean_inter,
        "separation_ratio": sep_ratio,
        "silhouette": sil,
    }
