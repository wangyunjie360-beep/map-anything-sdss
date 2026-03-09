#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mapanything.utils.astro_probe import load_feature_archive, save_csv, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cross-modal astronomical retrieval.")
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--query-mode", type=str, required=True)
    parser.add_argument("--gallery-mode", type=str, required=True)
    parser.add_argument("--query-key", type=str, required=True)
    parser.add_argument("--gallery-key", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "dot"])
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--heatmap-max-samples", type=int, default=64)
    return parser.parse_args()


def parse_keys(raw: str) -> List[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("feature key list must not be empty")
    return keys


def archive_path(features_root: Path, split: str, mode: str) -> Path:
    return features_root / f"{split}_{mode.replace('-', '_')}.npz"


def extract_features(archive: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    arrays = []
    missing = [key for key in keys if key not in archive]
    if missing:
        raise KeyError(f"Missing keys {missing}; available={sorted(archive.keys())}")
    for key in keys:
        arrays.append(np.asarray(archive[key], dtype=np.float32))
    return np.concatenate(arrays, axis=1)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def roc_auc_from_pos_neg(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")
    combined = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([
        np.ones(pos_scores.shape[0], dtype=np.int64),
        np.zeros(neg_scores.shape[0], dtype=np.int64),
    ])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_vals = combined[order]
    start = 0
    while start < combined.size:
        end = start + 1
        while end < combined.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    n_pos = float(pos_scores.shape[0])
    n_neg = float(neg_scores.shape[0])
    rank_sum_pos = float(ranks[labels == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def compute_retrieval(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    similarity: str,
    normalize: bool,
) -> tuple[np.ndarray, List[Dict[str, float]], Dict[str, float]]:
    query = np.asarray(query_features, dtype=np.float32)
    gallery = np.asarray(gallery_features, dtype=np.float32)
    if normalize or similarity == "cosine":
        query = l2_normalize(query)
        gallery = l2_normalize(gallery)
    sim = query @ gallery.T

    gallery_pos = {int(sample_id): idx for idx, sample_id in enumerate(gallery_ids.tolist())}
    if len(gallery_pos) != len(gallery_ids):
        raise ValueError("Gallery sample IDs are not unique.")

    ranks: List[int] = []
    per_query_rows: List[Dict[str, float]] = []
    positive_scores: List[float] = []
    negative_scores: List[float] = []

    for query_idx, sample_id in enumerate(query_ids.tolist()):
        sample_id = int(sample_id)
        if sample_id not in gallery_pos:
            raise KeyError(f"Sample id {sample_id} missing from gallery set.")
        target_idx = gallery_pos[sample_id]
        row_scores = sim[query_idx]
        order = np.argsort(-row_scores)
        rank = int(np.where(order == target_idx)[0][0]) + 1
        ranks.append(rank)
        target_score = float(row_scores[target_idx])
        positive_scores.append(target_score)
        negative_scores.extend(row_scores[np.arange(row_scores.shape[0]) != target_idx].tolist())
        topk = order[:10]
        per_query_rows.append(
            {
                "query_sample_id": sample_id,
                "target_gallery_sample_id": int(gallery_ids[target_idx]),
                "positive_rank": rank,
                "positive_similarity": target_score,
                "top1_gallery_sample_id": int(gallery_ids[topk[0]]),
                "top1_similarity": float(row_scores[topk[0]]),
                "top5_gallery_sample_ids": "|".join(str(int(gallery_ids[idx])) for idx in topk[:5]),
                "top5_similarities": "|".join(f"{float(row_scores[idx]):.6f}" for idx in topk[:5]),
            }
        )

    ranks_arr = np.asarray(ranks, dtype=np.int64)
    metrics = {
        "num_queries": int(query_ids.shape[0]),
        "recall_at_1": float(np.mean(ranks_arr <= 1)),
        "recall_at_5": float(np.mean(ranks_arr <= 5)),
        "recall_at_10": float(np.mean(ranks_arr <= 10)),
        "mrr": float(np.mean(1.0 / ranks_arr.astype(np.float64))),
        "mean_rank": float(np.mean(ranks_arr.astype(np.float64))),
        "median_rank": float(np.median(ranks_arr.astype(np.float64))),
        "positive_similarity_mean": float(np.mean(positive_scores)),
        "negative_similarity_mean": float(np.mean(negative_scores)),
        "similarity_auc": roc_auc_from_pos_neg(np.asarray(positive_scores), np.asarray(negative_scores)),
    }
    return sim, per_query_rows, metrics


def plot_similarity_heatmap(similarity_matrix: np.ndarray, output_path: Path, max_samples: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(similarity_matrix, dtype=np.float32)
    take = min(int(max_samples), matrix.shape[0], matrix.shape[1])
    matrix = matrix[:take, :take]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("gallery index")
    ax.set_ylabel("query index")
    ax.set_title("Cross-modal similarity heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_similarity_hist(pos_scores: np.ndarray, neg_scores: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(neg_scores, bins=40, alpha=0.65, label="negative", color="#4c72b0")
    ax.hist(pos_scores, bins=30, alpha=0.75, label="positive", color="#c44e52")
    ax.set_xlabel("similarity")
    ax.set_ylabel("count")
    ax.set_title("Positive vs negative similarity")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rank_hist(ranks: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(1, int(ranks.max()) + 2)
    ax.hist(ranks, bins=bins, color="#55a868", alpha=0.85, align="left", rwidth=0.85)
    ax.set_xlabel("positive rank")
    ax.set_ylabel("count")
    ax.set_title("Positive rank histogram")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    gallery_key = args.gallery_key or args.query_key
    query_keys = parse_keys(args.query_key)
    gallery_keys = parse_keys(gallery_key)

    query_archive = load_feature_archive(archive_path(args.features_root, args.split, args.query_mode))
    gallery_archive = load_feature_archive(archive_path(args.features_root, args.split, args.gallery_mode))

    query_ids = np.asarray(query_archive["sample_id"], dtype=np.int64)
    gallery_ids = np.asarray(gallery_archive["sample_id"], dtype=np.int64)

    query_features = extract_features(query_archive, query_keys)
    gallery_features = extract_features(gallery_archive, gallery_keys)

    similarity_matrix, per_query_rows, metrics = compute_retrieval(
        query_features=query_features,
        gallery_features=gallery_features,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        similarity=args.similarity,
        normalize=args.normalize,
    )

    ranks = np.asarray([row["positive_rank"] for row in per_query_rows], dtype=np.int64)
    positive_scores = np.asarray([row["positive_similarity"] for row in per_query_rows], dtype=np.float32)
    negative_scores = []
    gallery_pos = {int(sample_id): idx for idx, sample_id in enumerate(gallery_ids.tolist())}
    for row_idx, sample_id in enumerate(query_ids.tolist()):
        target = gallery_pos[int(sample_id)]
        row = similarity_matrix[row_idx]
        negative_scores.extend(row[np.arange(row.shape[0]) != target].tolist())
    negative_scores = np.asarray(negative_scores, dtype=np.float32)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "split": args.split,
        "query_mode": args.query_mode,
        "gallery_mode": args.gallery_mode,
        "query_key": args.query_key,
        "gallery_key": gallery_key,
        "query_keys": query_keys,
        "gallery_keys": gallery_keys,
        "similarity": args.similarity,
        "normalize": bool(args.normalize),
        **metrics,
    }
    save_json(payload, output_dir / "metrics.json")
    save_csv(per_query_rows, output_dir / "per_query.csv")
    np.savez_compressed(output_dir / "similarity_matrix.npz", similarity=similarity_matrix)
    plot_similarity_heatmap(similarity_matrix, output_dir / "similarity_heatmap.png", args.heatmap_max_samples)
    plot_similarity_hist(positive_scores, negative_scores, output_dir / "similarity_hist.png")
    plot_rank_hist(ranks, output_dir / "rank_hist.png")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
