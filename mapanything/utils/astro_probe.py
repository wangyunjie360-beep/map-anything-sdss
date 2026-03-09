from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from mapanything.datasets import AstroSDSSPairDataset, AstroSDSSPairDatasetV2
from mapanything.models import init_model


DATASET_EVAL_GLOBALS = {
    "AstroSDSSPairDataset": AstroSDSSPairDataset,
    "AstroSDSSPairDatasetV2": AstroSDSSPairDatasetV2,
}


def ensure_dino_environment(dino_local_repo: str | None = None, dino_local_ckpt: str | None = None) -> None:
    if dino_local_repo:
        os.environ["DINO_LOCAL_REPO"] = dino_local_repo
    if dino_local_ckpt:
        os.environ["DINO_LOCAL_CKPT"] = dino_local_ckpt


def infer_probe_root(experiment_dir: Path) -> Path:
    experiment_dir = experiment_dir.resolve()
    if experiment_dir.parent.name == "experiments":
        astro_root = experiment_dir.parent.parent
    else:
        astro_root = experiment_dir.parent
    return astro_root / "probes" / "redshift_probe"


def load_experiment_config(experiment_dir: Path):
    config_path = experiment_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    return OmegaConf.load(config_path)


def resolve_checkpoint_path(experiment_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()
    candidate = experiment_dir / raw_path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {raw_path}")


def strip_dataset_size_prefix(dataset_expr: str) -> str:
    if "@" not in dataset_expr:
        return dataset_expr.strip()
    _, rhs = dataset_expr.split("@", 1)
    return rhs.strip()


def unwrap_dataset(dataset: Any) -> Any:
    current = dataset
    seen = set()
    while hasattr(current, "dataset") and id(current) not in seen:
        seen.add(id(current))
        current = current.dataset
    return current


def build_dataset_from_cfg(cfg, split: str):
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_expr = str(dataset_cfg.train_dataset if split == "train" else dataset_cfg.test_dataset)
    dataset_expr = strip_dataset_size_prefix(dataset_expr)
    dataset = eval(dataset_expr, DATASET_EVAL_GLOBALS)
    return unwrap_dataset(dataset)


def load_model_for_checkpoint(cfg, checkpoint_path: Path, device: torch.device):
    model = init_model(
        model_str=cfg.model.model_str,
        model_config=cfg.model.model_config,
        torch_hub_force_reload=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    message = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, message


def build_row_lookup(dataset) -> Dict[str, Mapping[str, Any]]:
    if not hasattr(dataset, "entries"):
        raise AttributeError("Dataset does not expose `entries`; cannot build lookup.")
    return {str(row["sample_id"]): row for row in dataset.entries}


def parse_csv_arg(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def mode_to_suffix(mode: str) -> str:
    return mode.replace("-", "_")


def mode_to_inputs(mode: str, image: torch.Tensor, spectrum: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mode == "image-only":
        return image, None
    if mode == "spectrum-only":
        return None, spectrum
    if mode == "both":
        return image, spectrum
    raise ValueError(f"Unsupported mode: {mode}")


def feature_archive_path(output_dir: Path, split: str, mode: str) -> Path:
    return output_dir / f"{split}_{mode_to_suffix(mode)}.npz"


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return _safe_pearson(_average_ranks(x), _average_ranks(y))


def compute_redshift_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> Dict[str, float]:
    z_true = np.asarray(z_true, dtype=np.float64)
    z_pred = np.asarray(z_pred, dtype=np.float64)
    if z_true.shape != z_pred.shape:
        raise ValueError(f"Shape mismatch: {z_true.shape} vs {z_pred.shape}")
    delta = z_pred - z_true
    norm_abs = np.abs(delta) / (1.0 + z_true)
    return {
        "mae_z": float(np.mean(np.abs(delta))),
        "rmse_z": float(np.sqrt(np.mean(np.square(delta)))),
        "median_abs_dz_over_1pz": float(np.median(norm_abs)),
        "catastrophic_outlier_rate": float(np.mean(norm_abs > 0.15)),
        "pearson_corr": _safe_pearson(z_true, z_pred),
        "spearman_corr": _safe_spearman(z_true, z_pred),
    }


def save_json(data: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(rows: Sequence[Mapping[str, Any]], output_path: Path, fieldnames: Sequence[str] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_feature_archive(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def plot_redshift_scatter(z_true: np.ndarray, z_pred: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z_true = np.asarray(z_true, dtype=np.float64)
    z_pred = np.asarray(z_pred, dtype=np.float64)
    combined = np.concatenate([z_true.reshape(-1), z_pred.reshape(-1), np.asarray([0.0], dtype=np.float64)])
    limit = float(max(np.max(combined), 1e-6))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z_true, z_pred, s=18, alpha=0.65, edgecolors="none")
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("z_true")
    ax.set_ylabel("z_pred")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_hist(norm_abs: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norm_abs = np.asarray(norm_abs, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(norm_abs, bins=24, color="#4c72b0", alpha=0.85)
    ax.axvline(0.15, linestyle="--", color="red", linewidth=1.2, label="catastrophic = 0.15")
    ax.set_xlabel("|Δz| / (1 + z_true)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_vs_z(z_true: np.ndarray, norm_abs: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(z_true, norm_abs, s=18, alpha=0.65, edgecolors="none")
    ax.axhline(0.15, linestyle="--", color="red", linewidth=1.2)
    ax.set_xlabel("z_true")
    ax.set_ylabel("|Δz| / (1 + z_true)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_vs_sn(sn_median_r: np.ndarray, norm_abs: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sn_median_r, norm_abs, s=18, alpha=0.65, edgecolors="none")
    ax.axhline(0.15, linestyle="--", color="red", linewidth=1.2)
    ax.set_xlabel("sn_median_r")
    ax.set_ylabel("|Δz| / (1 + z_true)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_bars(summary_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summary_rows)
    if not rows:
        raise ValueError("No summary rows provided.")
    modes = [str(row["mode"]) for row in rows]
    mae = np.asarray([float(row["mae_z"]) for row in rows], dtype=np.float64)
    rmse = np.asarray([float(row["rmse_z"]) for row in rows], dtype=np.float64)
    outlier = np.asarray([float(row["catastrophic_outlier_rate"]) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    series = [mae, rmse, outlier]
    titles = ["MAE(z)", "RMSE(z)", "Outlier Rate"]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    x = np.arange(len(modes))

    for ax, values, title, color in zip(axes, series, titles, colors):
        ax.bar(x, values, color=color, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=15)
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_by_bins(x: np.ndarray, y: np.ndarray, bins: Sequence[float]) -> List[Dict[str, float]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    rows: List[Dict[str, float]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right == bins[-1]:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if not np.any(mask):
            rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": 0,
                    "mean_value": float("nan"),
                    "median_value": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(mask.sum()),
                "mean_value": float(np.mean(y[mask])),
                "median_value": float(np.median(y[mask])),
            }
        )
    return rows


def default_z_bins(z_true: np.ndarray) -> np.ndarray:
    z_true = np.asarray(z_true, dtype=np.float64)
    if z_true.size == 0:
        return np.asarray([0.0, 0.5], dtype=np.float64)
    upper = float(max(0.5, np.ceil(z_true.max() * 10.0) / 10.0))
    return np.linspace(0.0, upper, num=6, dtype=np.float64)


def default_sn_bins(sn_median_r: np.ndarray) -> np.ndarray:
    sn_median_r = np.asarray(sn_median_r, dtype=np.float64)
    if sn_median_r.size == 0:
        return np.asarray([0.0, 10.0], dtype=np.float64)
    quantiles = np.quantile(sn_median_r, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.unique(np.asarray(quantiles, dtype=np.float64))
    if bins.size < 2:
        bins = np.asarray([float(sn_median_r.min()), float(sn_median_r.max()) + 1.0], dtype=np.float64)
    return bins


def checkpoint_label(checkpoint_path: Path) -> str:
    return checkpoint_path.stem.replace("checkpoint-", "")


def tensor_to_numpy_list(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
