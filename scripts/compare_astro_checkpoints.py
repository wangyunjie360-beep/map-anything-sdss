#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mapanything.datasets import AstroSDSSPairDataset
from mapanything.models import init_model
from mapanything.utils.astro_inference import predict_missing_modality


@dataclass
class SampleBundle:
    index: int
    sample_id: str
    image_norm: torch.Tensor
    spectrum_norm: torch.Tensor
    spec_valid_mask: torch.Tensor
    metadata: torch.Tensor | None
    image_denorm: np.ndarray
    spectrum_denorm: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AstroMapAnything checkpoints on the same validation samples.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment directory containing .hydra/config.yaml and checkpoints.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=["checkpoint-best.pth", "checkpoint-final.pth", "checkpoint-120.pth"],
        help="Checkpoint paths or filenames relative to --experiment-dir.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels for checkpoints.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of random validation samples to visualize.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit validation-set indices to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and metric summaries.",
    )
    parser.add_argument(
        "--peak-quantile",
        type=float,
        default=0.90,
        help="Quantile used for peak metrics.",
    )
    parser.add_argument(
        "--rgb-bands",
        type=str,
        default="3,2,1",
        help="Comma-separated image channel indices used to form RGB for plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for saved figures.",
    )
    parser.add_argument(
        "--dino-local-repo",
        type=str,
        default=None,
        help="Optional override for DINO_LOCAL_REPO.",
    )
    parser.add_argument(
        "--dino-local-ckpt",
        type=str,
        default=None,
        help="Optional override for DINO_LOCAL_CKPT.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(experiment_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()
    candidate = experiment_dir / raw_path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {raw_path}")


def ensure_environment(args: argparse.Namespace) -> None:
    if args.dino_local_repo:
        os.environ["DINO_LOCAL_REPO"] = args.dino_local_repo
    if args.dino_local_ckpt:
        os.environ["DINO_LOCAL_CKPT"] = args.dino_local_ckpt


def build_val_dataset(cfg) -> AstroSDSSPairDataset:
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_expr = str(dataset_cfg.test_dataset)
    dataset = eval(dataset_expr, {"AstroSDSSPairDataset": AstroSDSSPairDataset})
    if hasattr(dataset, "dataset"):
        return dataset.dataset
    return dataset


def choose_sample_indices(
    dataset: AstroSDSSPairDataset,
    num_samples: int,
    sample_indices: Sequence[int] | None,
    seed: int,
) -> List[int]:
    if sample_indices:
        selected = sorted({int(idx) for idx in sample_indices})
    else:
        rng = np.random.default_rng(seed)
        total = min(int(num_samples), len(dataset))
        selected = sorted(rng.choice(len(dataset), size=total, replace=False).tolist())
    for idx in selected:
        if idx < 0 or idx >= len(dataset):
            raise IndexError(f"Sample index out of range: {idx}")
    return selected


def denormalize_image(image_norm: torch.Tensor, dataset: AstroSDSSPairDataset) -> np.ndarray:
    image = image_norm.detach().cpu().numpy()
    return image * dataset.image_std[:, None, None] + dataset.image_mean[:, None, None]


def denormalize_spectrum(spectrum_norm: torch.Tensor, dataset: AstroSDSSPairDataset) -> np.ndarray:
    spectrum = spectrum_norm.detach().cpu().numpy()
    return spectrum * dataset.spec_std[:, None] + dataset.spec_mean[:, None]


def to_rgb(image_5ch: np.ndarray, rgb_bands: Sequence[int]) -> np.ndarray:
    rgb = image_5ch[list(rgb_bands)].transpose(1, 2, 0)
    lo = np.percentile(rgb, 1.0)
    hi = np.percentile(rgb, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return rgb


def expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < target.ndim:
        expanded = expanded.unsqueeze(0)
    return expanded.expand_as(target)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = expand_mask(mask, value).to(dtype=value.dtype, device=value.device)
    denom = expanded_mask.sum().clamp_min(1.0)
    return (value * expanded_mask).sum() / denom


def masked_pearson(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = expand_mask(mask, pred).to(dtype=pred.dtype, device=pred.device)
    denom = expanded_mask.sum().clamp_min(1.0)
    pred_mean = (pred * expanded_mask).sum() / denom
    gt_mean = (gt * expanded_mask).sum() / denom
    pred_centered = (pred - pred_mean) * expanded_mask
    gt_centered = (gt - gt_mean) * expanded_mask
    cov = (pred_centered * gt_centered).sum()
    var_pred = (pred_centered * pred_centered).sum().clamp_min(1e-6)
    var_gt = (gt_centered * gt_centered).sum().clamp_min(1e-6)
    return cov / torch.sqrt(var_pred * var_gt + 1e-6)


def spec_grad_abs(spec: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(spec)
    grad[..., 1:] = (spec[..., 1:] - spec[..., :-1]).abs()
    return grad


def peak_recall_at_k(
    gt_spec: torch.Tensor,
    pred_spec: torch.Tensor,
    valid_mask: torch.Tensor,
    quantile: float,
) -> float:
    valid_idx = torch.where(valid_mask)[0]
    if valid_idx.numel() < 4:
        return float("nan")
    gt_grad = spec_grad_abs(gt_spec).mean(dim=0)
    pred_grad = spec_grad_abs(pred_spec).mean(dim=0)
    k = max(1, int(round((1.0 - quantile) * float(valid_idx.numel()))))
    k = min(k, int(valid_idx.numel()))
    gt_top = torch.topk(gt_grad[valid_idx], k=k, largest=True).indices
    pred_top = torch.topk(pred_grad[valid_idx], k=k, largest=True).indices
    gt_bins = valid_idx[gt_top]
    pred_bins = valid_idx[pred_top]
    intersection = torch.isin(pred_bins, gt_bins).sum().float()
    return float((intersection / float(k)).item())


def compute_metrics(
    pred_spec: torch.Tensor,
    gt_spec: torch.Tensor,
    valid_mask: torch.Tensor,
    peak_quantile: float,
) -> Dict[str, float]:
    grad_gt = spec_grad_abs(gt_spec)
    valid_expanded = valid_mask.unsqueeze(0).expand_as(grad_gt)
    masked_grad = torch.where(valid_expanded, grad_gt, torch.full_like(grad_gt, float("nan")))
    peak_threshold = torch.nanquantile(masked_grad, q=peak_quantile, dim=-1, keepdim=True)
    peak_mask = valid_expanded & (grad_gt >= peak_threshold)

    delta_pred = pred_spec[..., 1:] - pred_spec[..., :-1]
    delta_gt = gt_spec[..., 1:] - gt_spec[..., :-1]
    delta_valid = valid_mask[1:] & valid_mask[:-1]

    return {
        "spec_mae": float(masked_mean((pred_spec - gt_spec).abs(), valid_mask).item()),
        "spec_peak_mae": float(masked_mean((pred_spec - gt_spec).abs(), peak_mask).item()),
        "spec_grad_mae": float(masked_mean((delta_pred - delta_gt).abs(), delta_valid).item()),
        "spec_corr": float(masked_pearson(pred_spec, gt_spec, valid_mask).item()),
        "spec_peak_recall": peak_recall_at_k(gt_spec, pred_spec, valid_mask, peak_quantile),
    }


def load_samples(dataset: AstroSDSSPairDataset, indices: Sequence[int]) -> List[SampleBundle]:
    samples: List[SampleBundle] = []
    for index in indices:
        view_image, view_spectrum = dataset[index]
        metadata = view_image["meta_cond"] if bool(view_image["meta_valid"].item()) else None
        sample_id = str(dataset.entries[index]["sample_id"])
        samples.append(
            SampleBundle(
                index=index,
                sample_id=sample_id,
                image_norm=view_image["img_astro"].clone(),
                spectrum_norm=view_spectrum["spec_astro"].clone(),
                spec_valid_mask=view_spectrum["spec_valid_mask"].clone(),
                metadata=metadata.clone() if metadata is not None else None,
                image_denorm=denormalize_image(view_image["img_astro"], dataset),
                spectrum_denorm=denormalize_spectrum(view_spectrum["spec_astro"], dataset),
            )
        )
    return samples


def checkpoint_label(path: Path) -> str:
    return path.stem.replace("checkpoint-", "")


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


def summarize_metrics(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["checkpoint_label"]), []).append(record)
    for label, group in grouped.items():
        summary[label] = {
            key: float(np.nanmean([float(item[key]) for item in group]))
            for key in ["spec_mae", "spec_peak_mae", "spec_grad_mae", "spec_peak_recall", "spec_corr"]
        }
    return summary


def save_metrics_csv(records: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "checkpoint_label",
        "checkpoint_path",
        "sample_index",
        "sample_id",
        "spec_mae",
        "spec_peak_mae",
        "spec_grad_mae",
        "spec_peak_recall",
        "spec_corr",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fieldnames})


def plot_summary(summary: Dict[str, Dict[str, float]], output_path: Path, dpi: int) -> None:
    labels = list(summary.keys())
    metric_keys = ["spec_mae", "spec_peak_mae", "spec_grad_mae", "spec_peak_recall", "spec_corr"]
    titles = {
        "spec_mae": "Spec MAE",
        "spec_peak_mae": "Peak MAE",
        "spec_grad_mae": "Grad MAE",
        "spec_peak_recall": "Peak Recall",
        "spec_corr": "Spec Corr",
    }
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(16, 3.8), constrained_layout=True)
    for axis, metric_key in zip(axes, metric_keys):
        values = [summary[label][metric_key] for label in labels]
        axis.bar(np.arange(len(labels)), values, color=plt.cm.tab10.colors[: len(labels)])
        axis.set_title(titles[metric_key])
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        for xpos, value in enumerate(values):
            axis.text(xpos, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_sample(
    sample: SampleBundle,
    predictions_denorm: Dict[str, np.ndarray],
    metrics_by_checkpoint: Dict[str, Dict[str, float]],
    output_path: Path,
    rgb_bands: Sequence[int],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(14, 12),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.0, 1.0, 1.0]},
    )

    rgb = to_rgb(sample.image_denorm, rgb_bands)
    axes[0].imshow(rgb)
    axes[0].set_title(f"Sample {sample.sample_id} (val idx={sample.index})")
    axes[0].axis("off")

    metrics_text = []
    for label, values in metrics_by_checkpoint.items():
        metrics_text.append(
            f"{label}: mae={values['spec_mae']:.4f}, peak={values['spec_peak_mae']:.4f}, "
            f"grad={values['spec_grad_mae']:.4f}, recall={values['spec_peak_recall']:.4f}, corr={values['spec_corr']:.4f}"
        )
    axes[0].text(
        1.01,
        0.98,
        "\n".join(metrics_text),
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )

    valid_mask = sample.spec_valid_mask.detach().cpu().numpy().astype(bool)
    x_axis = np.arange(sample.spectrum_denorm.shape[-1])
    colors = plt.cm.tab10.colors

    for channel_idx in range(4):
        axis = axes[channel_idx + 1]
        gt_channel = np.where(valid_mask, sample.spectrum_denorm[channel_idx], np.nan)
        axis.plot(x_axis, gt_channel, color="black", linewidth=1.6, label="gt")
        for pred_idx, (label, pred_spec) in enumerate(predictions_denorm.items()):
            pred_channel = np.where(valid_mask, pred_spec[channel_idx], np.nan)
            axis.plot(
                x_axis,
                pred_channel,
                linewidth=1.1,
                color=colors[pred_idx % len(colors)],
                label=label,
            )
        axis.set_ylabel(f"spec[{channel_idx}]")
        axis.grid(alpha=0.18)
        if channel_idx == 0:
            axis.legend(loc="upper right", ncol=min(4, len(predictions_denorm) + 1))
        if channel_idx == 3:
            axis.set_xlabel("spectral bin")

    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ensure_environment(args)

    experiment_dir = args.experiment_dir.resolve()
    if args.labels and len(args.labels) != len(args.checkpoints):
        raise ValueError("--labels must match --checkpoints length.")

    config_path = experiment_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")

    cfg = OmegaConf.load(config_path)
    dataset = build_val_dataset(cfg)
    sample_indices = choose_sample_indices(dataset, args.num_samples, args.sample_indices, args.seed)
    samples = load_samples(dataset, sample_indices)

    output_dir = args.output_dir.resolve() if args.output_dir else (experiment_dir / "checkpoint_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [resolve_checkpoint_path(experiment_dir, raw_path) for raw_path in args.checkpoints]
    checkpoint_labels = list(args.labels) if args.labels else [checkpoint_label(path) for path in checkpoint_paths]
    rgb_bands = tuple(int(item.strip()) for item in args.rgb_bands.split(","))
    if len(rgb_bands) != 3:
        raise ValueError("--rgb-bands must contain exactly 3 channel indices.")

    device = torch.device(args.device)
    records: List[Dict[str, object]] = []
    predictions_for_plot: Dict[str, Dict[str, np.ndarray]] = {sample.sample_id: {} for sample in samples}
    load_report: Dict[str, Dict[str, object]] = {}

    for label, checkpoint_path in zip(checkpoint_labels, checkpoint_paths):
        print(f"[compare] loading {label}: {checkpoint_path}")
        model, load_message = load_model_for_checkpoint(cfg, checkpoint_path, device)
        load_report[label] = {
            "checkpoint_path": str(checkpoint_path),
            "missing_keys": list(load_message.missing_keys),
            "unexpected_keys": list(load_message.unexpected_keys),
        }

        for sample in samples:
            image = sample.image_norm.unsqueeze(0).to(device=device, dtype=torch.float32)
            metadata = None
            if sample.metadata is not None:
                metadata = sample.metadata.unsqueeze(0).to(device=device, dtype=torch.float32)

            with torch.inference_mode():
                pred = predict_missing_modality(
                    model,
                    image=image,
                    spectrum=None,
                    metadata=metadata,
                    return_self_recon=False,
                )

            pred_spec_norm = pred["pred_spectrum"][0].detach().cpu()
            metrics = compute_metrics(
                pred_spec=pred_spec_norm,
                gt_spec=sample.spectrum_norm,
                valid_mask=sample.spec_valid_mask,
                peak_quantile=args.peak_quantile,
            )
            pred_spec_denorm = denormalize_spectrum(pred_spec_norm, dataset)
            predictions_for_plot[sample.sample_id][label] = pred_spec_denorm

            records.append(
                {
                    "checkpoint_label": label,
                    "checkpoint_path": str(checkpoint_path),
                    "sample_index": sample.index,
                    "sample_id": sample.sample_id,
                    **metrics,
                }
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_metrics(records)

    selected_samples_path = output_dir / "selected_samples.json"
    with selected_samples_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_dir": str(experiment_dir),
                "sample_indices": sample_indices,
                "sample_ids": [sample.sample_id for sample in samples],
                "checkpoint_labels": checkpoint_labels,
            },
            handle,
            indent=2,
        )

    metrics_csv_path = output_dir / "sample_metrics.csv"
    save_metrics_csv(records, metrics_csv_path)

    summary_json_path = output_dir / "summary_metrics.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "load_report": load_report,
            },
            handle,
            indent=2,
        )

    summary_png_path = output_dir / "summary_metrics.png"
    plot_summary(summary, summary_png_path, args.dpi)

    for sample in samples:
        per_checkpoint_metrics = {
            str(record["checkpoint_label"]): {
                key: float(record[key])
                for key in ["spec_mae", "spec_peak_mae", "spec_grad_mae", "spec_peak_recall", "spec_corr"]
            }
            for record in records
            if str(record["sample_id"]) == sample.sample_id
        }
        sample_png_path = output_dir / f"sample_idx{sample.index:03d}_id{sample.sample_id}.png"
        plot_sample(
            sample=sample,
            predictions_denorm=predictions_for_plot[sample.sample_id],
            metrics_by_checkpoint=per_checkpoint_metrics,
            output_path=sample_png_path,
            rgb_bands=rgb_bands,
            dpi=args.dpi,
        )

    print(f"[compare] wrote outputs to: {output_dir}")
    print(f"[compare] summary json: {summary_json_path}")
    print(f"[compare] sample csv:   {metrics_csv_path}")


if __name__ == "__main__":
    main()
