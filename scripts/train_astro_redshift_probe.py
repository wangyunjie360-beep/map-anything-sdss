#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mapanything.utils.astro_probe import (
    compute_redshift_metrics,
    default_sn_bins,
    default_z_bins,
    feature_archive_path,
    load_feature_archive,
    plot_error_hist,
    plot_redshift_scatter,
    plot_residual_vs_sn,
    plot_residual_vs_z,
    save_csv,
    save_json,
    summarize_by_bins,
)


@dataclass
class ProbeBatch:
    features: torch.Tensor
    target_log1p_z: torch.Tensor
    z_true: torch.Tensor
    sample_id: torch.Tensor
    sn_median_r: torch.Tensor
    ra: torch.Tensor
    dec: torch.Tensor


def parse_input_keys(raw: str) -> List[str]:
    keys = [item.strip() for item in raw.split(',') if item.strip()]
    if not keys:
        raise ValueError('input key list must not be empty')
    return keys


class RedshiftFeatureDataset(Dataset):
    def __init__(self, archive: Dict[str, np.ndarray], input_keys: List[str]):
        missing = [key for key in input_keys if key not in archive]
        if missing:
            raise KeyError(
                f"Input keys {missing} not found in archive. Available: {sorted(archive.keys())}"
            )
        features = [np.asarray(archive[key], dtype=np.float32) for key in input_keys]
        self.x = torch.from_numpy(np.concatenate(features, axis=1))
        self.target_log1p_z = torch.from_numpy(
            np.asarray(archive["target_log1p_z"], dtype=np.float32)
        ).unsqueeze(-1)
        self.z_true = torch.from_numpy(np.asarray(archive["z"], dtype=np.float32)).unsqueeze(-1)
        self.sample_id = torch.from_numpy(np.asarray(archive["sample_id"], dtype=np.int64))
        self.sn_median_r = torch.from_numpy(
            np.asarray(archive["sn_median_r"], dtype=np.float32)
        ).unsqueeze(-1)
        self.ra = torch.from_numpy(np.asarray(archive["ra"], dtype=np.float32)).unsqueeze(-1)
        self.dec = torch.from_numpy(np.asarray(archive["dec"], dtype=np.float32)).unsqueeze(-1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> ProbeBatch:
        return ProbeBatch(
            features=self.x[idx],
            target_log1p_z=self.target_log1p_z[idx],
            z_true=self.z_true[idx],
            sample_id=self.sample_id[idx],
            sn_median_r=self.sn_median_r[idx],
            ra=self.ra[idx],
            dec=self.dec[idx],
        )


class MLPProbeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearProbeHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-feature redshift probe.")
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument("--train-features", type=Path, default=None)
    parser.add_argument("--val-features", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="both")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-key", type=str, default="z_shared")
    parser.add_argument("--head-type", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dims", type=str, default="256,64")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def parse_hidden_dims(raw: str) -> List[int]:
    dims = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not dims:
        raise ValueError("hidden dims must not be empty for mlp head")
    return dims


def resolve_feature_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.train_features and args.val_features:
        return args.train_features.resolve(), args.val_features.resolve()
    if args.features_root is None:
        raise ValueError("Provide either --features-root or both --train-features/--val-features.")
    train_path = feature_archive_path(args.features_root.resolve(), split="train", mode=args.mode)
    val_path = feature_archive_path(args.features_root.resolve(), split="val", mode=args.mode)
    return train_path, val_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_probe_batches(batch: List[ProbeBatch]) -> ProbeBatch:
    return ProbeBatch(
        features=torch.stack([item.features for item in batch], dim=0),
        target_log1p_z=torch.stack([item.target_log1p_z for item in batch], dim=0),
        z_true=torch.stack([item.z_true for item in batch], dim=0),
        sample_id=torch.stack([item.sample_id for item in batch], dim=0),
        sn_median_r=torch.stack([item.sn_median_r for item in batch], dim=0),
        ra=torch.stack([item.ra for item in batch], dim=0),
        dec=torch.stack([item.dec for item in batch], dim=0),
    )


def build_probe_head(input_dim: int, head_type: str, hidden_dims: List[int], dropout: float) -> nn.Module:
    if head_type == "linear":
        return LinearProbeHead(input_dim=input_dim)
    if head_type == "mlp":
        return MLPProbeHead(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    raise ValueError(f"Unsupported head_type: {head_type}")


def run_epoch(model, loader, optimizer, criterion, device: torch.device, train: bool) -> float:
    model.train(train)
    losses: List[float] = []
    for batch in loader:
        features = batch.features.to(device=device, dtype=torch.float32)
        targets = batch.target_log1p_z.to(device=device, dtype=torch.float32)
        if train:
            optimizer.zero_grad(set_to_none=True)
        preds = model(features)
        loss = criterion(preds, targets)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def predict(model, loader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    outputs: Dict[str, List[np.ndarray]] = {
        "sample_id": [],
        "z_true": [],
        "sn_median_r": [],
        "ra": [],
        "dec": [],
        "pred_log1p_z": [],
    }
    with torch.inference_mode():
        for batch in loader:
            features = batch.features.to(device=device, dtype=torch.float32)
            preds = model(features).detach().cpu().numpy().astype(np.float32).reshape(-1)
            outputs["pred_log1p_z"].append(preds)
            outputs["sample_id"].append(
                batch.sample_id.detach().cpu().numpy().astype(np.int64).reshape(-1)
            )
            outputs["z_true"].append(batch.z_true.detach().cpu().numpy().astype(np.float32).reshape(-1))
            outputs["sn_median_r"].append(
                batch.sn_median_r.detach().cpu().numpy().astype(np.float32).reshape(-1)
            )
            outputs["ra"].append(batch.ra.detach().cpu().numpy().astype(np.float32).reshape(-1))
            outputs["dec"].append(batch.dec.detach().cpu().numpy().astype(np.float32).reshape(-1))
    merged = {key: np.concatenate(parts, axis=0) for key, parts in outputs.items()}
    merged["z_pred"] = np.expm1(merged["pred_log1p_z"]).clip(min=0.0)
    merged["abs_dz"] = np.abs(merged["z_pred"] - merged["z_true"])
    merged["abs_dz_over_1pz"] = merged["abs_dz"] / (1.0 + merged["z_true"])
    return merged


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_path, val_path = resolve_feature_paths(args)
    train_archive = load_feature_archive(train_path)
    val_archive = load_feature_archive(val_path)

    input_keys = parse_input_keys(args.input_key)
    train_dataset = RedshiftFeatureDataset(train_archive, input_keys=input_keys)
    val_dataset = RedshiftFeatureDataset(val_archive, input_keys=input_keys)

    device = torch.device(args.device)
    hidden_dims = parse_hidden_dims(args.hidden_dims) if args.head_type == "mlp" else []
    model = build_probe_head(
        input_dim=train_dataset.x.shape[1],
        head_type=args.head_type,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_probe_batches,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_probe_batches,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_state = None
    best_epoch = -1
    best_mae = float("inf")
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device=device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device=device, train=False)
        val_pred = predict(model, val_loader, device=device)
        val_metrics = compute_redshift_metrics(val_pred["z_true"], val_pred["z_pred"])
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(history_row)
        print(
            f"[probe][{args.mode}][{args.head_type}] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} mae_z={val_metrics['mae_z']:.6f}"
        )

        if val_metrics["mae_z"] < best_mae:
            best_mae = float(val_metrics["mae_z"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[probe][{args.mode}][{args.head_type}] early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training produced no best state.")

    best_model_path = output_dir / "best_probe_head.pt"
    torch.save(
        {
            "state_dict": best_state,
            "mode": args.mode,
            "input_key": args.input_key,
            "input_keys": input_keys,
            "head_type": args.head_type,
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "best_epoch": best_epoch,
            "best_mae_z": best_mae,
            "train_features": str(train_path),
            "val_features": str(val_path),
        },
        best_model_path,
    )

    model.load_state_dict(best_state)
    val_pred = predict(model, val_loader, device=device)
    val_metrics = compute_redshift_metrics(val_pred["z_true"], val_pred["z_pred"])
    train_pred = predict(model, train_loader, device=device)
    train_metrics = compute_redshift_metrics(train_pred["z_true"], train_pred["z_pred"])

    predictions_rows = []
    for idx in range(val_pred["sample_id"].shape[0]):
        predictions_rows.append(
            {
                "sample_id": int(val_pred["sample_id"][idx]),
                "z_true": float(val_pred["z_true"][idx]),
                "z_pred": float(val_pred["z_pred"][idx]),
                "pred_log1p_z": float(val_pred["pred_log1p_z"][idx]),
                "sn_median_r": float(val_pred["sn_median_r"][idx]),
                "ra": float(val_pred["ra"][idx]),
                "dec": float(val_pred["dec"][idx]),
                "abs_dz": float(val_pred["abs_dz"][idx]),
                "abs_dz_over_1pz": float(val_pred["abs_dz_over_1pz"][idx]),
            }
        )

    z_bin_summary = summarize_by_bins(
        val_pred["z_true"],
        val_pred["abs_dz_over_1pz"],
        bins=default_z_bins(val_pred["z_true"]),
    )
    sn_bin_summary = summarize_by_bins(
        val_pred["sn_median_r"],
        val_pred["abs_dz_over_1pz"],
        bins=default_sn_bins(val_pred["sn_median_r"]),
    )

    metrics_payload = {
        "mode": args.mode,
        "input_key": args.input_key,
        "input_keys": input_keys,
        "head_type": args.head_type,
        "best_epoch": best_epoch,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_features": str(train_path),
        "val_features": str(val_path),
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "z_bin_summary": z_bin_summary,
        "sn_bin_summary": sn_bin_summary,
    }

    save_json(metrics_payload, output_dir / "metrics.json")
    save_json({"history": history}, output_dir / "history.json")
    save_csv(predictions_rows, output_dir / "predictions.csv")
    save_csv(history, output_dir / "history.csv")
    save_csv(z_bin_summary, output_dir / "z_bins.csv")
    save_csv(sn_bin_summary, output_dir / "sn_bins.csv")

    plot_redshift_scatter(
        val_pred["z_true"],
        val_pred["z_pred"],
        output_dir / "scatter_z_true_vs_pred.png",
        title=f"Redshift Probe ({args.mode}, {args.input_key}, {args.head_type})",
    )
    plot_error_hist(
        val_pred["abs_dz_over_1pz"],
        output_dir / "residual_hist.png",
        title=f"Normalized Error Histogram ({args.mode}, {args.input_key}, {args.head_type})",
    )
    plot_residual_vs_z(
        val_pred["z_true"],
        val_pred["abs_dz_over_1pz"],
        output_dir / "residual_vs_z.png",
        title=f"Normalized Error vs z ({args.mode}, {args.input_key}, {args.head_type})",
    )
    plot_residual_vs_sn(
        val_pred["sn_median_r"],
        val_pred["abs_dz_over_1pz"],
        output_dir / "residual_vs_sn_median_r.png",
        title=f"Normalized Error vs S/N ({args.mode}, {args.input_key}, {args.head_type})",
    )

    history_epochs = np.asarray([row["epoch"] for row in history], dtype=np.int64)
    history_train_loss = np.asarray([row["train_loss"] for row in history], dtype=np.float64)
    history_val_loss = np.asarray([row["val_loss"] for row in history], dtype=np.float64)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history_epochs, history_train_loss, label="train_loss", linewidth=1.5)
    ax.plot(history_epochs, history_val_loss, label="val_loss", linewidth=1.5)
    ax.axvline(best_epoch, linestyle="--", color="black", linewidth=1.1, label="best_epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Training Curve ({args.mode}, {args.input_key}, {args.head_type})")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "train_curve.png", dpi=180)
    plt.close(fig)

    print(f"[probe][{args.mode}][{args.head_type}] outputs written to {output_dir}")


if __name__ == "__main__":
    main()
