#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import torch
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mapanything.utils.astro_probe import infer_probe_root, plot_summary_bars, save_csv, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full frozen redshift probe pipeline.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint-final.pth")
    parser.add_argument("--probe-root", type=Path, default=None)
    parser.add_argument("--modes", type=str, default="image-only,spectrum-only,both")
    parser.add_argument("--feature-key", type=str, default="z_shared")
    parser.add_argument("--batch-size-export", type=int, default=16)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dims", type=str, default="256,64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-metadata", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--dino-local-repo", type=str, default=None)
    parser.add_argument("--dino-local-ckpt", type=str, default=None)
    return parser.parse_args()


def parse_modes(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_metrics(metrics_path: Path) -> Dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    row = {
        "mode": payload["mode"],
        "input_key": payload["input_key"],
        "best_epoch": payload["best_epoch"],
        "train_size": payload["train_size"],
        "val_size": payload["val_size"],
    }
    row.update(payload["val_metrics"])
    return row


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    probe_root = args.probe_root.resolve() if args.probe_root else infer_probe_root(experiment_dir)
    features_root = probe_root / "features"
    runs_root = probe_root / "runs"
    probe_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    modes = parse_modes(args.modes)

    if not args.skip_export:
        export_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_astro_probe_features.py"),
            "--experiment-dir",
            str(experiment_dir),
            "--checkpoint",
            args.checkpoint,
            "--output-dir",
            str(features_root),
            "--modes",
            args.modes,
            "--batch-size",
            str(args.batch_size_export),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ]
        if args.max_samples_per_split is not None:
            export_cmd.extend(["--max-samples-per-split", str(args.max_samples_per_split)])
        if args.disable_metadata:
            export_cmd.append("--disable-metadata")
        if args.dino_local_repo:
            export_cmd.extend(["--dino-local-repo", args.dino_local_repo])
        if args.dino_local_ckpt:
            export_cmd.extend(["--dino-local-ckpt", args.dino_local_ckpt])
        run_cmd(export_cmd)

    summary_rows: List[Dict[str, float]] = []
    for mode in modes:
        output_dir = runs_root / mode.replace("-", "_")
        train_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_astro_redshift_probe.py"),
            "--features-root",
            str(features_root),
            "--mode",
            mode,
            "--output-dir",
            str(output_dir),
            "--input-key",
            args.feature_key,
            "--batch-size",
            str(args.batch_size_train),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--dropout",
            str(args.dropout),
            "--hidden-dims",
            args.hidden_dims,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        run_cmd(train_cmd)
        summary_rows.append(load_metrics(output_dir / "metrics.json"))

    save_csv(summary_rows, probe_root / "summary.csv")
    save_json(
        {
            "experiment_dir": str(experiment_dir),
            "checkpoint": args.checkpoint,
            "feature_key": args.feature_key,
            "modes": modes,
            "summary": summary_rows,
        },
        probe_root / "summary.json",
    )
    plot_summary_bars(summary_rows, probe_root / "summary_metrics.png")
    print(f"[run] wrote summary to {probe_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
