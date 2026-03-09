#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard cross-modal retrieval evaluations.")
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "dot"])
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def configs() -> List[Dict[str, str]]:
    return [
        {
            "label": "image_zshared_to_spec_zshared",
            "query_mode": "image-only",
            "gallery_mode": "spectrum-only",
            "query_key": "z_shared",
            "gallery_key": "z_shared",
        },
        {
            "label": "spec_zshared_to_image_zshared",
            "query_mode": "spectrum-only",
            "gallery_mode": "image-only",
            "query_key": "z_shared",
            "gallery_key": "z_shared",
        },
        {
            "label": "image_lshared_to_spec_lshared",
            "query_mode": "image-only",
            "gallery_mode": "spectrum-only",
            "query_key": "latent_shared_proj",
            "gallery_key": "latent_shared_proj",
        },
        {
            "label": "spec_lshared_to_image_lshared",
            "query_mode": "spectrum-only",
            "gallery_mode": "image-only",
            "query_key": "latent_shared_proj",
            "gallery_key": "latent_shared_proj",
        },
        {
            "label": "image_zimg_to_spec_zspec",
            "query_mode": "image-only",
            "gallery_mode": "spectrum-only",
            "query_key": "z_img",
            "gallery_key": "z_spec",
        },
        {
            "label": "spec_zspec_to_image_zimg",
            "query_mode": "spectrum-only",
            "gallery_mode": "image-only",
            "query_key": "z_spec",
            "gallery_key": "z_img",
        },
    ]


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in configs():
        out_dir = output_root / cfg["label"]
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "eval_astro_cross_modal_retrieval.py"),
            "--features-root",
            str(args.features_root.resolve()),
            "--split",
            args.split,
            "--query-mode",
            cfg["query_mode"],
            "--gallery-mode",
            cfg["gallery_mode"],
            "--query-key",
            cfg["query_key"],
            "--gallery-key",
            cfg["gallery_key"],
            "--output-dir",
            str(out_dir),
            "--similarity",
            args.similarity,
        ]
        if args.normalize:
            cmd.append("--normalize")
        else:
            cmd.append("--no-normalize")
        run_cmd(cmd)
        metrics_path = out_dir / "metrics.json"
        payload = json.loads(metrics_path.read_text())
        row = {"label": cfg["label"]}
        row.update(payload)
        rows.append(row)

    summary_path = output_root / "summary.csv"
    fieldnames = [
        "label",
        "split",
        "query_mode",
        "gallery_mode",
        "query_key",
        "gallery_key",
        "similarity",
        "normalize",
        "num_queries",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "mean_rank",
        "median_rank",
        "positive_similarity_mean",
        "negative_similarity_mean",
        "similarity_auc",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"[run] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
