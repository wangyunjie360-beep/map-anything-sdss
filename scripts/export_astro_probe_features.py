#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mapanything.utils.astro_inference import extract_backbone_features
from mapanything.utils.astro_probe import (
    build_dataset_from_cfg,
    build_row_lookup,
    checkpoint_label,
    ensure_dino_environment,
    feature_archive_path,
    infer_probe_root,
    load_experiment_config,
    load_model_for_checkpoint,
    mode_to_inputs,
    parse_csv_arg,
    resolve_checkpoint_path,
    save_json,
    tensor_to_numpy_list,
)


FEATURE_KEYS = [
    "z_shared",
    "z_img",
    "z_spec",
    "z_nuis",
    "latent_shared_proj",
    "latent_image_proj",
    "latent_spectrum_proj",
    "pred_nuisance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen AstroMapAnything probe features.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint-final.pth")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--modes", type=str, default="image-only,spectrum-only,both")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-metadata", dest="disable_metadata", action="store_true")
    parser.add_argument("--allow-metadata", dest="disable_metadata", action="store_false")
    parser.set_defaults(disable_metadata=True)
    parser.add_argument("--dino-local-repo", type=str, default=None)
    parser.add_argument("--dino-local-ckpt", type=str, default=None)
    return parser.parse_args()


def _init_store() -> Dict[str, List[np.ndarray]]:
    store: Dict[str, List[np.ndarray]] = {
        "sample_id": [],
        "z": [],
        "ra": [],
        "dec": [],
        "sn_median_r": [],
        "target_log1p_z": [],
    }
    for key in FEATURE_KEYS:
        store[key] = []
    return store


def _append_batch(
    store: Dict[str, List[np.ndarray]],
    sample_ids: np.ndarray,
    row_lookup,
    features: Dict[str, torch.Tensor],
) -> None:
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    rows = [row_lookup[str(int(sample_id))] for sample_id in sample_ids]
    store["sample_id"].append(sample_ids)
    z = np.asarray([float(row.get("z", 0.0)) for row in rows], dtype=np.float32)
    store["z"].append(z)
    store["target_log1p_z"].append(np.log1p(np.clip(z, a_min=0.0, a_max=None)).astype(np.float32))
    store["ra"].append(np.asarray([float(row.get("ra", 0.0)) for row in rows], dtype=np.float32))
    store["dec"].append(np.asarray([float(row.get("dec", 0.0)) for row in rows], dtype=np.float32))
    store["sn_median_r"].append(
        np.asarray([float(row.get("sn_median_r", 0.0)) for row in rows], dtype=np.float32)
    )
    for key in FEATURE_KEYS:
        if key not in features:
            continue
        store[key].append(tensor_to_numpy_list(features[key]).astype(np.float32))


def _finalize_store(store: Dict[str, List[np.ndarray]], split: str, mode: str) -> Dict[str, np.ndarray]:
    finalized: Dict[str, np.ndarray] = {}
    n_rows = 0
    for key, parts in store.items():
        if not parts:
            continue
        finalized[key] = np.concatenate(parts, axis=0)
        n_rows = finalized[key].shape[0]
    finalized["split"] = np.asarray([split] * n_rows)
    finalized["feature_mode"] = np.asarray([mode] * n_rows)
    return finalized


def export_split_features(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    modes: List[str],
    batch_size: int,
    num_workers: int,
    max_samples_per_split: int | None,
    disable_metadata: bool,
):
    dataset_for_loader = dataset
    if max_samples_per_split is not None:
        limit = min(int(max_samples_per_split), len(dataset))
        dataset_for_loader = Subset(dataset, list(range(limit)))
    loader = DataLoader(
        dataset_for_loader,
        sampler=SequentialSampler(dataset_for_loader),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    row_lookup = build_row_lookup(dataset)
    stores = {mode: _init_store() for mode in modes}

    for batch in loader:
        view_image, view_spectrum = batch
        image = view_image["img_astro"].to(device=device, dtype=torch.float32, non_blocking=True)
        spectrum = view_spectrum["spec_astro"].to(device=device, dtype=torch.float32, non_blocking=True)
        sample_ids = view_image["sample_id"].detach().cpu().numpy()

        metadata = None
        if not disable_metadata:
            metadata = view_image["meta_cond"].to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.inference_mode():
            for mode in modes:
                image_input, spectrum_input = mode_to_inputs(mode, image=image, spectrum=spectrum)
                features = extract_backbone_features(
                    model,
                    image=image_input,
                    spectrum=spectrum_input,
                    metadata=metadata,
                    return_tokens=False,
                )
                _append_batch(stores[mode], sample_ids=sample_ids, row_lookup=row_lookup, features=features)

    return {mode: _finalize_store(stores[mode], split=split, mode=mode) for mode in modes}


def main() -> None:
    args = parse_args()
    ensure_dino_environment(args.dino_local_repo, args.dino_local_ckpt)

    experiment_dir = args.experiment_dir.resolve()
    cfg = load_experiment_config(experiment_dir)
    checkpoint_path = resolve_checkpoint_path(experiment_dir, args.checkpoint)

    output_dir = args.output_dir.resolve() if args.output_dir else (infer_probe_root(experiment_dir) / "features")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, load_message = load_model_for_checkpoint(cfg, checkpoint_path, device)

    splits = parse_csv_arg(args.splits)
    modes = parse_csv_arg(args.modes)
    for split in splits:
        dataset = build_dataset_from_cfg(cfg, split=split)
        exported = export_split_features(
            dataset=dataset,
            model=model,
            device=device,
            split=split,
            modes=modes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples_per_split=args.max_samples_per_split,
            disable_metadata=args.disable_metadata,
        )
        for mode, arrays in exported.items():
            archive_path = feature_archive_path(output_dir, split=split, mode=mode)
            np.savez_compressed(archive_path, **arrays)
            print(f"[export] wrote {archive_path} ({arrays['sample_id'].shape[0]} samples)")

    metadata = {
        "experiment_dir": str(experiment_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": checkpoint_label(checkpoint_path),
        "device": str(device),
        "disable_metadata": bool(args.disable_metadata),
        "splits": splits,
        "modes": modes,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "max_samples_per_split": args.max_samples_per_split,
        "feature_keys": FEATURE_KEYS,
        "load_report": {
            "missing_keys": list(load_message.missing_keys),
            "unexpected_keys": list(load_message.unexpected_keys),
        },
    }
    save_json(metadata, output_dir / "export_metadata.json")
    print(f"[export] metadata saved to {output_dir / 'export_metadata.json'}")


if __name__ == "__main__":
    main()
