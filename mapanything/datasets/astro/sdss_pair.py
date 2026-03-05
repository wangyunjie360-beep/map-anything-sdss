"""
Paired SDSS astronomy dataset for cross-modal self-supervision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mapanything.datasets.base.easy_dataset import EasyDataset


def _safe_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(x, eps)


def _hash_to_seed(value: str, offset: int = 0) -> int:
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return (int(h, 16) + offset) % (2**31 - 1)


def _load_manifest(manifest_path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if len(rows) == 0:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def _linear_ratio(start: float, end: float, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return start + (end - start) * progress


def _infer_train_manifest_path(manifest_path: str) -> str:
    # Prefer explicit train file if current manifest is val/test.
    candidates = []
    if "train_ready_val" in manifest_path:
        candidates.append(manifest_path.replace("train_ready_val", "train_ready_train"))
    if "train_ready_test" in manifest_path:
        candidates.append(manifest_path.replace("train_ready_test", "train_ready_train"))
    candidates.append(os.path.join(os.path.dirname(manifest_path), "train_ready_train.jsonl"))
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return manifest_path


def _make_image_block_mask(
    rng: np.random.Generator,
    image_size: int,
    patch_size: int,
    mask_ratio: float,
) -> np.ndarray:
    gh = image_size // patch_size
    gw = image_size // patch_size
    num_patches = gh * gw
    num_mask = int(round(mask_ratio * num_patches))
    num_mask = int(np.clip(num_mask, 0, num_patches))

    mask_grid = np.zeros((gh * gw,), dtype=bool)
    if num_mask > 0:
        chosen = rng.choice(num_patches, size=num_mask, replace=False)
        mask_grid[chosen] = True
    mask_grid = mask_grid.reshape(gh, gw)
    # Expand patch mask into pixel mask.
    return np.repeat(np.repeat(mask_grid, patch_size, axis=0), patch_size, axis=1)


def _make_spectrum_span_mask(
    rng: np.random.Generator,
    spec_len: int,
    mask_ratio: float,
    span_min: int,
    span_max: int,
) -> np.ndarray:
    target = int(round(mask_ratio * spec_len))
    target = int(np.clip(target, 0, spec_len))
    if target == 0:
        return np.zeros((spec_len,), dtype=bool)

    mask = np.zeros((spec_len,), dtype=bool)
    attempts = 0
    max_attempts = spec_len * 4
    while int(mask.sum()) < target and attempts < max_attempts:
        span = int(rng.integers(span_min, span_max + 1))
        span = int(np.clip(span, 1, spec_len))
        start = int(rng.integers(0, max(1, spec_len - span + 1)))
        mask[start : start + span] = True
        attempts += 1

    # Guarantee exact target by random fill/cut.
    cur = int(mask.sum())
    if cur < target:
        unmasked = np.flatnonzero(~mask)
        add = rng.choice(unmasked, size=target - cur, replace=False)
        mask[add] = True
    elif cur > target:
        masked = np.flatnonzero(mask)
        remove = rng.choice(masked, size=cur - target, replace=False)
        mask[remove] = False
    return mask


class AstroSDSSPairDataset(EasyDataset):
    """
    Paired SDSS dataset with astronomy-aware masking and normalization.

    This dataset always returns exactly two views:
    - view0: image modality
    - view1: spectrum modality
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        use_metadata: bool = True,
        mask_ratio_image: Tuple[float, float] = (0.4, 0.6),
        mask_ratio_spec: Tuple[float, float] = (0.3, 0.5),
        seed: Optional[int] = None,
        eval_fixed_mask: bool = False,
        image_size: int = 224,
        image_patch_size: int = 14,
        spec_length: int = 2048,
        spec_span_min: int = 8,
        spec_span_max: int = 64,
        modality_probs_stage_b: Tuple[float, float, float] = (0.45, 0.35, 0.20),
        modality_probs_stage_c: Tuple[float, float, float] = (0.50, 0.30, 0.20),
        eval_modality_probs: Tuple[float, float, float] = (0.45, 0.35, 0.20),
        stage_a_epochs: int = 20,
        stage_b_epochs: int = 80,
        stage_c_epochs: int = 20,
        mask_schedule_epochs: int = 120,
        stats_manifest_path: Optional[str] = None,
        stats_cache_path: Optional[str] = None,
        num_views: int = 2,
        variable_num_views: bool = False,
        resolution: Tuple[int, int] = (224, 224),
        data_norm_type: str = "identity",
        max_num_retries: int = 5,
    ):
        if variable_num_views:
            raise ValueError("AstroSDSSPairDataset only supports fixed 2-view setup.")
        if num_views != 2:
            raise ValueError(f"AstroSDSSPairDataset expects num_views=2, got {num_views}.")
        if resolution != (image_size, image_size):
            raise ValueError(
                f"Resolution must match image_size ({image_size}), got {resolution}."
            )

        self.manifest_path = manifest_path
        self.split = split
        self.use_metadata = use_metadata
        self.mask_ratio_image = tuple(mask_ratio_image)
        self.mask_ratio_spec = tuple(mask_ratio_spec)
        self.seed = seed
        self.eval_fixed_mask = eval_fixed_mask
        self.image_size = image_size
        self.image_patch_size = image_patch_size
        self.spec_length = spec_length
        self.spec_span_min = spec_span_min
        self.spec_span_max = spec_span_max
        self.modality_probs_stage_b = np.asarray(modality_probs_stage_b, dtype=np.float64)
        self.modality_probs_stage_c = np.asarray(modality_probs_stage_c, dtype=np.float64)
        self.eval_modality_probs = np.asarray(eval_modality_probs, dtype=np.float64)
        self.stage_a_epochs = stage_a_epochs
        self.stage_b_epochs = stage_b_epochs
        self.stage_c_epochs = stage_c_epochs
        self.mask_schedule_epochs = max(1, mask_schedule_epochs)
        self.num_views = num_views
        self.variable_num_views = variable_num_views
        self.data_norm_type = data_norm_type
        self.max_num_retries = max_num_retries

        self._resolutions = [resolution]
        self._seed_offset = 0
        self._epoch = 0

        self.entries = _load_manifest(self.manifest_path)
        self.num_of_scenes = len(self.entries)
        self.is_metric_scale = False
        self.is_synthetic = False

        # Paired modality integrity check.
        for entry in self.entries:
            sid = str(entry["sample_id"])
            sid_img = Path(entry["image_npz"]).stem
            sid_spec = Path(entry["spectrum_npz"]).stem
            if sid != sid_img or sid != sid_spec:
                raise ValueError(
                    f"Unpaired sample detected: sample_id={sid}, image={sid_img}, spectrum={sid_spec}"
                )

        if stats_manifest_path is None:
            stats_manifest_path = _infer_train_manifest_path(self.manifest_path)
        self.stats_manifest_path = stats_manifest_path

        if stats_cache_path is None:
            stats_cache_path = os.path.join(
                os.path.dirname(self.stats_manifest_path), "astro_train_norm_stats.json"
            )
        self.stats_cache_path = stats_cache_path

        stats = self._load_or_compute_stats()
        self.image_mean = np.asarray(stats["image_mean"], dtype=np.float32)
        self.image_std = _safe_std(np.asarray(stats["image_std"], dtype=np.float32))
        self.spec_mean = np.asarray(stats["spec_mean"], dtype=np.float32)
        self.spec_std = _safe_std(np.asarray(stats["spec_std"], dtype=np.float32))
        self.meta_mean = np.asarray(stats["meta_mean"], dtype=np.float32)
        self.meta_std = _safe_std(np.asarray(stats["meta_std"], dtype=np.float32))

    def __len__(self) -> int:
        return self.num_of_scenes

    def __repr__(self) -> str:
        return (
            f"AstroSDSSPairDataset(split={self.split}, samples={self.num_of_scenes}, "
            f"manifest='{self.manifest_path}', use_metadata={self.use_metadata})"
        )

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _set_seed_offset(self, idx: int):
        self._seed_offset = int(idx)

    def _load_or_compute_stats(self) -> Dict[str, List[float]]:
        if os.path.isfile(self.stats_cache_path):
            with open(self.stats_cache_path, encoding="utf-8") as f:
                return json.load(f)

        rows = _load_manifest(self.stats_manifest_path)
        img_sum = np.zeros((5,), dtype=np.float64)
        img_sq_sum = np.zeros((5,), dtype=np.float64)
        img_count = 0

        spec_sum = np.zeros((4,), dtype=np.float64)
        spec_sq_sum = np.zeros((4,), dtype=np.float64)
        spec_count = 0

        meta_sum = np.zeros((4,), dtype=np.float64)
        meta_sq_sum = np.zeros((4,), dtype=np.float64)
        meta_count = 0

        for row in rows:
            with np.load(row["image_npz"]) as img_npz:
                img = img_npz["image"].astype(np.float64)
            with np.load(row["spectrum_npz"]) as spec_npz:
                spec = spec_npz["spec_features"].astype(np.float64)

            spec[2] = np.clip(spec[2], -5.0, 5.0)

            img_flat = img.reshape(5, -1)
            spec_flat = spec.reshape(4, -1)

            img_sum += img_flat.sum(axis=1)
            img_sq_sum += (img_flat**2).sum(axis=1)
            img_count += img_flat.shape[1]

            spec_sum += spec_flat.sum(axis=1)
            spec_sq_sum += (spec_flat**2).sum(axis=1)
            spec_count += spec_flat.shape[1]

            z = max(float(row.get("z", 0.0)), 0.0)
            meta = np.asarray(
                [
                    float(row.get("ra", 0.0)),
                    float(row.get("dec", 0.0)),
                    math.log1p(z),
                    float(row.get("sn_median_r", 0.0)),
                ],
                dtype=np.float64,
            )
            meta_sum += meta
            meta_sq_sum += meta**2
            meta_count += 1

        img_mean = img_sum / max(1, img_count)
        img_var = img_sq_sum / max(1, img_count) - img_mean**2
        spec_mean = spec_sum / max(1, spec_count)
        spec_var = spec_sq_sum / max(1, spec_count) - spec_mean**2
        meta_mean = meta_sum / max(1, meta_count)
        meta_var = meta_sq_sum / max(1, meta_count) - meta_mean**2

        stats = {
            "image_mean": img_mean.tolist(),
            "image_std": np.sqrt(np.maximum(img_var, 1e-8)).tolist(),
            "spec_mean": spec_mean.tolist(),
            "spec_std": np.sqrt(np.maximum(spec_var, 1e-8)).tolist(),
            "meta_mean": meta_mean.tolist(),
            "meta_std": np.sqrt(np.maximum(meta_var, 1e-8)).tolist(),
            "stats_manifest_path": self.stats_manifest_path,
        }
        cache_dir = os.path.dirname(self.stats_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(self.stats_cache_path, "w", encoding="utf-8") as f:
                json.dump(stats, f)
        except OSError:
            # DDP workers may race on first-time cache write; training can proceed without caching.
            pass
        return stats

    def _get_rng(self, sample_id: str, sample_idx: int) -> np.random.Generator:
        if self.eval_fixed_mask and self.split != "train":
            local_seed = _hash_to_seed(sample_id, offset=self.seed or 0)
            return np.random.default_rng(seed=local_seed)
        if self.seed is None:
            seed = int(torch.initial_seed()) + self._seed_offset + sample_idx
        else:
            seed = self.seed + self._seed_offset + sample_idx + (self._epoch * 1_000_003)
        return np.random.default_rng(seed=seed)

    def _get_modality_probs(self) -> np.ndarray:
        if self.split != "train":
            probs = self.eval_modality_probs
        elif self._epoch < self.stage_a_epochs:
            probs = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        elif self._epoch < self.stage_a_epochs + self.stage_b_epochs:
            probs = self.modality_probs_stage_b
        else:
            probs = self.modality_probs_stage_c
        probs = probs / np.clip(probs.sum(), 1e-8, None)
        return probs

    def _sample_input_state(self, rng: np.random.Generator) -> int:
        # 0=image-only, 1=spectrum-only, 2=both
        probs = self._get_modality_probs()
        return int(rng.choice(3, p=probs))

    def _get_current_mask_ratios(self) -> Tuple[float, float]:
        if self.split != "train":
            return self.mask_ratio_image[1], self.mask_ratio_spec[1]
        progress = self._epoch / float(self.mask_schedule_epochs)
        ratio_img = _linear_ratio(self.mask_ratio_image[0], self.mask_ratio_image[1], progress)
        ratio_spec = _linear_ratio(self.mask_ratio_spec[0], self.mask_ratio_spec[1], progress)
        return ratio_img, ratio_spec

    def _getitem_impl(self, idx):
        if isinstance(idx, tuple):
            idx = int(idx[0])
        else:
            idx = int(idx)
        row = self.entries[idx]
        sample_id = str(row["sample_id"])
        rng = self._get_rng(sample_id=sample_id, sample_idx=idx)

        with np.load(row["image_npz"]) as image_npz:
            image = image_npz["image"].astype(np.float32)
            quality_mask = image_npz.get(
                "quality_mask", np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            )
        with np.load(row["spectrum_npz"]) as spec_npz:
            spec = spec_npz["spec_features"].astype(np.float32)
            # `mask_obs == 1` means masked / invalid bins in this dataset.
            # Fallback to all-valid if the key is missing.
            mask_obs = spec_npz.get("mask_obs", np.zeros((self.spec_length,), dtype=np.uint8))
            ivar_obs = spec_npz.get("ivar_obs", np.ones((self.spec_length,), dtype=np.float32))

        if image.shape != (5, self.image_size, self.image_size):
            raise ValueError(f"Unexpected image shape {image.shape} for sample {sample_id}")
        if spec.shape != (4, self.spec_length):
            raise ValueError(f"Unexpected spectrum shape {spec.shape} for sample {sample_id}")

        # Domain prior: keep residual channel in bounded range.
        spec[2] = np.clip(spec[2], -5.0, 5.0)

        # Valid masks.
        img_valid_mask = (1 - quality_mask.astype(np.uint8)).astype(bool)
        img_valid_mask &= np.isfinite(image).all(axis=0)
        # Keep bins that are explicitly marked as observed and finite.
        spec_valid_mask = (mask_obs.astype(np.uint8) == 0)
        spec_valid_mask &= np.isfinite(spec).all(axis=0)

        # Train-split normalization.
        image = (image - self.image_mean[:, None, None]) / (self.image_std[:, None, None] + 1e-6)
        spec = (spec - self.spec_mean[:, None]) / (self.spec_std[:, None] + 1e-6)

        # Spectral weighting from ivar.
        ivar = ivar_obs.astype(np.float32)
        ivar = np.clip(ivar, 0.0, np.percentile(ivar, 99.0) + 1e-6)
        ivar_med = np.median(ivar[ivar > 0]) if np.any(ivar > 0) else 1.0
        ivar = ivar / max(ivar_med, 1e-6)
        ivar = np.clip(ivar, 0.0, 5.0)

        # Metadata conditional token.
        if self.use_metadata:
            z = max(float(row.get("z", 0.0)), 0.0)
            meta = np.asarray(
                [
                    float(row.get("ra", 0.0)),
                    float(row.get("dec", 0.0)),
                    math.log1p(z),
                    float(row.get("sn_median_r", 0.0)),
                ],
                dtype=np.float32,
            )
            meta = (meta - self.meta_mean) / (self.meta_std + 1e-6)
            meta_valid = True
        else:
            meta = np.zeros((4,), dtype=np.float32)
            meta_valid = False

        ratio_img, ratio_spec = self._get_current_mask_ratios()
        img_mask_tokens = _make_image_block_mask(
            rng=rng,
            image_size=self.image_size,
            patch_size=self.image_patch_size,
            mask_ratio=ratio_img,
        )
        spec_mask_tokens = _make_spectrum_span_mask(
            rng=rng,
            spec_len=self.spec_length,
            mask_ratio=ratio_spec,
            span_min=self.spec_span_min,
            span_max=self.spec_span_max,
        )

        input_state = self._sample_input_state(rng=rng)
        image_input_mask = input_state in (0, 2)
        spectrum_input_mask = input_state in (1, 2)

        sample_id_int = int(sample_id)
        epoch_idx = int(self._epoch)

        view_image = {
            "img_astro": torch.from_numpy(image).float(),
            "img_valid_mask": torch.from_numpy(img_valid_mask.astype(np.bool_)),
            "img_mask_tokens": torch.from_numpy(img_mask_tokens.astype(np.bool_)),
            "image_input_mask": torch.tensor(image_input_mask, dtype=torch.bool),
            "input_state": torch.tensor(input_state, dtype=torch.long),
            "sample_id": torch.tensor(sample_id_int, dtype=torch.long),
            "meta_cond": torch.from_numpy(meta).float(),
            "meta_valid": torch.tensor(meta_valid, dtype=torch.bool),
            "epoch_idx": torch.tensor(epoch_idx, dtype=torch.long),
            "dataset": "AstroSDSS",
            "label": self.split,
            "instance": sample_id,
            "data_norm_type": self.data_norm_type,
        }
        view_spectrum = {
            "spec_astro": torch.from_numpy(spec).float(),
            "spec_valid_mask": torch.from_numpy(spec_valid_mask.astype(np.bool_)),
            "spec_mask_tokens": torch.from_numpy(spec_mask_tokens.astype(np.bool_)),
            "spec_ivar_weight": torch.from_numpy(ivar).float(),
            "spectrum_input_mask": torch.tensor(spectrum_input_mask, dtype=torch.bool),
            "input_state": torch.tensor(input_state, dtype=torch.long),
            "sample_id": torch.tensor(sample_id_int, dtype=torch.long),
            "meta_cond": torch.from_numpy(meta).float(),
            "meta_valid": torch.tensor(meta_valid, dtype=torch.bool),
            "epoch_idx": torch.tensor(epoch_idx, dtype=torch.long),
            "dataset": "AstroSDSS",
            "label": self.split,
            "instance": sample_id,
            "data_norm_type": self.data_norm_type,
        }
        return [view_image, view_spectrum]

    def __getitem__(self, idx):
        if self.max_num_retries <= 0:
            return self._getitem_impl(idx)

        retries = 0
        current_idx = idx
        while retries <= self.max_num_retries:
            try:
                return self._getitem_impl(current_idx)
            except Exception:
                retries += 1
                if retries > self.max_num_retries:
                    raise
                if isinstance(current_idx, tuple):
                    repl = int(np.random.randint(0, len(self)))
                    current_idx = (repl, *current_idx[1:])
                else:
                    current_idx = int(np.random.randint(0, len(self)))
