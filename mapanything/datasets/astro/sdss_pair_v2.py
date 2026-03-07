"""
Astro SDSS paired dataset v2 with astronomy-aware image masking and raw-spectrum auxiliaries.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mapanything.datasets.astro.sdss_pair import (
    AstroSDSSPairDataset,
    _hash_to_seed,
    _infer_train_manifest_path,
    _linear_ratio,
    _load_manifest,
    _make_image_block_mask,
    _make_spectrum_span_mask,
    _safe_std,
)


def _sample_band_mask(
    rng: np.random.Generator,
    num_bands: int,
    mask_count_range: Tuple[int, int],
) -> np.ndarray:
    min_count = int(max(0, mask_count_range[0]))
    max_count = int(min(num_bands - 1, mask_count_range[1]))
    if max_count <= 0 or max_count < min_count:
        return np.zeros((num_bands,), dtype=bool)
    num_mask = int(rng.integers(min_count, max_count + 1))
    if num_mask <= 0:
        return np.zeros((num_bands,), dtype=bool)
    chosen = rng.choice(num_bands, size=num_mask, replace=False)
    out = np.zeros((num_bands,), dtype=bool)
    out[chosen] = True
    return out


class AstroSDSSPairDatasetV2(AstroSDSSPairDataset):
    """
    V2 dataset adds image-band masking, sparse patch dropout, true spectrum span masking,
    and raw flux auxiliary targets derived from local DR17 products.
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
        enable_band_masking: bool = True,
        band_mask_count_range: Tuple[int, int] = (1, 2),
        enable_sparse_patch_dropout: bool = True,
        sparse_patch_dropout_ratio: Tuple[float, float] = (0.05, 0.15),
        enable_spectrum_input_masking: bool = True,
        use_raw_spectrum_aux_targets: bool = True,
        raw_spectrum_aux_keys: Tuple[str, str] = ("flux_obs", "flux_rest"),
    ):
        self.enable_band_masking = enable_band_masking
        self.band_mask_count_range = tuple(band_mask_count_range)
        self.enable_sparse_patch_dropout = enable_sparse_patch_dropout
        self.sparse_patch_dropout_ratio = tuple(sparse_patch_dropout_ratio)
        self.enable_spectrum_input_masking = enable_spectrum_input_masking
        self.use_raw_spectrum_aux_targets = use_raw_spectrum_aux_targets
        self.raw_spectrum_aux_keys = tuple(raw_spectrum_aux_keys)

        super().__init__(
            manifest_path=manifest_path,
            split=split,
            use_metadata=use_metadata,
            mask_ratio_image=mask_ratio_image,
            mask_ratio_spec=mask_ratio_spec,
            seed=seed,
            eval_fixed_mask=eval_fixed_mask,
            image_size=image_size,
            image_patch_size=image_patch_size,
            spec_length=spec_length,
            spec_span_min=spec_span_min,
            spec_span_max=spec_span_max,
            modality_probs_stage_b=modality_probs_stage_b,
            modality_probs_stage_c=modality_probs_stage_c,
            eval_modality_probs=eval_modality_probs,
            stage_a_epochs=stage_a_epochs,
            stage_b_epochs=stage_b_epochs,
            stage_c_epochs=stage_c_epochs,
            mask_schedule_epochs=mask_schedule_epochs,
            stats_manifest_path=stats_manifest_path,
            stats_cache_path=stats_cache_path,
            num_views=num_views,
            variable_num_views=variable_num_views,
            resolution=resolution,
            data_norm_type=data_norm_type,
            max_num_retries=max_num_retries,
        )

        flux_stats = getattr(self, "flux_aux_stats", None)
        if flux_stats is None:
            self.flux_aux_mean = np.zeros((len(self.raw_spectrum_aux_keys),), dtype=np.float32)
            self.flux_aux_std = np.ones((len(self.raw_spectrum_aux_keys),), dtype=np.float32)
        else:
            self.flux_aux_mean = np.asarray(flux_stats["mean"], dtype=np.float32)
            self.flux_aux_std = _safe_std(np.asarray(flux_stats["std"], dtype=np.float32))

        nuisance_stats = getattr(self, "image_nuisance_stats", None)
        if nuisance_stats is None:
            self.image_nuisance_mean = np.zeros((11,), dtype=np.float32)
            self.image_nuisance_std = np.ones((11,), dtype=np.float32)
        else:
            self.image_nuisance_mean = np.asarray(nuisance_stats["mean"], dtype=np.float32)
            self.image_nuisance_std = _safe_std(
                np.asarray(nuisance_stats["std"], dtype=np.float32)
            )

    def _load_or_compute_stats(self) -> Dict[str, List[float]]:
        required_keys = {
            "image_mean",
            "image_std",
            "image_nuisance_mean",
            "image_nuisance_std",
            "spec_mean",
            "spec_std",
            "meta_mean",
            "meta_std",
        }
        if self.use_raw_spectrum_aux_targets:
            required_keys.update({"flux_aux_mean", "flux_aux_std", "raw_spectrum_aux_keys"})

        if os.path.isfile(self.stats_cache_path):
            with open(self.stats_cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if required_keys.issubset(cached.keys()):
                if tuple(cached.get("raw_spectrum_aux_keys", ())) == tuple(self.raw_spectrum_aux_keys):
                    self.image_nuisance_stats = {
                        "mean": cached["image_nuisance_mean"],
                        "std": cached["image_nuisance_std"],
                    }
                    if self.use_raw_spectrum_aux_targets:
                        self.flux_aux_stats = {
                            "mean": cached["flux_aux_mean"],
                            "std": cached["flux_aux_std"],
                        }
                    return cached

        rows = _load_manifest(self.stats_manifest_path)
        img_sum = np.zeros((5,), dtype=np.float64)
        img_sq_sum = np.zeros((5,), dtype=np.float64)
        img_count = 0

        spec_sum = np.zeros((4,), dtype=np.float64)
        spec_sq_sum = np.zeros((4,), dtype=np.float64)
        spec_count = 0

        flux_aux_sum = np.zeros((len(self.raw_spectrum_aux_keys),), dtype=np.float64)
        flux_aux_sq_sum = np.zeros((len(self.raw_spectrum_aux_keys),), dtype=np.float64)
        flux_aux_count = 0

        nuisance_sum = np.zeros((11,), dtype=np.float64)
        nuisance_sq_sum = np.zeros((11,), dtype=np.float64)
        nuisance_count = 0

        meta_sum = np.zeros((4,), dtype=np.float64)
        meta_sq_sum = np.zeros((4,), dtype=np.float64)
        meta_count = 0

        for row in rows:
            with np.load(row["image_npz"]) as img_npz:
                img = img_npz["image"].astype(np.float64)
                bg = img_npz.get("bg", np.zeros((5,), dtype=np.float64)).astype(np.float64)
                noise = img_npz.get("noise", np.zeros((5,), dtype=np.float64)).astype(np.float64)
            with np.load(row["spectrum_npz"]) as spec_npz:
                spec = spec_npz["spec_features"].astype(np.float64)
                mask_obs = spec_npz.get("mask_obs", np.zeros((self.spec_length,), dtype=np.uint8))
                flux_aux = []
                for key in self.raw_spectrum_aux_keys:
                    flux_aux.append(spec_npz[key].astype(np.float64))
                flux_aux = np.stack(flux_aux, axis=0)

            valid = (mask_obs.astype(np.uint8) == 0)
            valid &= np.isfinite(flux_aux).all(axis=0)

            spec[2] = np.clip(spec[2], -5.0, 5.0)
            img_flat = img.reshape(5, -1)
            spec_flat = spec.reshape(4, -1)

            img_sum += img_flat.sum(axis=1)
            img_sq_sum += (img_flat**2).sum(axis=1)
            img_count += img_flat.shape[1]

            quality_ratio = float(row.get("quality_mask_ratio", 0.0))
            nuisance = np.concatenate(
                [
                    bg.reshape(-1),
                    noise.reshape(-1),
                    np.asarray([quality_ratio], dtype=np.float64),
                ],
                axis=0,
            )
            nuisance_sum += nuisance
            nuisance_sq_sum += nuisance**2
            nuisance_count += 1

            spec_sum += spec_flat.sum(axis=1)
            spec_sq_sum += (spec_flat**2).sum(axis=1)
            spec_count += spec_flat.shape[1]

            if self.use_raw_spectrum_aux_targets and np.any(valid):
                valid_flux = flux_aux[:, valid]
                flux_aux_sum += valid_flux.sum(axis=1)
                flux_aux_sq_sum += (valid_flux**2).sum(axis=1)
                flux_aux_count += valid_flux.shape[1]

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
            "image_nuisance_mean": (nuisance_sum / max(1, nuisance_count)).tolist(),
            "image_nuisance_std": np.sqrt(
                np.maximum(nuisance_sq_sum / max(1, nuisance_count) - (nuisance_sum / max(1, nuisance_count)) ** 2, 1e-8)
            ).tolist(),
            "spec_mean": spec_mean.tolist(),
            "spec_std": np.sqrt(np.maximum(spec_var, 1e-8)).tolist(),
            "meta_mean": meta_mean.tolist(),
            "meta_std": np.sqrt(np.maximum(meta_var, 1e-8)).tolist(),
            "stats_manifest_path": self.stats_manifest_path,
        }

        self.image_nuisance_stats = {
            "mean": stats["image_nuisance_mean"],
            "std": stats["image_nuisance_std"],
        }

        if self.use_raw_spectrum_aux_targets:
            flux_aux_mean = flux_aux_sum / max(1, flux_aux_count)
            flux_aux_var = flux_aux_sq_sum / max(1, flux_aux_count) - flux_aux_mean**2
            stats["flux_aux_mean"] = flux_aux_mean.tolist()
            stats["flux_aux_std"] = np.sqrt(np.maximum(flux_aux_var, 1e-8)).tolist()
            stats["raw_spectrum_aux_keys"] = list(self.raw_spectrum_aux_keys)
            self.flux_aux_stats = {
                "mean": stats["flux_aux_mean"],
                "std": stats["flux_aux_std"],
            }

        cache_dir = os.path.dirname(self.stats_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(self.stats_cache_path, "w", encoding="utf-8") as f:
                json.dump(stats, f)
        except OSError:
            pass
        return stats

    def _get_sparse_dropout_ratio(self) -> float:
        if not self.enable_sparse_patch_dropout:
            return 0.0
        if self.split != "train":
            return float(self.sparse_patch_dropout_ratio[-1])
        progress = self._epoch / float(self.mask_schedule_epochs)
        return float(
            _linear_ratio(
                self.sparse_patch_dropout_ratio[0],
                self.sparse_patch_dropout_ratio[-1],
                progress,
            )
        )

    def _get_rng(self, sample_id: str, sample_idx: int) -> np.random.Generator:
        if self.eval_fixed_mask and self.split != "train":
            local_seed = _hash_to_seed(sample_id, offset=self.seed or 0)
            return np.random.default_rng(seed=local_seed)
        return super()._get_rng(sample_id=sample_id, sample_idx=sample_idx)

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
            bg = image_npz.get("bg", np.zeros((5,), dtype=np.float32)).astype(np.float32)
            noise = image_npz.get("noise", np.zeros((5,), dtype=np.float32)).astype(np.float32)
        with np.load(row["spectrum_npz"]) as spec_npz:
            spec = spec_npz["spec_features"].astype(np.float32)
            mask_obs = spec_npz.get("mask_obs", np.zeros((self.spec_length,), dtype=np.uint8))
            ivar_obs = spec_npz.get("ivar_obs", np.ones((self.spec_length,), dtype=np.float32))
            flux_aux = []
            for key in self.raw_spectrum_aux_keys:
                flux_aux.append(spec_npz[key].astype(np.float32))
            flux_aux = np.stack(flux_aux, axis=0)

        if image.shape != (5, self.image_size, self.image_size):
            raise ValueError(f"Unexpected image shape {image.shape} for sample {sample_id}")
        if spec.shape != (4, self.spec_length):
            raise ValueError(f"Unexpected spectrum shape {spec.shape} for sample {sample_id}")

        spec[2] = np.clip(spec[2], -5.0, 5.0)

        img_valid_mask = (1 - quality_mask.astype(np.uint8)).astype(bool)
        img_valid_mask &= np.isfinite(image).all(axis=0)
        spec_valid_mask = (mask_obs.astype(np.uint8) == 0)
        spec_valid_mask &= np.isfinite(spec).all(axis=0)

        quality_ratio = float(row.get("quality_mask_ratio", quality_mask.mean()))
        nuisance_target = np.concatenate(
            [
                bg.reshape(-1),
                noise.reshape(-1),
                np.asarray([quality_ratio], dtype=np.float32),
            ],
            axis=0,
        )

        image = (image - self.image_mean[:, None, None]) / (self.image_std[:, None, None] + 1e-6)
        spec = (spec - self.spec_mean[:, None]) / (self.spec_std[:, None] + 1e-6)

        nuisance_target = (nuisance_target - self.image_nuisance_mean) / (
            self.image_nuisance_std + 1e-6
        )

        flux_aux = (flux_aux - self.flux_aux_mean[:, None]) / (self.flux_aux_std[:, None] + 1e-6)

        ivar = ivar_obs.astype(np.float32)
        ivar = np.clip(ivar, 0.0, np.percentile(ivar, 99.0) + 1e-6)
        ivar_med = np.median(ivar[ivar > 0]) if np.any(ivar > 0) else 1.0
        ivar = ivar / max(ivar_med, 1e-6)
        ivar = np.clip(ivar, 0.0, 5.0)

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
        sparse_ratio = self._get_sparse_dropout_ratio()

        sparse_patch_mask = _make_image_block_mask(
            rng=rng,
            image_size=self.image_size,
            patch_size=self.image_patch_size,
            mask_ratio=sparse_ratio,
        )
        if self.enable_band_masking and (self.split == "train" or self.eval_fixed_mask):
            band_mask = _sample_band_mask(rng, num_bands=5, mask_count_range=self.band_mask_count_range)
        else:
            band_mask = np.zeros((5,), dtype=bool)

        img_mask_tokens = np.broadcast_to(sparse_patch_mask[None], (5, self.image_size, self.image_size)).copy()
        if np.any(band_mask):
            img_mask_tokens |= np.broadcast_to(band_mask[:, None, None], img_mask_tokens.shape)

        spec_mask_tokens = _make_spectrum_span_mask(
            rng=rng,
            spec_len=self.spec_length,
            mask_ratio=ratio_spec,
            span_min=self.spec_span_min,
            span_max=self.spec_span_max,
        )

        img_masked = image.copy()
        if np.any(band_mask):
            img_masked[band_mask] = 0.0
        if np.any(sparse_patch_mask):
            img_masked = np.where(np.broadcast_to(sparse_patch_mask[None], img_masked.shape), 0.0, img_masked)

        spec_masked = spec.copy()
        if self.enable_spectrum_input_masking and np.any(spec_mask_tokens):
            spec_masked[:, spec_mask_tokens] = 0.0

        input_state = self._sample_input_state(rng=rng)
        image_input_mask = input_state in (0, 2)
        spectrum_input_mask = input_state in (1, 2)

        sample_id_int = int(sample_id)
        epoch_idx = int(self._epoch)

        view_image = {
            "img_astro": torch.from_numpy(image).float(),
            "img_masked_astro": torch.from_numpy(img_masked).float(),
            "img_valid_mask": torch.from_numpy(img_valid_mask.astype(np.bool_)),
            "img_mask_tokens": torch.from_numpy(img_mask_tokens.astype(np.bool_)),
            "img_band_mask": torch.from_numpy(band_mask.astype(np.bool_)),
            "img_nuisance_targets": torch.from_numpy(nuisance_target).float(),
            "image_input_mask": torch.tensor(image_input_mask, dtype=torch.bool),
            "input_state": torch.tensor(input_state, dtype=torch.long),
            "sample_id": torch.tensor(sample_id_int, dtype=torch.long),
            "meta_cond": torch.from_numpy(meta).float(),
            "meta_valid": torch.tensor(meta_valid, dtype=torch.bool),
            "epoch_idx": torch.tensor(epoch_idx, dtype=torch.long),
            "dataset": "AstroSDSSV2",
            "label": self.split,
            "instance": sample_id,
            "data_norm_type": self.data_norm_type,
        }
        view_spectrum = {
            "spec_astro": torch.from_numpy(spec).float(),
            "spec_masked_astro": torch.from_numpy(spec_masked).float(),
            "spec_aux_targets": torch.from_numpy(flux_aux).float(),
            "spec_valid_mask": torch.from_numpy(spec_valid_mask.astype(np.bool_)),
            "spec_mask_tokens": torch.from_numpy(spec_mask_tokens.astype(np.bool_)),
            "spec_ivar_weight": torch.from_numpy(ivar).float(),
            "spectrum_input_mask": torch.tensor(spectrum_input_mask, dtype=torch.bool),
            "input_state": torch.tensor(input_state, dtype=torch.long),
            "sample_id": torch.tensor(sample_id_int, dtype=torch.long),
            "meta_cond": torch.from_numpy(meta).float(),
            "meta_valid": torch.tensor(meta_valid, dtype=torch.bool),
            "epoch_idx": torch.tensor(epoch_idx, dtype=torch.long),
            "dataset": "AstroSDSSV2",
            "label": self.split,
            "instance": sample_id,
            "data_norm_type": self.data_norm_type,
        }
        return [view_image, view_spectrum]
