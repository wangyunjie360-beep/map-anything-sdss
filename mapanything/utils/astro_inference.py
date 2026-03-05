"""
Inference utilities for AstroMapAnything.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_batched_tensor(
    x: Optional[Any],
    expected_dims: int,
    name: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor or np.ndarray.")
    if x.ndim == expected_dims - 1:
        x = x.unsqueeze(0)
    if x.ndim != expected_dims:
        raise ValueError(f"{name} must have {expected_dims} dims, got shape {tuple(x.shape)}.")
    return x.to(device=device, dtype=torch.float32)


@torch.no_grad()
def predict_missing_modality(
    model: torch.nn.Module,
    image: Optional[Any] = None,
    spectrum: Optional[Any] = None,
    metadata: Optional[Any] = None,
    return_self_recon: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Predict missing astronomy modality using AstroMapAnything.

    Args:
        model: AstroMapAnything model.
        image: Optional image tensor of shape (5, 224, 224) or (B, 5, 224, 224).
        spectrum: Optional spectrum tensor of shape (4, 2048) or (B, 4, 2048).
        metadata: Optional metadata tensor of shape (4,) or (B, 4).
        return_self_recon: If True, also return reconstructions for available modalities.
    """
    if image is None and spectrum is None:
        raise ValueError("At least one modality must be provided.")

    model_device = next(model.parameters()).device
    image_t = _to_batched_tensor(image, expected_dims=4, name="image", device=model_device)
    spectrum_t = _to_batched_tensor(spectrum, expected_dims=3, name="spectrum", device=model_device)
    metadata_t = _to_batched_tensor(metadata, expected_dims=2, name="metadata", device=model_device)

    if image_t is not None:
        batch_size = image_t.shape[0]
    else:
        batch_size = spectrum_t.shape[0]

    if image_t is None:
        image_t = torch.zeros((batch_size, 5, 224, 224), device=model_device, dtype=torch.float32)
        image_available = torch.zeros((batch_size,), dtype=torch.bool, device=model_device)
    else:
        image_available = torch.ones((batch_size,), dtype=torch.bool, device=model_device)

    if spectrum_t is None:
        spectrum_t = torch.zeros((batch_size, 4, 2048), device=model_device, dtype=torch.float32)
        spectrum_available = torch.zeros((batch_size,), dtype=torch.bool, device=model_device)
    else:
        spectrum_available = torch.ones((batch_size,), dtype=torch.bool, device=model_device)

    if metadata_t is None:
        metadata_t = torch.zeros((batch_size, 4), device=model_device, dtype=torch.float32)
        meta_valid = torch.zeros((batch_size,), dtype=torch.bool, device=model_device)
    else:
        if metadata_t.shape[0] != batch_size:
            raise ValueError(
                f"metadata batch size ({metadata_t.shape[0]}) must match modality batch size ({batch_size})."
            )
        meta_valid = torch.ones((batch_size,), dtype=torch.bool, device=model_device)

    input_state = torch.full((batch_size,), 2, dtype=torch.long, device=model_device)
    input_state = torch.where(
        image_available & (~spectrum_available),
        torch.zeros_like(input_state),
        input_state,
    )
    input_state = torch.where(
        spectrum_available & (~image_available),
        torch.ones_like(input_state),
        input_state,
    )

    views = [
        {
            "img_astro": image_t,
            "image_input_mask": image_available,
            "input_state": input_state,
            "meta_cond": metadata_t,
            "meta_valid": meta_valid,
        },
        {
            "spec_astro": spectrum_t,
            "spectrum_input_mask": spectrum_available,
            "input_state": input_state,
            "meta_cond": metadata_t,
            "meta_valid": meta_valid,
        },
    ]

    model_was_training = model.training
    model.eval()
    preds = model(views, mode="infer")
    if model_was_training:
        model.train()
    pred = preds[0]

    out: Dict[str, torch.Tensor] = {}
    if not image_available.all():
        out["pred_image"] = pred["pred_image"]
    if not spectrum_available.all():
        out["pred_spectrum"] = pred["pred_spectrum"]

    if return_self_recon:
        if image_available.any():
            out["recon_image"] = pred["pred_image"]
        if spectrum_available.any():
            out["recon_spectrum"] = pred["pred_spectrum"]
        out["pred_image_from_spectrum"] = pred["pred_image_from_spectrum"]
        out["pred_spectrum_from_image"] = pred["pred_spectrum_from_image"]

    if "pred_image" not in out:
        out["pred_image"] = pred["pred_image"]
    if "pred_spectrum" not in out:
        out["pred_spectrum"] = pred["pred_spectrum"]
    return out
