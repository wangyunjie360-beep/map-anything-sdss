"""
Losses for astronomy multi-modal self-supervised training.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand_mask_to_tensor(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = mask
    while out.ndim < target.ndim:
        out = out.unsqueeze(1)
    return out.expand_as(target)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if value.numel() == 0:
        return value.new_zeros(())
    m = _expand_mask_to_tensor(mask, value).to(dtype=value.dtype, device=value.device)
    denom = m.sum().clamp_min(eps)
    return (value * m).sum() / denom


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def _huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_x = x.abs()
    return torch.where(abs_x <= delta, 0.5 * (x**2) / delta, abs_x - 0.5 * delta)


def _masked_pearson(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    if a.numel() == 0:
        return a.new_zeros(())
    m = _expand_mask_to_tensor(mask, a).to(dtype=a.dtype, device=a.device)
    denom = m.sum().clamp_min(1.0)
    a_m = (a * m).sum() / denom
    b_m = (b * m).sum() / denom
    a_c = (a - a_m) * m
    b_c = (b - b_m) * m
    cov = (a_c * b_c).sum()
    var_a = (a_c * a_c).sum().clamp_min(eps)
    var_b = (b_c * b_c).sum().clamp_min(eps)
    return cov / torch.sqrt(var_a * var_b + eps)


class AstroBiModalSelfSupLoss(nn.Module):
    """
    Combined objective for SDSS image-spectrum self-supervision.
    """

    def __init__(
        self,
        w_i2s: float = 1.40,
        w_spec_self: float = 0.80,
        w_s2i: float = 0.35,
        w_img_self: float = 0.25,
        w_align: float = 0.15,
        w_cons: float = 0.05,
        w_unc: float = 0.02,
        align_temperature: float = 0.07,
        charbonnier_eps: float = 1e-3,
        huber_delta: float = 1.0,
        stage_a_epochs: int = 20,
    ):
        super().__init__()
        self.w_i2s = w_i2s
        self.w_spec_self = w_spec_self
        self.w_s2i = w_s2i
        self.w_img_self = w_img_self
        self.w_align = w_align
        self.w_cons = w_cons
        self.w_unc = w_unc
        self.align_temperature = align_temperature
        self.charbonnier_eps = charbonnier_eps
        self.huber_delta = huber_delta
        self.stage_a_epochs = stage_a_epochs

    def _info_nce(self, z_img: torch.Tensor, z_spec: torch.Tensor) -> torch.Tensor:
        if z_img.shape[0] < 2:
            return z_img.new_zeros(())
        z_img = F.normalize(z_img, dim=-1)
        z_spec = F.normalize(z_spec, dim=-1)
        logits = z_img @ z_spec.t()
        logits = logits / self.align_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, batch: List[Dict], preds: List[Dict]):
        # Dataset returns 2 views: image view, spectrum view.
        view_img = batch[0]
        view_spec = batch[1]
        pred = preds[0]

        gt_img = view_img["img_astro"]  # (B, 5, H, W)
        gt_spec = view_spec["spec_astro"]  # (B, 4, L)
        img_valid = view_img["img_valid_mask"].bool()  # (B, H, W)
        spec_valid = view_spec["spec_valid_mask"].bool()  # (B, L)
        img_mask = view_img["img_mask_tokens"].bool()  # (B, H, W)
        spec_mask = view_spec["spec_mask_tokens"].bool()  # (B, L)
        ivar = view_spec.get("spec_ivar_weight", torch.ones_like(spec_valid, dtype=gt_spec.dtype))
        state = view_img["input_state"].long()  # 0=image-only, 1=spectrum-only, 2=both
        image_input_mask = view_img["image_input_mask"].bool()
        spectrum_input_mask = view_spec["spectrum_input_mask"].bool()

        # Stage-aware behavior (A: only self-reconstruction losses).
        epoch_idx = int(view_img["epoch_idx"].flatten()[0].item()) if "epoch_idx" in view_img else 0
        stage_a = epoch_idx < self.stage_a_epochs

        # 1) Image self reconstruction (masked).
        img_self_residual = _charbonnier(
            pred["pred_image_self"] - gt_img,
            eps=self.charbonnier_eps,
        )
        img_self_mask = img_valid & img_mask & image_input_mask[:, None, None]
        loss_img_self = _masked_mean(img_self_residual, img_self_mask)

        # 2) Spectrum self reconstruction (masked, ivar weighted).
        spec_self_residual = _huber(
            pred["pred_spectrum_self"] - gt_spec,
            delta=self.huber_delta,
        )
        spec_self_residual = spec_self_residual * ivar[:, None, :]
        spec_self_mask = spec_valid & spec_mask & spectrum_input_mask[:, None]
        loss_spec_self = _masked_mean(spec_self_residual, spec_self_mask)

        # 3) Cross-modal image->spectrum, supervised in image-only scenario.
        img_only_mask = state == 0
        i2s_residual = _huber(
            pred["pred_spectrum_from_image"] - gt_spec,
            delta=self.huber_delta,
        )
        i2s_residual = i2s_residual * ivar[:, None, :]
        i2s_mask = spec_valid & img_only_mask[:, None]
        loss_i2s = _masked_mean(i2s_residual, i2s_mask)

        # 4) Cross-modal spectrum->image, supervised in spectrum-only scenario.
        spec_only_mask = state == 1
        s2i_residual = _charbonnier(
            pred["pred_image_from_spectrum"] - gt_img,
            eps=self.charbonnier_eps,
        )
        s2i_mask = img_valid & spec_only_mask[:, None, None]
        loss_s2i = _masked_mean(s2i_residual, s2i_mask)

        # 5) Pair alignment (both-input samples).
        both_indices = torch.where(state == 2)[0]
        if both_indices.numel() >= 2:
            loss_align = self._info_nce(
                pred["latent_image_cls"][both_indices],
                pred["latent_spectrum_cls"][both_indices],
            )
        else:
            loss_align = gt_img.new_zeros(())

        # 6) Consistency between both-input and single-input reconstructions.
        if both_indices.numel() > 0:
            cons_spec = (pred["pred_spectrum_from_both"] - pred["pred_spectrum_from_image"].detach()).abs()
            cons_img = (pred["pred_image_from_both"] - pred["pred_image_from_spectrum"].detach()).abs()
            both_mask_spec = spec_valid & (state == 2)[:, None]
            both_mask_img = img_valid & (state == 2)[:, None, None]
            loss_cons = _masked_mean(cons_spec, both_mask_spec) + _masked_mean(cons_img, both_mask_img)
        else:
            loss_cons = gt_img.new_zeros(())

        # 7) Uncertainty regularization.
        loss_unc = pred["pred_logvar_image"].mean() + pred["pred_logvar_spectrum"].mean()

        if stage_a:
            total_loss = self.w_spec_self * loss_spec_self + self.w_img_self * loss_img_self
        else:
            total_loss = (
                self.w_i2s * loss_i2s
                + self.w_spec_self * loss_spec_self
                + self.w_s2i * loss_s2i
                + self.w_img_self * loss_img_self
                + self.w_align * loss_align
                + self.w_cons * loss_cons
                + self.w_unc * loss_unc
            )

        # Monitoring metrics.
        spec_mae = _masked_mean((pred["pred_spectrum"] - gt_spec).abs(), spec_valid)
        img_mae = _masked_mean((pred["pred_image"] - gt_img).abs(), img_valid)
        spec_corr = _masked_pearson(pred["pred_spectrum"], gt_spec, spec_valid)
        img_mse = _masked_mean((pred["pred_image"] - gt_img) ** 2, img_valid)
        img_psnr = 10.0 * torch.log10(1.0 / img_mse.clamp_min(1e-8))

        details = {
            "astro_total_loss": float(total_loss.detach()),
            "astro_loss_i2s": float(loss_i2s.detach()),
            "astro_loss_spec_self": float(loss_spec_self.detach()),
            "astro_loss_s2i": float(loss_s2i.detach()),
            "astro_loss_img_self": float(loss_img_self.detach()),
            "astro_loss_align": float(loss_align.detach()),
            "astro_loss_cons": float(loss_cons.detach()),
            "astro_loss_unc": float(loss_unc.detach()),
            "astro_stage_a": float(stage_a),
            "astro_metric_spec_mae": float(spec_mae.detach()),
            "astro_metric_img_mae": float(img_mae.detach()),
            "astro_metric_spec_corr": float(spec_corr.detach()),
            "astro_metric_img_psnr": float(img_psnr.detach()),
        }
        return total_loss, details
