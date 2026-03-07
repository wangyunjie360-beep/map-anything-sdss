"""
Astro loss v2: shared-backbone training with auxiliary image-to-spectrum reconstruction.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapanything.train.losses_astro import (
    _charbonnier,
    _compute_peak_weight,
    _delta_spec,
    _heteroscedastic_nll,
    _huber,
    _masked_mean,
    _masked_pearson,
    _peak_recall_at_k,
    _spec_grad_abs,
)


def _cosine_distill(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.numel() == 0:
        return student.new_zeros(())
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    return 1.0 - (student * teacher).sum(dim=-1).mean()


def _apply_channel_weights(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        return value
    shape = [1, value.shape[1]] + [1] * (value.ndim - 2)
    return value * weights.view(*shape).to(dtype=value.dtype, device=value.device)


def _orthogonality_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros(())
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    cosine = (a * b).sum(dim=-1)
    return (cosine**2).mean()


class AstroBiModalSelfSupLossV2(nn.Module):
    def __init__(
        self,
        w_i2s: float = 1.00,
        w_spec_self: float = 0.60,
        w_s2i: float = 0.0,
        w_img_self: float = 0.10,
        w_align: float = 0.25,
        w_cons: float = 0.20,
        w_unc: float = 0.0,
        w_nuis: float = 0.10,
        w_disentangle: float = 0.05,
        use_uncertainty_nll: bool = True,
        use_uncertainty_nll_self: bool | None = None,
        use_uncertainty_nll_cross: bool | None = None,
        enable_s2i_loss: bool = False,
        enable_s2i_consistency: bool = False,
        use_peak_weighting: bool = True,
        peak_weight_alpha: float = 2.0,
        peak_weight_clip: float = 6.0,
        grad_loss_weight_i2s: float = 0.35,
        grad_loss_weight_spec_self: float = 0.20,
        peak_quantile_monitor: float = 0.90,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        align_temperature: float = 0.07,
        charbonnier_eps: float = 1e-3,
        huber_delta: float = 1.0,
        stage_a_epochs: int = 20,
        enable_aux_spectrum_loss: bool = True,
        w_aux_i2s: float = 0.25,
        w_aux_spec_self: float = 0.10,
        aux_channel_weights: Tuple[float, float] = (0.5, 1.0),
        enable_latent_self_distill: bool = True,
        w_distill_image: float = 0.05,
        w_distill_spectrum: float = 0.05,
        spec_channel_weights: Tuple[float, float, float, float] = (1.0, 1.0, 0.8, 0.9),
    ):
        super().__init__()
        self.w_i2s = w_i2s
        self.w_spec_self = w_spec_self
        self.w_s2i = w_s2i
        self.w_img_self = w_img_self
        self.w_align = w_align
        self.w_cons = w_cons
        self.w_unc = w_unc
        self.w_nuis = w_nuis
        self.w_disentangle = w_disentangle
        self.use_uncertainty_nll_self = (
            use_uncertainty_nll if use_uncertainty_nll_self is None else use_uncertainty_nll_self
        )
        self.use_uncertainty_nll_cross = (
            use_uncertainty_nll if use_uncertainty_nll_cross is None else use_uncertainty_nll_cross
        )
        self.enable_s2i_loss = enable_s2i_loss
        self.enable_s2i_consistency = enable_s2i_consistency
        self.use_peak_weighting = use_peak_weighting
        self.peak_weight_alpha = peak_weight_alpha
        self.peak_weight_clip = peak_weight_clip
        self.grad_loss_weight_i2s = grad_loss_weight_i2s
        self.grad_loss_weight_spec_self = grad_loss_weight_spec_self
        self.peak_quantile_monitor = peak_quantile_monitor
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.align_temperature = align_temperature
        self.charbonnier_eps = charbonnier_eps
        self.huber_delta = huber_delta
        self.stage_a_epochs = stage_a_epochs
        self.enable_aux_spectrum_loss = enable_aux_spectrum_loss
        self.w_aux_i2s = w_aux_i2s
        self.w_aux_spec_self = w_aux_spec_self
        self.enable_latent_self_distill = enable_latent_self_distill
        self.w_distill_image = w_distill_image
        self.w_distill_spectrum = w_distill_spectrum
        self.register_buffer("aux_channel_weights_tensor", torch.tensor(aux_channel_weights, dtype=torch.float32))
        self.register_buffer("spec_channel_weights_tensor", torch.tensor(spec_channel_weights, dtype=torch.float32))

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
        view_img = batch[0]
        view_spec = batch[1]
        pred = preds[0]

        gt_img = view_img["img_astro"]
        gt_spec = view_spec["spec_astro"]
        gt_spec_aux = view_spec.get("spec_aux_targets")
        gt_nuisance = view_img.get("img_nuisance_targets")
        img_valid = view_img["img_valid_mask"].bool()
        spec_valid = view_spec["spec_valid_mask"].bool()
        img_mask = view_img["img_mask_tokens"].bool()
        spec_mask = view_spec["spec_mask_tokens"].bool()
        ivar = view_spec.get("spec_ivar_weight", torch.ones_like(spec_valid, dtype=gt_spec.dtype))
        state = view_img["input_state"].long()
        image_input_mask = view_img["image_input_mask"].bool()
        spectrum_input_mask = view_spec["spectrum_input_mask"].bool()

        epoch_idx = int(view_img["epoch_idx"].flatten()[0].item()) if "epoch_idx" in view_img else 0
        stage_a = epoch_idx < self.stage_a_epochs
        logvar_image = pred["pred_logvar_image"]
        logvar_spectrum = pred["pred_logvar_spectrum"]

        if self.use_peak_weighting:
            peak_weight, spec_grad_gt_abs, spec_valid_expanded = _compute_peak_weight(
                gt_spec=gt_spec,
                spec_valid=spec_valid,
                alpha=self.peak_weight_alpha,
                clip_value=self.peak_weight_clip,
            )
        else:
            spec_valid_expanded = spec_valid[:, None, :].expand_as(gt_spec)
            spec_grad_gt_abs = _spec_grad_abs(gt_spec)
            peak_weight = torch.where(spec_valid_expanded, torch.ones_like(gt_spec), torch.zeros_like(gt_spec))

        ivar_expanded = ivar[:, None, :]
        spec_base_weight = ivar_expanded * peak_weight
        spec_base_weight = _apply_channel_weights(spec_base_weight, self.spec_channel_weights_tensor)

        img_self_residual = _charbonnier(pred["pred_image_self"] - gt_img, eps=self.charbonnier_eps)
        img_valid_expanded = img_valid[:, None, :, :].expand_as(gt_img)
        img_self_mask = img_valid_expanded & img_mask & image_input_mask[:, None, None, None]
        if self.use_uncertainty_nll_self:
            loss_img_self, mean_logvar_image = _heteroscedastic_nll(
                residual_loss=img_self_residual,
                sample_logvar=logvar_image,
                mask=img_self_mask,
                logvar_min=self.logvar_min,
                logvar_max=self.logvar_max,
            )
        else:
            loss_img_self = _masked_mean(img_self_residual, img_self_mask)
            mean_logvar_image = logvar_image.mean()

        spec_self_residual = _huber(pred["pred_spectrum_self"] - gt_spec, delta=self.huber_delta)
        spec_self_residual = spec_self_residual * spec_base_weight
        spec_self_mask = spec_valid & spec_mask & spectrum_input_mask[:, None]
        if self.use_uncertainty_nll_self:
            loss_spec_self, mean_logvar_spectrum = _heteroscedastic_nll(
                residual_loss=spec_self_residual,
                sample_logvar=logvar_spectrum,
                mask=spec_self_mask,
                logvar_min=self.logvar_min,
                logvar_max=self.logvar_max,
            )
        else:
            loss_spec_self = _masked_mean(spec_self_residual, spec_self_mask)
            mean_logvar_spectrum = logvar_spectrum.mean()

        img_only_mask = state == 0
        i2s_residual = _huber(pred["pred_spectrum_from_image"] - gt_spec, delta=self.huber_delta)
        i2s_residual = i2s_residual * spec_base_weight
        i2s_mask = spec_valid & img_only_mask[:, None]
        if self.use_uncertainty_nll_cross:
            loss_i2s, _ = _heteroscedastic_nll(
                residual_loss=i2s_residual,
                sample_logvar=logvar_spectrum,
                mask=i2s_mask,
                logvar_min=self.logvar_min,
                logvar_max=self.logvar_max,
            )
        else:
            loss_i2s = _masked_mean(i2s_residual, i2s_mask)

        loss_s2i = gt_img.new_zeros(())
        loss_align = self._info_nce(pred["z_img"], pred["z_spec"])
        loss_cons_image = _cosine_distill(pred["z_shared_from_image"], pred["z_shared_from_both"])
        loss_cons_spectrum = _cosine_distill(
            pred["z_shared_from_spectrum"],
            pred["z_shared_from_both"],
        )
        loss_cons = 0.5 * (loss_cons_image + loss_cons_spectrum)
        loss_disentangle = _orthogonality_penalty(pred["z_shared"], pred["z_nuis"])

        delta_gt = _delta_spec(gt_spec)
        delta_i2s = _delta_spec(pred["pred_spectrum_from_image"])
        delta_self = _delta_spec(pred["pred_spectrum_self"])
        delta_valid_base = spec_valid[:, 1:] & spec_valid[:, :-1]
        grad_weight = 0.5 * (spec_base_weight[..., 1:] + spec_base_weight[..., :-1])
        i2s_grad_residual = (delta_i2s - delta_gt).abs() * grad_weight
        i2s_grad_mask = delta_valid_base & img_only_mask[:, None]
        loss_i2s_grad = _masked_mean(i2s_grad_residual, i2s_grad_mask)

        spec_self_grad_residual = (delta_self - delta_gt).abs() * grad_weight
        spec_self_grad_mask = delta_valid_base & spec_mask[:, 1:] & spec_mask[:, :-1] & spectrum_input_mask[:, None]
        loss_spec_self_grad = _masked_mean(spec_self_grad_residual, spec_self_grad_mask)

        loss_aux_i2s = gt_spec.new_zeros(())
        loss_aux_spec_self = gt_spec.new_zeros(())
        aux_flux_rest_mae = gt_spec.new_zeros(())
        if self.enable_aux_spectrum_loss and gt_spec_aux is not None and "pred_spectrum_aux_from_image" in pred:
            aux_weights = self.aux_channel_weights_tensor.to(dtype=gt_spec_aux.dtype, device=gt_spec_aux.device)
            aux_i2s_residual = _apply_channel_weights(
                _huber(pred["pred_spectrum_aux_from_image"] - gt_spec_aux, delta=self.huber_delta),
                aux_weights,
            )
            aux_i2s_mask = spec_valid & img_only_mask[:, None]
            loss_aux_i2s = _masked_mean(aux_i2s_residual, aux_i2s_mask)

            aux_self_residual = _apply_channel_weights(
                _huber(pred["pred_spectrum_aux_self"] - gt_spec_aux, delta=self.huber_delta),
                aux_weights,
            )
            aux_self_mask = spec_valid & spec_mask & spectrum_input_mask[:, None]
            loss_aux_spec_self = _masked_mean(aux_self_residual, aux_self_mask)
            aux_flux_rest_mae = _masked_mean(
                (pred["pred_spectrum_aux"][:, 1] - gt_spec_aux[:, 1]).abs(),
                spec_valid,
            )

        loss_nuis = gt_img.new_zeros(())
        nuis_mae = gt_img.new_zeros(())
        nuis_bg_mae = gt_img.new_zeros(())
        nuis_noise_mae = gt_img.new_zeros(())
        nuis_quality_mae = gt_img.new_zeros(())
        if gt_nuisance is not None and "pred_nuisance" in pred:
            nuisance_residual = _huber(pred["pred_nuisance"] - gt_nuisance, delta=self.huber_delta)
            loss_nuis = nuisance_residual.mean()
            nuis_abs = (pred["pred_nuisance"] - gt_nuisance).abs()
            nuis_mae = nuis_abs.mean()
            nuis_bg_mae = nuis_abs[:, :5].mean()
            nuis_noise_mae = nuis_abs[:, 5:10].mean()
            nuis_quality_mae = nuis_abs[:, 10:].mean()

        loss_distill_image = gt_img.new_zeros(())
        loss_distill_spectrum = gt_img.new_zeros(())
        if self.enable_latent_self_distill and "latent_image_proj_student" in pred:
            loss_distill_image = _cosine_distill(
                pred["latent_image_proj_student"],
                pred["latent_image_proj_teacher"],
            )
            loss_distill_spectrum = _cosine_distill(
                pred["latent_spectrum_proj_student"],
                pred["latent_spectrum_proj_teacher"],
            )

        loss_unc = mean_logvar_image + mean_logvar_spectrum
        total_backbone = loss_align + loss_cons + loss_disentangle

        if stage_a:
            total_loss = (
                self.w_align * loss_align
                + self.w_cons * loss_cons
                + self.w_disentangle * loss_disentangle
                + self.w_spec_self * loss_spec_self
                + self.w_img_self * loss_img_self
                + self.grad_loss_weight_spec_self * loss_spec_self_grad
                + self.w_aux_spec_self * loss_aux_spec_self
                + self.w_nuis * loss_nuis
                + self.w_distill_image * loss_distill_image
                + self.w_distill_spectrum * loss_distill_spectrum
            )
        else:
            total_loss = (
                self.w_align * loss_align
                + self.w_cons * loss_cons
                + self.w_disentangle * loss_disentangle
                + self.w_i2s * loss_i2s
                + self.w_spec_self * loss_spec_self
                + self.w_img_self * loss_img_self
                + self.grad_loss_weight_i2s * loss_i2s_grad
                + self.grad_loss_weight_spec_self * loss_spec_self_grad
                + self.w_aux_i2s * loss_aux_i2s
                + self.w_aux_spec_self * loss_aux_spec_self
                + self.w_nuis * loss_nuis
                + self.w_distill_image * loss_distill_image
                + self.w_distill_spectrum * loss_distill_spectrum
            )

        pred_spec_mixed = pred["pred_spectrum"]
        spec_mae = _masked_mean((pred_spec_mixed - gt_spec).abs(), spec_valid)
        img_mae = _masked_mean((pred["pred_image"] - gt_img).abs(), img_valid)
        spec_corr = _masked_pearson(pred_spec_mixed, gt_spec, spec_valid)
        img_mse = _masked_mean((pred["pred_image"] - gt_img) ** 2, img_valid)
        img_psnr = 10.0 * torch.log10(1.0 / img_mse.clamp_min(1e-8))

        masked_grad = torch.where(spec_valid_expanded, spec_grad_gt_abs, torch.full_like(spec_grad_gt_abs, float("nan")))
        peak_threshold = torch.nanquantile(masked_grad, q=self.peak_quantile_monitor, dim=-1, keepdim=True)
        peak_mask = spec_valid_expanded & (spec_grad_gt_abs >= peak_threshold)
        spec_mae_peak = _masked_mean((pred_spec_mixed - gt_spec).abs(), peak_mask)

        pred_grad_main = _delta_spec(pred_spec_mixed)
        spec_grad_mae = _masked_mean((pred_grad_main - delta_gt).abs(), delta_valid_base)
        spec_peak_recall = _peak_recall_at_k(
            gt_spec=gt_spec,
            pred_spec=pred_spec_mixed,
            spec_valid=spec_valid,
            quantile=self.peak_quantile_monitor,
        )

        details = {
            "astro_total_loss": float(total_loss.detach()),
            "astro_loss_i2s": float(loss_i2s.detach()),
            "astro_loss_spec_self": float(loss_spec_self.detach()),
            "astro_loss_s2i": float(loss_s2i.detach()),
            "astro_loss_img_self": float(loss_img_self.detach()),
            "astro_loss_align": float(loss_align.detach()),
            "astro_loss_cons": float(loss_cons.detach()),
            "astro_loss_shared_cons": float(loss_cons.detach()),
            "astro_loss_unc": float(loss_unc.detach()),
            "astro_loss_nuis": float(loss_nuis.detach()),
            "astro_loss_disentangle": float(loss_disentangle.detach()),
            "astro_loss_i2s_grad": float(loss_i2s_grad.detach()),
            "astro_loss_spec_self_grad": float(loss_spec_self_grad.detach()),
            "astro_loss_aux_i2s": float(loss_aux_i2s.detach()),
            "astro_loss_aux_spec_self": float(loss_aux_spec_self.detach()),
            "astro_loss_distill_image": float(loss_distill_image.detach()),
            "astro_loss_distill_spectrum": float(loss_distill_spectrum.detach()),
            "astro_logvar_image": float(mean_logvar_image.detach()),
            "astro_logvar_spectrum": float(mean_logvar_spectrum.detach()),
            "astro_metric_backbone": float(total_backbone.detach()),
            "astro_metric_spec_mae": float(spec_mae.detach()),
            "astro_metric_spec_mae_peak": float(spec_mae_peak.detach()),
            "astro_metric_spec_grad_mae": float(spec_grad_mae.detach()),
            "astro_metric_spec_peak_recall": float(spec_peak_recall.detach()),
            "astro_metric_img_mae": float(img_mae.detach()),
            "astro_metric_spec_corr": float(spec_corr.detach()),
            "astro_metric_img_psnr": float(img_psnr.detach()),
            "astro_metric_flux_rest_aux_mae": float(aux_flux_rest_mae.detach()),
            "astro_metric_nuis_mae": float(nuis_mae.detach()),
            "astro_metric_nuis_bg_mae": float(nuis_bg_mae.detach()),
            "astro_metric_nuis_noise_mae": float(nuis_noise_mae.detach()),
            "astro_metric_nuis_quality_mae": float(nuis_quality_mae.detach()),
            "astro_stage_a": float(stage_a),
            "astro_enable_aux_spectrum_loss": float(self.enable_aux_spectrum_loss),
            "astro_enable_latent_self_distill": float(self.enable_latent_self_distill),
        }
        return total_loss, details
