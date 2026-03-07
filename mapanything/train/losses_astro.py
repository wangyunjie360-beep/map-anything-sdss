"""
Losses for astronomy multi-modal self-supervised training.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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


def _expand_sample_scalar(sample_scalar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = sample_scalar
    while out.ndim < target.ndim:
        out = out.unsqueeze(-1)
    return out.expand_as(target)


def _heteroscedastic_nll(
    residual_loss: torch.Tensor,
    sample_logvar: torch.Tensor,
    mask: torch.Tensor,
    logvar_min: float,
    logvar_max: float,
    eps: float = 1e-6,
):
    if residual_loss.numel() == 0:
        zero = residual_loss.new_zeros(())
        return zero, zero
    clamped_logvar = sample_logvar.clamp(min=logvar_min, max=logvar_max)
    logvar = _expand_sample_scalar(clamped_logvar, residual_loss)
    # Heteroscedastic negative log-likelihood (up to a constant):
    # 0.5 * exp(-logvar) * rho + 0.5 * logvar
    nll = 0.5 * (torch.exp(-logvar) * residual_loss + logvar)
    return _masked_mean(nll, mask, eps=eps), clamped_logvar.mean()


def _spec_grad_abs(spec: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(spec)
    out[..., 1:] = (spec[..., 1:] - spec[..., :-1]).abs()
    return out


def _delta_spec(spec: torch.Tensor) -> torch.Tensor:
    return spec[..., 1:] - spec[..., :-1]


def _compute_peak_weight(
    gt_spec: torch.Tensor,
    spec_valid: torch.Tensor,
    alpha: float,
    clip_value: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_abs = _spec_grad_abs(gt_spec)
    valid_expanded = spec_valid[:, None, :].expand_as(grad_abs)
    masked_grad = torch.where(valid_expanded, grad_abs, torch.full_like(grad_abs, float("nan")))
    med = torch.nanmedian(masked_grad, dim=-1).values
    med = torch.nan_to_num(med, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)
    grad_norm = (grad_abs / med.unsqueeze(-1)).clamp(min=0.0, max=clip_value)
    weight = 1.0 + alpha * torch.tanh(grad_norm)
    weight = torch.where(valid_expanded, weight, torch.zeros_like(weight))
    return weight, grad_abs, valid_expanded


def _peak_recall_at_k(
    gt_spec: torch.Tensor,
    pred_spec: torch.Tensor,
    spec_valid: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    gt_grad = _spec_grad_abs(gt_spec).mean(dim=1)
    pred_grad = _spec_grad_abs(pred_spec).mean(dim=1)
    recalls = []
    bsz = gt_spec.shape[0]
    for b in range(bsz):
        valid_idx = torch.where(spec_valid[b])[0]
        if valid_idx.numel() < 4:
            continue
        gt_score = gt_grad[b, valid_idx]
        pred_score = pred_grad[b, valid_idx]
        k = max(1, int(round((1.0 - quantile) * float(valid_idx.numel()))))
        k = min(k, int(valid_idx.numel()))
        gt_top = torch.topk(gt_score, k=k, largest=True).indices
        pred_top = torch.topk(pred_score, k=k, largest=True).indices
        gt_bins = valid_idx[gt_top]
        pred_bins = valid_idx[pred_top]
        inter = torch.isin(pred_bins, gt_bins).sum().float()
        recalls.append(inter / float(k))
    if len(recalls) == 0:
        return gt_spec.new_zeros(())
    return torch.stack(recalls).mean()


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
        use_uncertainty_nll: bool = True,
        use_uncertainty_nll_self: bool | None = None,
        use_uncertainty_nll_cross: bool | None = None,
        enable_s2i_loss: bool = True,
        enable_s2i_consistency: bool = True,
        use_peak_weighting: bool = False,
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
    ):
        super().__init__()
        self.w_i2s = w_i2s
        self.w_spec_self = w_spec_self
        self.w_s2i = w_s2i
        self.w_img_self = w_img_self
        self.w_align = w_align
        self.w_cons = w_cons
        self.w_unc = w_unc
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
        logvar_image = pred["pred_logvar_image"]
        logvar_spectrum = pred["pred_logvar_spectrum"]

        # Peak-aware spectral weighting based on GT local gradient strength.
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

        # 1) Image self reconstruction (masked).
        img_self_residual = _charbonnier(
            pred["pred_image_self"] - gt_img,
            eps=self.charbonnier_eps,
        )
        img_self_mask = img_valid & img_mask & image_input_mask[:, None, None]
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

        # 2) Spectrum self reconstruction (masked, ivar + peak weighted).
        spec_self_residual = _huber(
            pred["pred_spectrum_self"] - gt_spec,
            delta=self.huber_delta,
        )
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

        # 3) Cross-modal image->spectrum, supervised in image-only scenario.
        img_only_mask = state == 0
        i2s_residual = _huber(
            pred["pred_spectrum_from_image"] - gt_spec,
            delta=self.huber_delta,
        )
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

        # 4) Cross-modal spectrum->image (optional), supervised in spectrum-only scenario.
        if self.enable_s2i_loss:
            spec_only_mask = state == 1
            s2i_residual = _charbonnier(
                pred["pred_image_from_spectrum"] - gt_img,
                eps=self.charbonnier_eps,
            )
            s2i_mask = img_valid & spec_only_mask[:, None, None]
            if self.use_uncertainty_nll_cross:
                loss_s2i, _ = _heteroscedastic_nll(
                    residual_loss=s2i_residual,
                    sample_logvar=logvar_image,
                    mask=s2i_mask,
                    logvar_min=self.logvar_min,
                    logvar_max=self.logvar_max,
                )
            else:
                loss_s2i = _masked_mean(s2i_residual, s2i_mask)
        else:
            loss_s2i = gt_img.new_zeros(())

        # 5) Pair alignment (both-input samples). Prefer projected embeddings when available.
        both_indices = torch.where(state == 2)[0]
        if both_indices.numel() >= 2:
            z_img = pred.get("latent_image_proj", pred["latent_image_cls"])
            z_spec = pred.get("latent_spectrum_proj", pred["latent_spectrum_cls"])
            loss_align = self._info_nce(
                z_img[both_indices],
                z_spec[both_indices],
            )
        else:
            loss_align = gt_img.new_zeros(())

        # 6) Consistency between both-input and single-input reconstructions.
        if both_indices.numel() > 0:
            cons_spec = (pred["pred_spectrum_from_both"] - pred["pred_spectrum_from_image"].detach()).abs()
            both_mask_spec = spec_valid & (state == 2)[:, None]
            loss_cons_spec = _masked_mean(cons_spec, both_mask_spec)

            if self.enable_s2i_consistency:
                cons_img = (pred["pred_image_from_both"] - pred["pred_image_from_spectrum"].detach()).abs()
                both_mask_img = img_valid & (state == 2)[:, None, None]
                loss_cons_img = _masked_mean(cons_img, both_mask_img)
            else:
                loss_cons_img = gt_img.new_zeros(())
            loss_cons = loss_cons_spec + loss_cons_img
        else:
            loss_cons = gt_img.new_zeros(())

        # 7) Spectral first-derivative losses to emphasize line/break localization.
        delta_gt = _delta_spec(gt_spec)
        delta_i2s = _delta_spec(pred["pred_spectrum_from_image"])
        delta_self = _delta_spec(pred["pred_spectrum_self"])
        delta_valid_base = spec_valid[:, 1:] & spec_valid[:, :-1]

        grad_weight = 0.5 * (spec_base_weight[..., 1:] + spec_base_weight[..., :-1])
        i2s_grad_residual = (delta_i2s - delta_gt).abs() * grad_weight
        i2s_grad_mask = delta_valid_base & img_only_mask[:, None]
        loss_i2s_grad = _masked_mean(i2s_grad_residual, i2s_grad_mask)

        spec_self_grad_residual = (delta_self - delta_gt).abs() * grad_weight
        spec_self_grad_mask = (
            delta_valid_base
            & spec_mask[:, 1:]
            & spec_mask[:, :-1]
            & spectrum_input_mask[:, None]
        )
        loss_spec_self_grad = _masked_mean(spec_self_grad_residual, spec_self_grad_mask)

        # 8) Uncertainty monitoring / legacy regularization.
        loss_unc = mean_logvar_image + mean_logvar_spectrum

        if stage_a:
            total_loss = (
                self.w_spec_self * loss_spec_self
                + self.w_img_self * loss_img_self
                + self.grad_loss_weight_spec_self * loss_spec_self_grad
            )
        else:
            total_loss = (
                self.w_i2s * loss_i2s
                + self.w_spec_self * loss_spec_self
                + self.w_img_self * loss_img_self
                + self.w_align * loss_align
                + self.w_cons * loss_cons
                + self.grad_loss_weight_i2s * loss_i2s_grad
                + self.grad_loss_weight_spec_self * loss_spec_self_grad
            )
            if self.enable_s2i_loss:
                total_loss = total_loss + self.w_s2i * loss_s2i
            # Legacy path: keep backward compatibility if uncertainty-NLL is disabled.
            if (
                not self.use_uncertainty_nll_self
                and not self.use_uncertainty_nll_cross
                and self.w_unc != 0.0
            ):
                total_loss = total_loss + self.w_unc * loss_unc

        # Monitoring metrics.
        pred_spec_mixed = pred["pred_spectrum"]
        spec_mae = _masked_mean((pred_spec_mixed - gt_spec).abs(), spec_valid)
        img_mae = _masked_mean((pred["pred_image"] - gt_img).abs(), img_valid)
        spec_corr = _masked_pearson(pred_spec_mixed, gt_spec, spec_valid)
        img_mse = _masked_mean((pred["pred_image"] - gt_img) ** 2, img_valid)
        img_psnr = 10.0 * torch.log10(1.0 / img_mse.clamp_min(1e-8))

        masked_grad = torch.where(
            spec_valid_expanded,
            spec_grad_gt_abs,
            torch.full_like(spec_grad_gt_abs, float("nan")),
        )
        peak_threshold = torch.nanquantile(
            masked_grad,
            q=self.peak_quantile_monitor,
            dim=-1,
            keepdim=True,
        )
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
            "astro_loss_unc": float(loss_unc.detach()),
            "astro_loss_i2s_grad": float(loss_i2s_grad.detach()),
            "astro_loss_spec_self_grad": float(loss_spec_self_grad.detach()),
            "astro_logvar_image": float(mean_logvar_image.detach()),
            "astro_logvar_spectrum": float(mean_logvar_spectrum.detach()),
            "astro_use_uncertainty_nll_self": float(self.use_uncertainty_nll_self),
            "astro_use_uncertainty_nll_cross": float(self.use_uncertainty_nll_cross),
            "astro_enable_s2i_loss": float(self.enable_s2i_loss),
            "astro_enable_s2i_consistency": float(self.enable_s2i_consistency),
            "astro_use_peak_weighting": float(self.use_peak_weighting),
            "astro_stage_a": float(stage_a),
            "astro_metric_spec_mae": float(spec_mae.detach()),
            "astro_metric_spec_mae_peak": float(spec_mae_peak.detach()),
            "astro_metric_spec_grad_mae": float(spec_grad_mae.detach()),
            "astro_metric_spec_peak_recall": float(spec_peak_recall.detach()),
            "astro_metric_img_mae": float(img_mae.detach()),
            "astro_metric_spec_corr": float(spec_corr.detach()),
            "astro_metric_img_psnr": float(img_psnr.detach()),
        }
        return total_loss, details
