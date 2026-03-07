import torch

from mapanything.models.astro_mapanything.model import AstroMapAnything
from mapanything.train.losses_astro import AstroBiModalSelfSupLoss


def _make_batch(batch_size: int, image_size: int, spec_length: int, state: int, epoch_idx: int):
    img = torch.randn(batch_size, 5, image_size, image_size)
    spec = torch.randn(batch_size, 4, spec_length)
    img_valid = torch.ones(batch_size, image_size, image_size, dtype=torch.bool)
    spec_valid = torch.ones(batch_size, spec_length, dtype=torch.bool)
    img_mask = torch.ones_like(img_valid)
    spec_mask = torch.ones_like(spec_valid)

    image_input_mask = torch.full((batch_size,), state in (0, 2), dtype=torch.bool)
    spectrum_input_mask = torch.full((batch_size,), state in (1, 2), dtype=torch.bool)
    state_tensor = torch.full((batch_size,), state, dtype=torch.long)
    epoch_tensor = torch.full((batch_size,), epoch_idx, dtype=torch.long)

    view_img = {
        "img_astro": img,
        "img_valid_mask": img_valid,
        "img_mask_tokens": img_mask,
        "image_input_mask": image_input_mask,
        "input_state": state_tensor,
        "meta_cond": torch.zeros(batch_size, 4),
        "meta_valid": torch.ones(batch_size, dtype=torch.bool),
        "epoch_idx": epoch_tensor,
    }
    view_spec = {
        "spec_astro": spec,
        "spec_valid_mask": spec_valid,
        "spec_mask_tokens": spec_mask,
        "spec_ivar_weight": torch.ones(batch_size, spec_length),
        "spectrum_input_mask": spectrum_input_mask,
        "input_state": state_tensor,
        "meta_cond": torch.zeros(batch_size, 4),
        "meta_valid": torch.ones(batch_size, dtype=torch.bool),
        "epoch_idx": epoch_tensor,
    }
    return [view_img, view_spec]


def test_patch7_specstride2_forward_shapes():
    model = AstroMapAnything(
        image_size=28,
        patch_size=7,
        spec_length=32,
        spec_stride=2,
        embed_dim=120,
        fusion_num_heads=12,
        fusion_num_layers=2,
        spectrum_encoder_layers=2,
        spectrum_decoder_layers=2,
        image_encoder_name="fallback_encoder",
        image_encoder_pretrained=False,
        use_metadata=True,
        contrast_proj_dim=64,
    )
    batch = _make_batch(batch_size=2, image_size=28, spec_length=32, state=2, epoch_idx=2)
    preds = model(batch)
    pred = preds[0]

    assert pred["pred_image"].shape == (2, 5, 28, 28)
    assert pred["pred_spectrum"].shape == (2, 4, 32)
    assert pred["latent_image_cls"].shape == (2, 120)
    assert pred["latent_spectrum_cls"].shape == (2, 120)
    assert pred["latent_image_proj"].shape == (2, 64)
    assert pred["latent_spectrum_proj"].shape == (2, 64)


def test_loss_stage_switch_and_disable_s2i():
    model = AstroMapAnything(
        image_size=28,
        patch_size=7,
        spec_length=32,
        spec_stride=2,
        embed_dim=120,
        fusion_num_heads=12,
        fusion_num_layers=2,
        spectrum_encoder_layers=2,
        spectrum_decoder_layers=2,
        image_encoder_name="fallback_encoder",
        image_encoder_pretrained=False,
        use_metadata=True,
        contrast_proj_dim=64,
    )
    criterion = AstroBiModalSelfSupLoss(
        stage_a_epochs=1,
        enable_s2i_loss=False,
        enable_s2i_consistency=False,
        use_uncertainty_nll=True,
        use_uncertainty_nll_self=True,
        use_uncertainty_nll_cross=False,
        use_peak_weighting=True,
    )

    # Stage A batch
    batch_a = _make_batch(batch_size=2, image_size=28, spec_length=32, state=2, epoch_idx=0)
    preds_a = model(batch_a)
    loss_a, details_a = criterion(batch_a, preds_a)
    assert torch.isfinite(loss_a)
    assert details_a["astro_stage_a"] == 1.0

    # Stage B batch (image-only should activate i2s)
    batch_b = _make_batch(batch_size=2, image_size=28, spec_length=32, state=0, epoch_idx=2)
    preds_b = model(batch_b)
    loss_b, details_b = criterion(batch_b, preds_b)
    assert torch.isfinite(loss_b)
    assert details_b["astro_stage_a"] == 0.0
    assert details_b["astro_loss_s2i"] == 0.0
    assert details_b["astro_loss_i2s"] >= 0.0


def _make_manual_batch_and_pred_for_peak_test(use_error_scale: float = 1.0):
    bsz, h, w, l = 1, 8, 8, 32
    gt_img = torch.zeros(bsz, 5, h, w)
    gt_spec = torch.zeros(bsz, 4, l)
    gt_spec[:, :, 16:] = 10.0

    state = torch.zeros(bsz, dtype=torch.long)  # image-only

    batch = [
        {
            "img_astro": gt_img,
            "img_valid_mask": torch.ones(bsz, h, w, dtype=torch.bool),
            "img_mask_tokens": torch.ones(bsz, h, w, dtype=torch.bool),
            "image_input_mask": torch.ones(bsz, dtype=torch.bool),
            "input_state": state,
            "meta_cond": torch.zeros(bsz, 4),
            "meta_valid": torch.ones(bsz, dtype=torch.bool),
            "epoch_idx": torch.full((bsz,), 3, dtype=torch.long),
        },
        {
            "spec_astro": gt_spec,
            "spec_valid_mask": torch.ones(bsz, l, dtype=torch.bool),
            "spec_mask_tokens": torch.ones(bsz, l, dtype=torch.bool),
            "spec_ivar_weight": torch.ones(bsz, l),
            "spectrum_input_mask": torch.zeros(bsz, dtype=torch.bool),
            "input_state": state,
            "meta_cond": torch.zeros(bsz, 4),
            "meta_valid": torch.ones(bsz, dtype=torch.bool),
            "epoch_idx": torch.full((bsz,), 3, dtype=torch.long),
        },
    ]

    pred_spec_i2s = gt_spec.clone()
    pred_spec_i2s[:, :, 16] += use_error_scale

    pred = {
        "pred_image": gt_img.clone(),
        "pred_spectrum": pred_spec_i2s.clone(),
        "pred_image_self": gt_img.clone(),
        "pred_spectrum_self": gt_spec.clone(),
        "pred_image_from_spectrum": gt_img.clone(),
        "pred_spectrum_from_image": pred_spec_i2s,
        "pred_image_from_both": gt_img.clone(),
        "pred_spectrum_from_both": gt_spec.clone(),
        "latent_image_cls": torch.zeros(bsz, 16),
        "latent_spectrum_cls": torch.zeros(bsz, 16),
        "latent_image_proj": torch.zeros(bsz, 8),
        "latent_spectrum_proj": torch.zeros(bsz, 8),
        "pred_logvar_image": torch.zeros(bsz),
        "pred_logvar_spectrum": torch.zeros(bsz),
    }
    return batch, [pred, pred]


def test_peak_weighting_increases_peak_bin_penalty():
    batch, preds = _make_manual_batch_and_pred_for_peak_test(use_error_scale=1.0)

    crit_plain = AstroBiModalSelfSupLoss(
        stage_a_epochs=0,
        enable_s2i_loss=False,
        enable_s2i_consistency=False,
        use_uncertainty_nll=False,
        use_uncertainty_nll_self=False,
        use_uncertainty_nll_cross=False,
        use_peak_weighting=False,
        w_align=0.0,
        w_cons=0.0,
    )
    _, details_plain = crit_plain(batch, preds)

    crit_peak = AstroBiModalSelfSupLoss(
        stage_a_epochs=0,
        enable_s2i_loss=False,
        enable_s2i_consistency=False,
        use_uncertainty_nll=False,
        use_uncertainty_nll_self=False,
        use_uncertainty_nll_cross=False,
        use_peak_weighting=True,
        peak_weight_alpha=2.0,
        peak_weight_clip=6.0,
        w_align=0.0,
        w_cons=0.0,
    )
    _, details_peak = crit_peak(batch, preds)

    assert details_peak["astro_loss_i2s"] > details_plain["astro_loss_i2s"]
