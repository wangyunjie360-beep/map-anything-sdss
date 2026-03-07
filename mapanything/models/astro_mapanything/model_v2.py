"""
AstroMapAnything v2 with masked students, shared backbone outputs, and auxiliary spectrum heads.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .model import AstroMapAnything


class AstroMapAnythingV2(AstroMapAnything):
    def __init__(
        self,
        name: str = "astro_mapanything_v2",
        enable_aux_spectrum_head: bool = True,
        aux_spectrum_channels: int = 2,
        enable_student_teacher_distill: bool = True,
        **kwargs,
    ):
        self.enable_aux_spectrum_head = enable_aux_spectrum_head
        self.aux_spectrum_channels = aux_spectrum_channels
        self.enable_student_teacher_distill = enable_student_teacher_distill
        super().__init__(name=name, **kwargs)

        if self.enable_aux_spectrum_head:
            stride = self.spec_length // self.spec_tokens
            self.spec_aux_out = nn.Linear(self.embed_dim, self.aux_spectrum_channels * stride)
        else:
            self.spec_aux_out = None

    def _decode_spectrum_with_head(
        self,
        spec_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        out_head: nn.Linear,
        out_channels: int,
    ) -> torch.Tensor:
        bsz = spec_tokens.shape[0]
        tgt = spec_tokens + self.spec_dec_pos
        dec = self.spec_decoder(tgt=tgt, memory=memory_tokens)
        out = out_head(dec)
        stride = self.spec_length // self.spec_tokens
        out = out.view(bsz, self.spec_tokens, stride, out_channels)
        out = out.reshape(bsz, self.spec_length, out_channels).permute(0, 2, 1).contiguous()
        return out

    def _fuse_decode(
        self,
        image_tokens: torch.Tensor,
        spectrum_tokens: torch.Tensor,
        meta_token: torch.Tensor,
        image_available: torch.Tensor,
        spectrum_available: torch.Tensor,
        image_register_tokens: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        out = super()._fuse_decode(
            image_tokens=image_tokens,
            spectrum_tokens=spectrum_tokens,
            meta_token=meta_token,
            image_available=image_available,
            spectrum_available=spectrum_available,
            image_register_tokens=image_register_tokens,
        )
        if self.enable_aux_spectrum_head:
            spec_tokens_for_head = spectrum_tokens
            image_available_view = image_available.view(spectrum_tokens.shape[0], 1, 1).to(
                dtype=torch.bool,
                device=spectrum_tokens.device,
            )
            spectrum_available_view = spectrum_available.view(spectrum_tokens.shape[0], 1, 1).to(
                dtype=torch.bool,
                device=spectrum_tokens.device,
            )
            image_in = torch.where(
                image_available_view,
                image_tokens,
                self.image_mask_token.expand(spectrum_tokens.shape[0], self.image_tokens, self.embed_dim),
            )
            spectrum_in = torch.where(
                spectrum_available_view,
                spectrum_tokens,
                self.spec_mask_token.expand(spectrum_tokens.shape[0], self.spec_tokens, self.embed_dim),
            )
            image_in = image_in + self.modality_type_tokens[0].view(1, 1, -1)
            spectrum_in = spectrum_in + self.modality_type_tokens[1].view(1, 1, -1)
            meta_in = meta_token + self.modality_type_tokens[2].view(1, 1, -1)
            fused = torch.cat([image_in, spectrum_in, meta_in], dim=1)
            if self.enable_fusion_gradient_checkpointing and self.training:
                fused = torch.utils.checkpoint.checkpoint(self.fusion, fused, use_reentrant=False)
            else:
                fused = self.fusion(fused)
            spectrum_latent = fused[:, self.image_tokens : self.image_tokens + self.spec_tokens, :]
            out["pred_spectrum_aux"] = self._decode_spectrum_with_head(
                spectrum_latent,
                fused,
                self.spec_aux_out,
                self.aux_spectrum_channels,
            )
        return out

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        mode: str = "train",
    ) -> List[Dict[str, torch.Tensor]]:
        if len(views) != 2:
            raise ValueError(f"AstroMapAnythingV2 expects 2 views, got {len(views)}")

        view_image, view_spec = views
        image = view_image["img_astro"]
        spectrum = view_spec["spec_astro"]
        state = view_image["input_state"].long()
        bsz = image.shape[0]

        if mode == "infer":
            image_encoded = self.image_encoder(image)
            image_tokens = image_encoded["patch_tokens"]
            image_register_tokens = image_encoded.get("register_tokens")
            spectrum_tokens = self.spectrum_encoder(spectrum)
            meta_token = self._encode_metadata(
                meta_cond=view_image["meta_cond"],
                meta_valid=view_image.get("meta_valid"),
            )
            ones = torch.ones((bsz,), dtype=torch.bool, device=image.device)
            zeros = torch.zeros((bsz,), dtype=torch.bool, device=image.device)
            out_image_only = self._fuse_decode(
                image_tokens,
                spectrum_tokens,
                meta_token,
                ones,
                zeros,
                image_register_tokens=image_register_tokens,
            )
            out_spectrum_only = self._fuse_decode(
                image_tokens,
                spectrum_tokens,
                meta_token,
                zeros,
                ones,
                image_register_tokens=image_register_tokens,
            )
            out_both = self._fuse_decode(
                image_tokens,
                spectrum_tokens,
                meta_token,
                ones,
                ones,
                image_register_tokens=image_register_tokens,
            )
            image_available = view_image.get("image_input_mask", ones).to(dtype=torch.bool)
            spectrum_available = view_spec.get("spectrum_input_mask", ones).to(dtype=torch.bool)
            out_infer = self._fuse_decode(
                image_tokens,
                spectrum_tokens,
                meta_token,
                image_available,
                spectrum_available,
                image_register_tokens=image_register_tokens,
            )
            pred_dict = dict(out_infer)
            pred_dict["input_state"] = state
            pred_dict["pred_image_from_spectrum"] = out_spectrum_only["pred_image"]
            pred_dict["pred_spectrum_from_image"] = out_image_only["pred_spectrum"]
            pred_dict["pred_image_from_both"] = out_both["pred_image"]
            pred_dict["pred_spectrum_from_both"] = out_both["pred_spectrum"]
            pred_dict["z_shared_from_image"] = out_image_only["z_shared"]
            pred_dict["z_shared_from_spectrum"] = out_spectrum_only["z_shared"]
            pred_dict["z_shared_from_both"] = out_both["z_shared"]
            if self.enable_aux_spectrum_head:
                pred_dict["pred_spectrum_aux_from_image"] = out_image_only["pred_spectrum_aux"]
                pred_dict["pred_spectrum_aux_from_both"] = out_both["pred_spectrum_aux"]
            return [dict(pred_dict), dict(pred_dict)]

        image_student = view_image.get("img_masked_astro", image)
        spectrum_student = view_spec.get("spec_masked_astro", spectrum)

        image_teacher_encoded = self.image_encoder(image)
        image_tokens_teacher = image_teacher_encoded["patch_tokens"]
        image_register_tokens_teacher = image_teacher_encoded.get("register_tokens")
        spectrum_tokens_teacher = self.spectrum_encoder(spectrum)

        if "img_masked_astro" in view_image:
            image_student_encoded = self.image_encoder(image_student)
        else:
            image_student_encoded = image_teacher_encoded
        image_tokens_student = image_student_encoded["patch_tokens"]
        image_register_tokens_student = image_student_encoded.get("register_tokens")

        if "spec_masked_astro" in view_spec:
            spectrum_tokens_student = self.spectrum_encoder(spectrum_student)
        else:
            spectrum_tokens_student = spectrum_tokens_teacher

        meta_token = self._encode_metadata(
            meta_cond=view_image["meta_cond"],
            meta_valid=view_image.get("meta_valid"),
        )
        ones = torch.ones((bsz,), dtype=torch.bool, device=image.device)
        zeros = torch.zeros((bsz,), dtype=torch.bool, device=image.device)

        out_image_only_student = self._fuse_decode(
            image_tokens=image_tokens_student,
            spectrum_tokens=spectrum_tokens_student,
            meta_token=meta_token,
            image_available=ones,
            spectrum_available=zeros,
            image_register_tokens=image_register_tokens_student,
        )
        out_spectrum_only_student = self._fuse_decode(
            image_tokens=image_tokens_student,
            spectrum_tokens=spectrum_tokens_student,
            meta_token=meta_token,
            image_available=zeros,
            spectrum_available=ones,
            image_register_tokens=image_register_tokens_student,
        )
        out_both_teacher = self._fuse_decode(
            image_tokens=image_tokens_teacher,
            spectrum_tokens=spectrum_tokens_teacher,
            meta_token=meta_token,
            image_available=ones,
            spectrum_available=ones,
            image_register_tokens=image_register_tokens_teacher,
        )
        out_image_only_teacher = self._fuse_decode(
            image_tokens=image_tokens_teacher,
            spectrum_tokens=spectrum_tokens_teacher,
            meta_token=meta_token,
            image_available=ones,
            spectrum_available=zeros,
            image_register_tokens=image_register_tokens_teacher,
        )
        out_spectrum_only_teacher = self._fuse_decode(
            image_tokens=image_tokens_teacher,
            spectrum_tokens=spectrum_tokens_teacher,
            meta_token=meta_token,
            image_available=zeros,
            spectrum_available=ones,
            image_register_tokens=image_register_tokens_teacher,
        )

        pred_image_mixed = self._select_by_state(
            state=state,
            image_only_tensor=out_image_only_student["pred_image"],
            spectrum_only_tensor=out_spectrum_only_student["pred_image"],
            both_tensor=out_both_teacher["pred_image"],
        )
        pred_spectrum_mixed = self._select_by_state(
            state=state,
            image_only_tensor=out_image_only_student["pred_spectrum"],
            spectrum_only_tensor=out_spectrum_only_student["pred_spectrum"],
            both_tensor=out_both_teacher["pred_spectrum"],
        )

        pred_dict = {
            "pred_image": pred_image_mixed,
            "pred_spectrum": pred_spectrum_mixed,
            "pred_image_self": out_image_only_student["pred_image"],
            "pred_spectrum_self": out_spectrum_only_student["pred_spectrum"],
            "pred_image_from_spectrum": out_spectrum_only_student["pred_image"],
            "pred_spectrum_from_image": out_image_only_student["pred_spectrum"],
            "pred_image_from_both": out_both_teacher["pred_image"],
            "pred_spectrum_from_both": out_both_teacher["pred_spectrum"],
            "pred_nuisance": out_both_teacher["pred_nuisance"],
            "z_img": out_both_teacher["z_img"],
            "z_spec": out_both_teacher["z_spec"],
            "z_shared": out_both_teacher["z_shared"],
            "z_nuis": out_both_teacher["z_nuis"],
            "z_shared_from_image": out_image_only_student["z_shared"],
            "z_shared_from_spectrum": out_spectrum_only_student["z_shared"],
            "z_shared_from_both": out_both_teacher["z_shared"],
            "latent_image_cls": out_both_teacher["latent_image_cls"],
            "latent_spectrum_cls": out_both_teacher["latent_spectrum_cls"],
            "latent_shared_cls": out_both_teacher["latent_shared_cls"],
            "latent_image_proj": out_both_teacher["latent_image_proj"],
            "latent_spectrum_proj": out_both_teacher["latent_spectrum_proj"],
            "latent_shared_proj": out_both_teacher["latent_shared_proj"],
            "pred_logvar_image": out_both_teacher["pred_logvar_image"],
            "pred_logvar_spectrum": out_both_teacher["pred_logvar_spectrum"],
            "latent_image_proj_student": out_image_only_student["latent_image_proj"],
            "latent_image_proj_teacher": out_image_only_teacher["latent_image_proj"],
            "latent_spectrum_proj_student": out_spectrum_only_student["latent_spectrum_proj"],
            "latent_spectrum_proj_teacher": out_spectrum_only_teacher["latent_spectrum_proj"],
            "latent_image_cls_student": out_image_only_student["latent_image_cls"],
            "latent_image_cls_teacher": out_image_only_teacher["latent_image_cls"],
            "latent_spectrum_cls_student": out_spectrum_only_student["latent_spectrum_cls"],
            "latent_spectrum_cls_teacher": out_spectrum_only_teacher["latent_spectrum_cls"],
            "input_state": state,
        }
        if self.enable_aux_spectrum_head:
            pred_spectrum_aux_mixed = self._select_by_state(
                state=state,
                image_only_tensor=out_image_only_student["pred_spectrum_aux"],
                spectrum_only_tensor=out_spectrum_only_student["pred_spectrum_aux"],
                both_tensor=out_both_teacher["pred_spectrum_aux"],
            )
            pred_dict.update(
                {
                    "pred_spectrum_aux": pred_spectrum_aux_mixed,
                    "pred_spectrum_aux_self": out_spectrum_only_student["pred_spectrum_aux"],
                    "pred_spectrum_aux_from_image": out_image_only_student["pred_spectrum_aux"],
                    "pred_spectrum_aux_from_both": out_both_teacher["pred_spectrum_aux"],
                }
            )

        return [dict(pred_dict), dict(pred_dict)]
