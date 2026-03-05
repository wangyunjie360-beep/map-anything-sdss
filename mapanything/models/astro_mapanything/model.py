"""
AstroMapAnything: dual-modal (image + spectrum) self-supervised model.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class _AstroImageEncoder(nn.Module):
    """
    Image token encoder.

    Preferred path uses DINOv2 ViT-L/14 from torch hub.
    Fallback path uses a lightweight Conv + Transformer stack.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        encoder_name: str = "dinov2_vitl14",
        use_pretrained: bool = True,
        freeze_first_n_blocks: int = 12,
        torch_hub_force_reload: bool = False,
        dino_offline_mode: bool = False,
        dino_local_repo_path: Optional[str] = None,
        dino_local_checkpoint_path: Optional[str] = None,
        dino_allow_network_fallback: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_name = encoder_name
        self.use_pretrained = use_pretrained
        self.freeze_first_n_blocks = freeze_first_n_blocks
        self.torch_hub_force_reload = torch_hub_force_reload
        self.dino_offline_mode = dino_offline_mode
        self.dino_local_repo_path = dino_local_repo_path
        self.dino_local_checkpoint_path = dino_local_checkpoint_path
        self.dino_allow_network_fallback = dino_allow_network_fallback

        self.num_tokens_h = image_size // patch_size
        self.num_tokens_w = image_size // patch_size
        self.num_tokens = self.num_tokens_h * self.num_tokens_w

        self.input_adapter = nn.Conv2d(5, 3, kernel_size=1, stride=1, padding=0)
        self.use_dino = False
        self.dino = None
        self.token_proj = None

        if encoder_name.startswith("dinov2"):
            self.use_dino = self._init_dino_encoder(encoder_name=encoder_name)

        if not self.use_dino:
            self.patch_embed = nn.Conv2d(
                5,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=4 * embed_dim,
                dropout=0.0,
                batch_first=True,
            )
            self.fallback_encoder = nn.TransformerEncoder(layer, num_layers=4)

    def _load_local_checkpoint_if_provided(self) -> bool:
        if not self.use_pretrained:
            return True
        ckpt_path = self.dino_local_checkpoint_path
        if not ckpt_path:
            warnings.warn(
                "DINOv2 pretrained requested in offline mode, but no local checkpoint path is set. "
                "Using uninitialized local DINOv2 weights."
            )
            return True
        if not os.path.isfile(ckpt_path):
            warnings.warn(
                f"DINOv2 local checkpoint path does not exist: {ckpt_path}. "
                "Using uninitialized local DINOv2 weights."
            )
            return True
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        msg = self.dino.load_state_dict(state_dict, strict=False)
        warnings.warn(f"Loaded local DINOv2 checkpoint from {ckpt_path}: {msg}")
        return True

    def _try_load_dino_from_local_repo(self, encoder_name: str) -> bool:
        repo_path = self.dino_local_repo_path
        if not repo_path:
            return False
        hubconf_path = os.path.join(repo_path, "hubconf.py")
        if not os.path.isfile(hubconf_path):
            warnings.warn(
                f"DINO local repo path provided but hubconf.py missing: {repo_path}"
            )
            return False
        try:
            self.dino = torch.hub.load(
                repo_or_dir=repo_path,
                model=encoder_name,
                source="local",
                pretrained=False,
            )
            return self._load_local_checkpoint_if_provided()
        except Exception as exc:
            warnings.warn(f"Failed loading DINOv2 from local repo {repo_path}: {exc}")
            return False

    def _try_load_dino_from_local_module(self, encoder_name: str) -> bool:
        try:
            from mapanything.models.external.dinov2.hub import backbones

            if not hasattr(backbones, encoder_name):
                return False
            builder = getattr(backbones, encoder_name)
            self.dino = builder(pretrained=False, img_size=self.image_size, patch_size=self.patch_size)
            return self._load_local_checkpoint_if_provided()
        except Exception as exc:
            warnings.warn(f"Failed loading DINOv2 from local module: {exc}")
            return False

    def _init_dino_encoder(self, encoder_name: str) -> bool:
        if self.dino_offline_mode:
            loaded = self._try_load_dino_from_local_repo(encoder_name=encoder_name)
            if not loaded:
                loaded = self._try_load_dino_from_local_module(encoder_name=encoder_name)
            if not loaded and not self.dino_allow_network_fallback:
                warnings.warn(
                    "Offline DINOv2 requested and no local source available; "
                    "falling back to lightweight image encoder without network access."
                )
                return False
            if loaded:
                dino_dim = getattr(self.dino, "embed_dim", 1024)
                self.token_proj = nn.Linear(dino_dim, self.embed_dim)
                self._freeze_first_dino_blocks()
                return True

        try:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2",
                encoder_name,
                pretrained=self.use_pretrained,
                force_reload=self.torch_hub_force_reload,
            )
            dino_dim = getattr(self.dino, "embed_dim", 1024)
            self.token_proj = nn.Linear(dino_dim, self.embed_dim)
            self._freeze_first_dino_blocks()
            return True
        except Exception as exc:
            warnings.warn(
                f"Failed to load {encoder_name} from torch hub ({exc}). "
                "Falling back to lightweight image encoder."
            )
            return False

    def _freeze_first_dino_blocks(self):
        if not self.use_dino:
            return
        blocks = getattr(self.dino, "blocks", None)
        if blocks is None:
            return
        n = min(self.freeze_first_n_blocks, len(blocks))
        for block_idx, block in enumerate(blocks):
            if block_idx < n:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dino:
            x_3 = self.input_adapter(x)
            feat = self.dino.forward_features(x_3)
            if isinstance(feat, dict):
                if "x_norm_patchtokens" in feat:
                    tokens = feat["x_norm_patchtokens"]
                elif "x_prenorm" in feat:
                    # Remove cls token when present.
                    tokens = feat["x_prenorm"][:, 1:, :]
                else:
                    raise RuntimeError("Unsupported DINOv2 forward_features output format.")
            else:
                tokens = feat
            if tokens.shape[1] != self.num_tokens:
                # Keep the expected token count for downstream decoders.
                tokens = tokens[:, : self.num_tokens, :]
            return self.token_proj(tokens)

        x = self.patch_embed(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, T, D)
        x = x + self.pos_embed
        x = self.fallback_encoder(x)
        return x


class _AstroSpectrumEncoder(nn.Module):
    """Spectrum encoder: Conv stem + Transformer."""

    def __init__(
        self,
        spec_length: int = 2048,
        stem_stride: int = 8,
        embed_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spec_length = spec_length
        self.stem_stride = stem_stride
        self.num_tokens = spec_length // stem_stride
        self.embed_dim = embed_dim

        self.stem = nn.Conv1d(
            in_channels=4,
            out_channels=embed_dim,
            kernel_size=stem_stride,
            stride=stem_stride,
            padding=0,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.stem(spec)  # (B, D, T)
        x = x.transpose(1, 2).contiguous()  # (B, T, D)
        x = x + self.pos_embed
        return self.encoder(x)


class AstroMapAnything(nn.Module):
    """
    Astronomy-focused dual-modal architecture with dynamic modality support.
    """

    def __init__(
        self,
        name: str = "astro_mapanything",
        image_size: int = 224,
        patch_size: int = 14,
        spec_length: int = 2048,
        spec_stride: int = 8,
        embed_dim: int = 768,
        fusion_num_heads: int = 12,
        fusion_num_layers: int = 8,
        spectrum_encoder_layers: int = 8,
        spectrum_decoder_layers: int = 6,
        dropout: float = 0.0,
        image_encoder_name: str = "dinov2_vitl14",
        image_encoder_pretrained: bool = True,
        freeze_image_encoder_first_n_blocks: int = 12,
        use_metadata: bool = True,
        metadata_dim: int = 4,
        pretrained_checkpoint_path: Optional[str] = None,
        load_specific_pretrained_submodules: bool = False,
        specific_pretrained_submodules: Optional[List[str]] = None,
        torch_hub_force_reload: bool = False,
        dino_offline_mode: bool = False,
        dino_local_repo_path: Optional[str] = None,
        dino_local_checkpoint_path: Optional[str] = None,
        dino_allow_network_fallback: bool = True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            warnings.warn(
                f"Ignoring unsupported AstroMapAnything kwargs: {sorted(kwargs.keys())}"
            )
        self.name = name
        self.image_size = image_size
        self.patch_size = patch_size
        self.spec_length = spec_length
        self.embed_dim = embed_dim
        self.use_metadata = use_metadata
        self.metadata_dim = metadata_dim
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules or []
        self.torch_hub_force_reload = torch_hub_force_reload

        self.image_encoder = _AstroImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_name=image_encoder_name,
            use_pretrained=image_encoder_pretrained,
            freeze_first_n_blocks=freeze_image_encoder_first_n_blocks,
            torch_hub_force_reload=torch_hub_force_reload,
            dino_offline_mode=dino_offline_mode,
            dino_local_repo_path=dino_local_repo_path,
            dino_local_checkpoint_path=dino_local_checkpoint_path,
            dino_allow_network_fallback=dino_allow_network_fallback,
        )
        self.spectrum_encoder = _AstroSpectrumEncoder(
            spec_length=spec_length,
            stem_stride=spec_stride,
            embed_dim=embed_dim,
            num_layers=spectrum_encoder_layers,
            num_heads=fusion_num_heads,
            dropout=dropout,
        )

        self.image_tokens = (image_size // patch_size) ** 2
        self.spec_tokens = spec_length // spec_stride
        if self.image_tokens != self.spec_tokens:
            raise ValueError(
                f"Expected equal token counts for image and spectrum paths, got "
                f"{self.image_tokens} and {self.spec_tokens}."
            )

        self.modality_type_tokens = nn.Parameter(torch.zeros(3, embed_dim))
        nn.init.trunc_normal_(self.modality_type_tokens, std=0.02)

        self.image_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spec_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.image_mask_token, std=0.02)
        nn.init.trunc_normal_(self.spec_mask_token, std=0.02)

        if self.use_metadata:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.meta_null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.meta_null_token, std=0.02)
        else:
            self.metadata_encoder = None
            self.meta_null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.meta_null_token, std=0.02)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=fusion_num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=fusion_num_layers)

        spec_dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=fusion_num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.spec_decoder = nn.TransformerDecoder(
            spec_dec_layer,
            num_layers=spectrum_decoder_layers,
        )
        self.spec_dec_pos = nn.Parameter(torch.zeros(1, self.spec_tokens, embed_dim))
        nn.init.trunc_normal_(self.spec_dec_pos, std=0.02)

        # Decode each spectrum token into 8 bins for each of the 4 channels.
        self.spec_out = nn.Linear(embed_dim, 4 * spec_stride)

        # Decode each image token into one patch.
        self.img_out = nn.Linear(embed_dim, 5 * patch_size * patch_size)

        self.logvar_img_head = nn.Linear(embed_dim, 1)
        self.logvar_spec_head = nn.Linear(embed_dim, 1)

        self._load_pretrained_weights_if_requested()

    @staticmethod
    def _broadcast_state_selector(
        tensor: torch.Tensor,
        state: torch.Tensor,
        target_state: int,
    ) -> torch.Tensor:
        view_shape = [state.shape[0]] + [1] * (tensor.ndim - 1)
        return (state == target_state).view(*view_shape).to(dtype=torch.bool, device=tensor.device)

    def _encode_metadata(self, meta_cond: torch.Tensor, meta_valid: Optional[torch.Tensor]) -> torch.Tensor:
        bsz = meta_cond.shape[0]
        if self.use_metadata:
            meta = self.metadata_encoder(meta_cond).unsqueeze(1)  # (B, 1, D)
            if meta_valid is not None:
                valid = meta_valid.view(bsz, 1, 1).to(dtype=torch.bool, device=meta.device)
                meta = torch.where(valid, meta, self.meta_null_token.expand_as(meta))
            return meta
        return self.meta_null_token.expand(bsz, 1, self.embed_dim)

    def _decode_image(self, image_tokens: torch.Tensor) -> torch.Tensor:
        bsz = image_tokens.shape[0]
        grid = self.image_size // self.patch_size
        patches = self.img_out(image_tokens)  # (B, T, 5*P*P)
        patches = patches.view(
            bsz,
            grid,
            grid,
            5,
            self.patch_size,
            self.patch_size,
        )
        # (B, grid_h, grid_w, C, ph, pw) -> (B, C, H, W)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(bsz, 5, self.image_size, self.image_size)

    def _decode_spectrum(self, spec_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        bsz = spec_tokens.shape[0]
        tgt = spec_tokens + self.spec_dec_pos
        dec = self.spec_decoder(tgt=tgt, memory=memory_tokens)  # (B, T, D)
        out = self.spec_out(dec)  # (B, T, 4*stride)
        stride = self.spec_length // self.spec_tokens
        out = out.view(bsz, self.spec_tokens, stride, 4)  # (B, 256, 8, 4)
        out = out.reshape(bsz, self.spec_length, 4).permute(0, 2, 1).contiguous()
        return out

    def _fuse_decode(
        self,
        image_tokens: torch.Tensor,
        spectrum_tokens: torch.Tensor,
        meta_token: torch.Tensor,
        image_available: torch.Tensor,
        spectrum_available: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz = image_tokens.shape[0]
        image_available = image_available.view(bsz, 1, 1).to(dtype=torch.bool, device=image_tokens.device)
        spectrum_available = spectrum_available.view(bsz, 1, 1).to(
            dtype=torch.bool, device=spectrum_tokens.device
        )

        image_in = torch.where(
            image_available,
            image_tokens,
            self.image_mask_token.expand(bsz, self.image_tokens, self.embed_dim),
        )
        spectrum_in = torch.where(
            spectrum_available,
            spectrum_tokens,
            self.spec_mask_token.expand(bsz, self.spec_tokens, self.embed_dim),
        )

        image_in = image_in + self.modality_type_tokens[0].view(1, 1, -1)
        spectrum_in = spectrum_in + self.modality_type_tokens[1].view(1, 1, -1)
        meta_in = meta_token + self.modality_type_tokens[2].view(1, 1, -1)

        fused = torch.cat([image_in, spectrum_in, meta_in], dim=1)
        fused = self.fusion(fused)

        image_latent = fused[:, : self.image_tokens, :]
        spectrum_latent = fused[:, self.image_tokens : self.image_tokens + self.spec_tokens, :]

        pred_image = self._decode_image(image_latent)
        pred_spectrum = self._decode_spectrum(spectrum_latent, fused)
        latent_image_cls = image_latent.mean(dim=1)
        latent_spectrum_cls = spectrum_latent.mean(dim=1)
        pred_logvar_image = self.logvar_img_head(latent_image_cls).squeeze(-1)
        pred_logvar_spectrum = self.logvar_spec_head(latent_spectrum_cls).squeeze(-1)

        return {
            "pred_image": pred_image,
            "pred_spectrum": pred_spectrum,
            "latent_image_cls": latent_image_cls,
            "latent_spectrum_cls": latent_spectrum_cls,
            "pred_logvar_image": pred_logvar_image,
            "pred_logvar_spectrum": pred_logvar_spectrum,
        }

    def _select_by_state(
        self,
        state: torch.Tensor,
        image_only_tensor: torch.Tensor,
        spectrum_only_tensor: torch.Tensor,
        both_tensor: torch.Tensor,
    ) -> torch.Tensor:
        is_image_only = self._broadcast_state_selector(image_only_tensor, state, target_state=0)
        is_spectrum_only = self._broadcast_state_selector(image_only_tensor, state, target_state=1)
        mixed = torch.where(is_image_only, image_only_tensor, both_tensor)
        mixed = torch.where(is_spectrum_only, spectrum_only_tensor, mixed)
        return mixed

    def _load_pretrained_weights_if_requested(self):
        if self.pretrained_checkpoint_path is None:
            return
        ckpt = torch.load(self.pretrained_checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if self.load_specific_pretrained_submodules and len(self.specific_pretrained_submodules) > 0:
            filtered = {}
            for key, value in state_dict.items():
                if any(key.startswith(prefix) for prefix in self.specific_pretrained_submodules):
                    filtered[key] = value
            state_dict = filtered
            msg = self.load_state_dict(state_dict, strict=False)
        else:
            msg = self.load_state_dict(state_dict, strict=False)
        warnings.warn(f"Loaded AstroMapAnything pretrained weights from {self.pretrained_checkpoint_path}: {msg}")

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        mode: str = "train",
    ) -> List[Dict[str, torch.Tensor]]:
        if len(views) != 2:
            raise ValueError(f"AstroMapAnything expects 2 views, got {len(views)}")

        view_image, view_spec = views
        image = view_image["img_astro"]  # (B, 5, 224, 224)
        spectrum = view_spec["spec_astro"]  # (B, 4, 2048)
        state = view_image["input_state"].long()  # 0=image-only, 1=spectrum-only, 2=both
        bsz = image.shape[0]

        image_tokens = self.image_encoder(image)
        spectrum_tokens = self.spectrum_encoder(spectrum)
        meta_token = self._encode_metadata(
            meta_cond=view_image["meta_cond"],
            meta_valid=view_image.get("meta_valid"),
        )

        ones = torch.ones((bsz,), dtype=torch.bool, device=image.device)
        zeros = torch.zeros((bsz,), dtype=torch.bool, device=image.device)

        out_image_only = self._fuse_decode(
            image_tokens=image_tokens,
            spectrum_tokens=spectrum_tokens,
            meta_token=meta_token,
            image_available=ones,
            spectrum_available=zeros,
        )
        out_spectrum_only = self._fuse_decode(
            image_tokens=image_tokens,
            spectrum_tokens=spectrum_tokens,
            meta_token=meta_token,
            image_available=zeros,
            spectrum_available=ones,
        )
        out_both = self._fuse_decode(
            image_tokens=image_tokens,
            spectrum_tokens=spectrum_tokens,
            meta_token=meta_token,
            image_available=ones,
            spectrum_available=ones,
        )

        if mode == "infer":
            image_available = view_image.get("image_input_mask", ones).to(dtype=torch.bool)
            spectrum_available = view_spec.get("spectrum_input_mask", ones).to(dtype=torch.bool)
            out_infer = self._fuse_decode(
                image_tokens=image_tokens,
                spectrum_tokens=spectrum_tokens,
                meta_token=meta_token,
                image_available=image_available,
                spectrum_available=spectrum_available,
            )
            pred_dict = dict(out_infer)
            pred_dict["input_state"] = state
            pred_dict["pred_image_from_spectrum"] = out_spectrum_only["pred_image"]
            pred_dict["pred_spectrum_from_image"] = out_image_only["pred_spectrum"]
            pred_dict["pred_image_from_both"] = out_both["pred_image"]
            pred_dict["pred_spectrum_from_both"] = out_both["pred_spectrum"]
            return [dict(pred_dict), dict(pred_dict)]

        pred_image_mixed = self._select_by_state(
            state=state,
            image_only_tensor=out_image_only["pred_image"],
            spectrum_only_tensor=out_spectrum_only["pred_image"],
            both_tensor=out_both["pred_image"],
        )
        pred_spectrum_mixed = self._select_by_state(
            state=state,
            image_only_tensor=out_image_only["pred_spectrum"],
            spectrum_only_tensor=out_spectrum_only["pred_spectrum"],
            both_tensor=out_both["pred_spectrum"],
        )

        pred_dict = {
            "pred_image": pred_image_mixed,
            "pred_spectrum": pred_spectrum_mixed,
            "pred_image_self": out_image_only["pred_image"],
            "pred_spectrum_self": out_spectrum_only["pred_spectrum"],
            "pred_image_from_spectrum": out_spectrum_only["pred_image"],
            "pred_spectrum_from_image": out_image_only["pred_spectrum"],
            "pred_image_from_both": out_both["pred_image"],
            "pred_spectrum_from_both": out_both["pred_spectrum"],
            "latent_image_cls": out_both["latent_image_cls"],
            "latent_spectrum_cls": out_both["latent_spectrum_cls"],
            "pred_logvar_image": out_both["pred_logvar_image"],
            "pred_logvar_spectrum": out_both["pred_logvar_spectrum"],
            "input_state": state,
        }
        return [dict(pred_dict), dict(pred_dict)]
