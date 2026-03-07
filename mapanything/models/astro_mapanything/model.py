"""
AstroMapAnything: dual-modal (image + spectrum) self-supervised model.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _AstroImageEncoder(nn.Module):
    """Image token encoder with native 5-channel support and optional registers."""

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
        dino_require_local_checkpoint: bool = True,
        dino_fail_on_missing_local_checkpoint: bool = True,
        dino_patch7_migration: bool = False,
        dino_patch7_init_from_vitl14: bool = True,
        dino_patch_kernel_interp_mode: str = "bilinear",
        dino_pos_embed_interp_mode: str = "bicubic",
        image_in_chans: int = 5,
        num_register_tokens: int = 4,
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
        self.dino_require_local_checkpoint = dino_require_local_checkpoint
        self.dino_fail_on_missing_local_checkpoint = dino_fail_on_missing_local_checkpoint
        self.dino_patch7_migration = dino_patch7_migration
        self.dino_patch7_init_from_vitl14 = dino_patch7_init_from_vitl14
        self.dino_patch_kernel_interp_mode = dino_patch_kernel_interp_mode
        self.dino_pos_embed_interp_mode = dino_pos_embed_interp_mode
        self.image_in_chans = image_in_chans
        self.num_register_tokens = num_register_tokens

        self.num_tokens_h = image_size // patch_size
        self.num_tokens_w = image_size // patch_size
        self.num_tokens = self.num_tokens_h * self.num_tokens_w

        self.use_dino = False
        self.dino = None
        self.token_proj = None

        if encoder_name.startswith("dinov2"):
            self.use_dino = self._init_dino_encoder(encoder_name=encoder_name)

        if not self.use_dino:
            self.patch_embed = nn.Conv2d(
                image_in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if self.num_register_tokens > 0:
                self.fallback_register_tokens = nn.Parameter(
                    torch.zeros(1, self.num_register_tokens, embed_dim)
                )
                nn.init.trunc_normal_(self.fallback_register_tokens, std=0.02)
            else:
                self.fallback_register_tokens = None
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=4 * embed_dim,
                dropout=0.0,
                batch_first=True,
            )
            self.fallback_encoder = nn.TransformerEncoder(layer, num_layers=4)

    def _handle_missing_local_checkpoint(self, reason: str) -> bool:
        msg = (
            f"DINOv2 local checkpoint is required but unavailable: {reason}. "
            f"dino_local_checkpoint_path={self.dino_local_checkpoint_path!r}"
        )
        if self.dino_require_local_checkpoint and self.dino_fail_on_missing_local_checkpoint:
            raise RuntimeError(msg)
        warnings.warn(msg if self.dino_require_local_checkpoint else reason)
        return not self.dino_require_local_checkpoint

    @staticmethod
    def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prefixes = ("module.", "backbone.", "student.", "teacher.")
        out = dict(state_dict)
        changed = True
        while changed and len(out) > 0:
            changed = False
            for prefix in prefixes:
                if all(k.startswith(prefix) for k in out.keys()):
                    out = {k[len(prefix) :]: v for k, v in out.items()}
                    changed = True
        return out

    @staticmethod
    def _extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
        def _as_tensor_dict(obj):
            if not isinstance(obj, dict):
                return None
            tensor_items = {k: v for k, v in obj.items() if torch.is_tensor(v)}
            if len(tensor_items) == 0:
                return None
            return tensor_items

        if isinstance(ckpt_obj, dict):
            for key in ("model", "state_dict", "teacher", "student"):
                if key in ckpt_obj:
                    nested = _as_tensor_dict(ckpt_obj[key])
                    if nested is not None:
                        return nested
        tensor_dict = _as_tensor_dict(ckpt_obj)
        if tensor_dict is not None:
            return tensor_dict
        raise TypeError("Unsupported checkpoint format for DINOv2 state dict extraction.")

    @staticmethod
    def _interpolate_patch_kernel(
        weight: torch.Tensor,
        target_h: int,
        target_w: int,
        mode: str,
    ) -> torch.Tensor:
        if weight.ndim != 4:
            raise ValueError(f"Expected 4D patch kernel, got shape {tuple(weight.shape)}")
        src_h, src_w = weight.shape[-2], weight.shape[-1]
        if (src_h, src_w) == (target_h, target_w):
            return weight
        kwargs = {}
        if mode in ("linear", "bilinear", "bicubic", "trilinear"):
            kwargs["align_corners"] = False
        return F.interpolate(weight.float(), size=(target_h, target_w), mode=mode, **kwargs).to(
            dtype=weight.dtype
        )

    @staticmethod
    def _interpolate_pos_embed(
        pos_embed: torch.Tensor,
        target_h: int,
        target_w: int,
        mode: str,
    ) -> torch.Tensor:
        if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
            raise ValueError(f"Expected pos_embed shape (1, N, D), got {tuple(pos_embed.shape)}")
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        src_tokens = patch_pos.shape[1]
        src_grid = int(math.sqrt(src_tokens))
        if src_grid * src_grid != src_tokens:
            raise ValueError(
                f"Cannot reshape source pos_embed tokens={src_tokens} into square grid."
            )
        dim = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2).float()
        kwargs = {}
        if mode in ("linear", "bilinear", "bicubic", "trilinear"):
            kwargs["align_corners"] = False
        patch_pos = F.interpolate(patch_pos, size=(target_h, target_w), mode=mode, **kwargs)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, dim)
        return torch.cat((cls_pos, patch_pos.to(dtype=cls_pos.dtype)), dim=1)

    @staticmethod
    def _adapt_patch_input_channels(
        weight: torch.Tensor,
        target_in_channels: int,
    ) -> torch.Tensor:
        src_in_channels = weight.shape[1]
        if src_in_channels == target_in_channels:
            return weight
        if src_in_channels > target_in_channels:
            return weight[:, :target_in_channels, :, :]
        extra_channels = target_in_channels - src_in_channels
        mean_channel = weight.mean(dim=1, keepdim=True)
        channel_std = weight.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        extra = mean_channel.expand(-1, extra_channels, -1, -1).clone()
        extra = extra + 0.01 * channel_std.expand_as(extra) * torch.randn_like(extra)
        return torch.cat([weight, extra], dim=1)

    @staticmethod
    def _adapt_register_tokens(register_tokens: torch.Tensor, target_num_tokens: int) -> torch.Tensor:
        if register_tokens.ndim != 3:
            raise ValueError(
                f"Expected register tokens shape (1, N, D), got {tuple(register_tokens.shape)}"
            )
        src_num_tokens = register_tokens.shape[1]
        if src_num_tokens == target_num_tokens:
            return register_tokens
        if src_num_tokens > target_num_tokens:
            return register_tokens[:, :target_num_tokens, :]
        if src_num_tokens == 0:
            return register_tokens.new_zeros((1, target_num_tokens, register_tokens.shape[-1]))
        pad = register_tokens[:, -1:, :].expand(-1, target_num_tokens - src_num_tokens, -1).clone()
        return torch.cat([register_tokens, pad], dim=1)

    def _load_local_checkpoint_if_provided(self) -> bool:
        if not self.use_pretrained:
            return True
        ckpt_path = self.dino_local_checkpoint_path
        if not ckpt_path:
            return self._handle_missing_local_checkpoint(
                "DINOv2 pretrained requested in offline mode, but no local checkpoint path is set"
            )
        if not os.path.isfile(ckpt_path):
            return self._handle_missing_local_checkpoint(
                f"DINOv2 local checkpoint path does not exist: {ckpt_path}"
            )
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            source_state = self._strip_known_prefixes(self._extract_state_dict(ckpt))
            target_state = self.dino.state_dict()

            migrated_state: Dict[str, torch.Tensor] = {}
            skipped = {}
            for key, value in source_state.items():
                if key not in target_state:
                    skipped[key] = (tuple(value.shape), None)
                    continue

                target_value = target_state[key]
                candidate = value
                if key == "patch_embed.proj.weight":
                    candidate = self._interpolate_patch_kernel(
                        candidate,
                        target_h=target_value.shape[-2],
                        target_w=target_value.shape[-1],
                        mode=self.dino_patch_kernel_interp_mode,
                    )
                    candidate = self._adapt_patch_input_channels(
                        candidate,
                        target_in_channels=target_value.shape[1],
                    )
                elif key == "pos_embed":
                    candidate = self._interpolate_pos_embed(
                        candidate,
                        target_h=self.num_tokens_h,
                        target_w=self.num_tokens_w,
                        mode=self.dino_pos_embed_interp_mode,
                    )
                elif key == "register_tokens":
                    candidate = self._adapt_register_tokens(
                        candidate,
                        target_num_tokens=target_value.shape[1],
                    )

                if tuple(candidate.shape) != tuple(target_value.shape):
                    skipped[key] = (tuple(value.shape), tuple(target_value.shape))
                    continue
                migrated_state[key] = candidate.to(dtype=target_value.dtype)

            msg = self.dino.load_state_dict(migrated_state, strict=False)
            critical_prefixes = ("patch_embed.proj", "blocks.", "norm", "pos_embed", "cls_token")
            critical_missing = [
                key for key in msg.missing_keys if any(key.startswith(prefix) for prefix in critical_prefixes)
            ]
            if critical_missing:
                raise RuntimeError(
                    "DINOv2 migration missing critical keys: "
                    + ", ".join(sorted(critical_missing)[:20])
                )
            print(
                f"[AstroMapAnything] Loaded local DINOv2 checkpoint from {ckpt_path}: {msg}, skipped={len(skipped)}"
            )
        except RuntimeError:
            raise
        except Exception as exc:
            return self._handle_missing_local_checkpoint(
                f"Failed to load local DINOv2 checkpoint at {ckpt_path}: {exc}"
            )
        return True

    def _try_init_patch7_dino_from_local_module(self) -> bool:
        try:
            from mapanything.models.external.dinov2.models import vision_transformer as vits

            self.dino = vits.vit_large(
                patch_size=self.patch_size,
                img_size=self.image_size,
                in_chans=self.image_in_chans,
                init_values=1.0,
                ffn_layer="mlp",
                block_chunks=0,
                num_register_tokens=self.num_register_tokens,
                interpolate_antialias=False,
                interpolate_offset=0.1,
            )
        except Exception as exc:
            warnings.warn(f"Failed to instantiate patch7 DINOv2 from local module: {exc}")
            return False

        if not self.use_pretrained:
            print("[AstroMapAnything] Initialized native patch7 DINOv2 image backbone.")
            return True

        loaded_ckpt = self._load_local_checkpoint_if_provided()
        if loaded_ckpt:
            print("[AstroMapAnything] Patch7 DINOv2 migration loaded with native input channels.")
        return loaded_ckpt

    def _try_load_dino_from_local_repo(self, encoder_name: str) -> bool:
        if self.image_in_chans != 3 or self.num_register_tokens != 0 or self.patch_size != 14:
            return False
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
            loaded_ckpt = self._load_local_checkpoint_if_provided()
            if loaded_ckpt:
                print(
                    f"[AstroMapAnything] Loaded DINOv2 backbone from local repo: {repo_path}"
                )
            return loaded_ckpt
        except RuntimeError:
            raise
        except Exception as exc:
            warnings.warn(f"Failed loading DINOv2 from local repo {repo_path}: {exc}")
            return False

    def _try_load_dino_from_local_module(self, encoder_name: str) -> bool:
        try:
            from mapanything.models.external.dinov2.hub import backbones

            if not hasattr(backbones, encoder_name):
                return False
            builder = getattr(backbones, encoder_name)
            self.dino = builder(
                pretrained=False,
                img_size=self.image_size,
                in_chans=self.image_in_chans,
                num_register_tokens=self.num_register_tokens,
            )
            loaded_ckpt = self._load_local_checkpoint_if_provided()
            if loaded_ckpt:
                print("[AstroMapAnything] Loaded DINOv2 backbone from bundled local module.")
            return loaded_ckpt
        except RuntimeError:
            raise
        except Exception as exc:
            warnings.warn(f"Failed loading DINOv2 from local module: {exc}")
            return False

    def _init_dino_encoder(self, encoder_name: str) -> bool:
        if (
            self.patch_size == 7
            and self.dino_patch7_migration
            and self.dino_patch7_init_from_vitl14
        ):
            loaded = self._try_init_patch7_dino_from_local_module()
            if loaded:
                dino_dim = getattr(self.dino, "embed_dim", 1024)
                self.token_proj = nn.Linear(dino_dim, self.embed_dim)
                self._freeze_first_dino_blocks()
                print(
                    "[AstroMapAnything] Using native patch7 DINOv2 with migrated weights."
                )
                return True
            if not self.dino_allow_network_fallback:
                warnings.warn(
                    "Patch7 DINOv2 migration failed and network fallback is disabled; "
                    "falling back to lightweight image encoder."
                )
                return False

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

        if self.image_in_chans != 3 or self.patch_size != 14 or self.num_register_tokens != 0:
            warnings.warn(
                "Torch hub DINOv2 fallback only supports canonical 3-channel patch14 models; "
                "falling back to lightweight image encoder."
            )
            return False

        try:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2",
                encoder_name,
                pretrained=self.use_pretrained,
                force_reload=self.torch_hub_force_reload,
            )
            print(
                f"[AstroMapAnything] Loaded DINOv2 backbone from network source: "
                f"facebookresearch/dinov2::{encoder_name}"
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

    def _reshape_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == self.num_tokens:
            return tokens
        src_tokens = tokens.shape[1]
        src_grid = int(math.sqrt(src_tokens))
        if src_grid * src_grid == src_tokens:
            tokens_dtype = tokens.dtype
            token_grid = tokens.reshape(tokens.shape[0], src_grid, src_grid, tokens.shape[-1]).permute(
                0,
                3,
                1,
                2,
            )
            token_grid = F.interpolate(
                token_grid.float(),
                size=(self.num_tokens_h, self.num_tokens_w),
                mode="bilinear",
                align_corners=False,
            )
            return token_grid.permute(0, 2, 3, 1).reshape(
                tokens.shape[0],
                self.num_tokens,
                tokens.shape[-1],
            ).to(dtype=tokens_dtype)
        tokens = tokens[:, : min(tokens.shape[1], self.num_tokens), :]
        if tokens.shape[1] < self.num_tokens:
            pad = self.num_tokens - tokens.shape[1]
            tokens = torch.cat(
                [tokens, tokens.new_zeros(tokens.shape[0], pad, tokens.shape[-1])],
                dim=1,
            )
        return tokens

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        if self.use_dino:
            feat = self.dino.forward_features(x)
            if isinstance(feat, dict):
                if "x_norm_patchtokens" in feat:
                    patch_tokens = feat["x_norm_patchtokens"]
                    register_tokens = feat.get("x_norm_regtokens")
                elif "x_prenorm" in feat:
                    patch_tokens = feat["x_prenorm"][:, 1 + self.num_register_tokens :, :]
                    register_tokens = None
                    if self.num_register_tokens > 0:
                        register_tokens = feat["x_prenorm"][:, 1 : 1 + self.num_register_tokens, :]
                else:
                    raise RuntimeError("Unsupported DINOv2 forward_features output format.")
            else:
                patch_tokens = feat
                register_tokens = None
            patch_tokens = self._reshape_patch_tokens(patch_tokens)
            patch_tokens = self.token_proj(patch_tokens)
            if register_tokens is not None and register_tokens.numel() > 0:
                register_tokens = self.token_proj(register_tokens)
            else:
                register_tokens = None
            return {
                "patch_tokens": patch_tokens,
                "register_tokens": register_tokens,
            }

        patch_tokens = self.patch_embed(x)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        patch_tokens = patch_tokens + self.pos_embed
        patch_tokens = self.fallback_encoder(patch_tokens)
        register_tokens = None
        if self.fallback_register_tokens is not None:
            register_tokens = self.fallback_register_tokens.expand(x.shape[0], -1, -1)
        return {
            "patch_tokens": patch_tokens,
            "register_tokens": register_tokens,
        }


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
    """Astronomy-focused dual-modal architecture with a shared backbone API."""

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
        dino_require_local_checkpoint: bool = True,
        dino_fail_on_missing_local_checkpoint: bool = True,
        dino_patch7_migration: bool = False,
        dino_patch7_init_from_vitl14: bool = True,
        dino_patch_kernel_interp_mode: str = "bilinear",
        dino_pos_embed_interp_mode: str = "bicubic",
        enable_fusion_gradient_checkpointing: bool = False,
        contrast_proj_dim: int = 256,
        image_in_chans: int = 5,
        num_register_tokens: int = 4,
        enable_nuisance_head: bool = True,
        nuisance_output_dim: int = 11,
        shared_pooling_type: str = "attention",
        register_usage: str = "nuisance_only",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            warnings.warn(
                f"Ignoring unsupported AstroMapAnything kwargs: {sorted(kwargs.keys())}"
            )
        if shared_pooling_type != "attention":
            raise ValueError(
                f"Unsupported shared_pooling_type={shared_pooling_type}. Only 'attention' is supported."
            )
        if register_usage != "nuisance_only":
            raise ValueError(
                f"Unsupported register_usage={register_usage}. Only 'nuisance_only' is supported."
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
        self.enable_fusion_gradient_checkpointing = enable_fusion_gradient_checkpointing
        self.image_in_chans = image_in_chans
        self.num_register_tokens = num_register_tokens
        self.enable_nuisance_head = enable_nuisance_head
        self.nuisance_output_dim = nuisance_output_dim

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
            dino_require_local_checkpoint=dino_require_local_checkpoint,
            dino_fail_on_missing_local_checkpoint=dino_fail_on_missing_local_checkpoint,
            dino_patch7_migration=dino_patch7_migration,
            dino_patch7_init_from_vitl14=dino_patch7_init_from_vitl14,
            dino_patch_kernel_interp_mode=dino_patch_kernel_interp_mode,
            dino_pos_embed_interp_mode=dino_pos_embed_interp_mode,
            image_in_chans=image_in_chans,
            num_register_tokens=num_register_tokens,
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

        self.spec_out = nn.Linear(embed_dim, 4 * spec_stride)
        self.img_out = nn.Linear(embed_dim, image_in_chans * patch_size * patch_size)

        self.logvar_img_head = nn.Linear(embed_dim, 1)
        self.logvar_spec_head = nn.Linear(embed_dim, 1)
        self.image_proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, contrast_proj_dim),
        )
        self.spec_proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, contrast_proj_dim),
        )
        self.shared_proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, contrast_proj_dim),
        )

        self.image_pool_query = nn.Parameter(torch.zeros(1, embed_dim))
        self.spec_pool_query = nn.Parameter(torch.zeros(1, embed_dim))
        self.shared_pool_query = nn.Parameter(torch.zeros(1, embed_dim))
        self.nuis_pool_query = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.image_pool_query, std=0.02)
        nn.init.trunc_normal_(self.spec_pool_query, std=0.02)
        nn.init.trunc_normal_(self.shared_pool_query, std=0.02)
        nn.init.trunc_normal_(self.nuis_pool_query, std=0.02)

        if enable_nuisance_head:
            self.nuisance_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, nuisance_output_dim),
            )
        else:
            self.nuisance_head = None

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
            meta = self.metadata_encoder(meta_cond).unsqueeze(1)
            if meta_valid is not None:
                valid = meta_valid.view(bsz, 1, 1).to(dtype=torch.bool, device=meta.device)
                meta = torch.where(valid, meta, self.meta_null_token.expand_as(meta))
            return meta
        return self.meta_null_token.expand(bsz, 1, self.embed_dim)

    def _decode_image(self, image_tokens: torch.Tensor) -> torch.Tensor:
        bsz = image_tokens.shape[0]
        grid = self.image_size // self.patch_size
        patches = self.img_out(image_tokens)
        patches = patches.view(
            bsz,
            grid,
            grid,
            self.image_in_chans,
            self.patch_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(bsz, self.image_in_chans, self.image_size, self.image_size)

    def _decode_spectrum(self, spec_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        bsz = spec_tokens.shape[0]
        tgt = spec_tokens + self.spec_dec_pos
        dec = self.spec_decoder(tgt=tgt, memory=memory_tokens)
        out = self.spec_out(dec)
        stride = self.spec_length // self.spec_tokens
        out = out.view(bsz, self.spec_tokens, stride, 4)
        out = out.reshape(bsz, self.spec_length, 4).permute(0, 2, 1).contiguous()
        return out

    def _attn_pool(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if tokens is None or tokens.numel() == 0:
            batch_size = 0 if tokens is None else tokens.shape[0]
            device = self.modality_type_tokens.device
            dtype = self.modality_type_tokens.dtype
            return torch.zeros((batch_size, self.embed_dim), device=device, dtype=dtype)
        scores = (tokens * query.view(1, 1, -1)).sum(dim=-1) / math.sqrt(tokens.shape[-1])
        weights = F.softmax(scores, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)

    def _fuse_decode(
        self,
        image_tokens: torch.Tensor,
        spectrum_tokens: torch.Tensor,
        meta_token: torch.Tensor,
        image_available: torch.Tensor,
        spectrum_available: torch.Tensor,
        image_register_tokens: Optional[torch.Tensor] = None,
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
        if self.enable_fusion_gradient_checkpointing and self.training:
            fused = checkpoint(self.fusion, fused, use_reentrant=False)
        else:
            fused = self.fusion(fused)

        image_latent = fused[:, : self.image_tokens, :]
        spectrum_latent = fused[:, self.image_tokens : self.image_tokens + self.spec_tokens, :]
        meta_latent = fused[:, self.image_tokens + self.spec_tokens :, :]

        pred_image = self._decode_image(image_latent)
        pred_spectrum = self._decode_spectrum(spectrum_latent, fused)

        z_img = self._attn_pool(image_tokens, self.image_pool_query)
        z_spec = self._attn_pool(spectrum_tokens, self.spec_pool_query)
        z_shared = self._attn_pool(
            torch.cat([image_latent, spectrum_latent, meta_latent], dim=1),
            self.shared_pool_query,
        )
        if image_register_tokens is None or image_register_tokens.numel() == 0:
            z_nuis = z_shared.new_zeros((bsz, self.embed_dim))
        else:
            z_nuis = self._attn_pool(image_register_tokens, self.nuis_pool_query)

        latent_image_proj = self.image_proj_head(z_img)
        latent_spectrum_proj = self.spec_proj_head(z_spec)
        latent_shared_proj = self.shared_proj_head(z_shared)
        pred_logvar_image = self.logvar_img_head(z_img).squeeze(-1)
        pred_logvar_spectrum = self.logvar_spec_head(z_spec).squeeze(-1)
        if self.nuisance_head is not None:
            pred_nuisance = self.nuisance_head(z_nuis)
        else:
            pred_nuisance = z_nuis.new_zeros((bsz, self.nuisance_output_dim))

        return {
            "pred_image": pred_image,
            "pred_spectrum": pred_spectrum,
            "pred_nuisance": pred_nuisance,
            "z_img": z_img,
            "z_spec": z_spec,
            "z_shared": z_shared,
            "z_nuis": z_nuis,
            "latent_image_cls": z_img,
            "latent_spectrum_cls": z_spec,
            "latent_shared_cls": z_shared,
            "latent_image_proj": latent_image_proj,
            "latent_spectrum_proj": latent_spectrum_proj,
            "latent_shared_proj": latent_shared_proj,
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
        warnings.warn(
            f"Loaded AstroMapAnything pretrained weights from {self.pretrained_checkpoint_path}: {msg}"
        )

    def encode(
        self,
        views: List[Dict[str, torch.Tensor]],
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if len(views) != 2:
            raise ValueError(f"AstroMapAnything expects 2 views, got {len(views)}")

        view_image, view_spec = views
        image = view_image["img_astro"]
        spectrum = view_spec["spec_astro"]
        bsz = image.shape[0]
        ones = torch.ones((bsz,), dtype=torch.bool, device=image.device)

        image_encoded = self.image_encoder(image)
        image_tokens = image_encoded["patch_tokens"]
        image_register_tokens = image_encoded.get("register_tokens")
        spectrum_tokens = self.spectrum_encoder(spectrum)
        meta_token = self._encode_metadata(
            meta_cond=view_image["meta_cond"],
            meta_valid=view_image.get("meta_valid"),
        )
        out = self._fuse_decode(
            image_tokens=image_tokens,
            spectrum_tokens=spectrum_tokens,
            meta_token=meta_token,
            image_available=view_image.get("image_input_mask", ones).to(dtype=torch.bool),
            spectrum_available=view_spec.get("spectrum_input_mask", ones).to(dtype=torch.bool),
            image_register_tokens=image_register_tokens,
        )
        features = {
            key: out[key]
            for key in (
                "z_img",
                "z_spec",
                "z_shared",
                "z_nuis",
                "latent_image_proj",
                "latent_spectrum_proj",
                "latent_shared_proj",
                "pred_nuisance",
            )
        }
        if return_tokens:
            features["image_patch_tokens"] = image_tokens
            features["image_register_tokens"] = image_register_tokens
            features["spectrum_tokens"] = spectrum_tokens
        return features

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        mode: str = "train",
    ) -> List[Dict[str, torch.Tensor]]:
        if len(views) != 2:
            raise ValueError(f"AstroMapAnything expects 2 views, got {len(views)}")

        view_image, view_spec = views
        image = view_image["img_astro"]
        spectrum = view_spec["spec_astro"]
        state = view_image["input_state"].long()
        bsz = image.shape[0]

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
            image_tokens=image_tokens,
            spectrum_tokens=spectrum_tokens,
            meta_token=meta_token,
            image_available=ones,
            spectrum_available=zeros,
            image_register_tokens=image_register_tokens,
        )
        out_spectrum_only = self._fuse_decode(
            image_tokens=image_tokens,
            spectrum_tokens=spectrum_tokens,
            meta_token=meta_token,
            image_available=zeros,
            spectrum_available=ones,
            image_register_tokens=image_register_tokens,
        )
        out_both = self._fuse_decode(
            image_tokens=image_tokens,
            spectrum_tokens=spectrum_tokens,
            meta_token=meta_token,
            image_available=ones,
            spectrum_available=ones,
            image_register_tokens=image_register_tokens,
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
            "pred_nuisance": out_both["pred_nuisance"],
            "z_img": out_both["z_img"],
            "z_spec": out_both["z_spec"],
            "z_shared": out_both["z_shared"],
            "z_nuis": out_both["z_nuis"],
            "z_shared_from_image": out_image_only["z_shared"],
            "z_shared_from_spectrum": out_spectrum_only["z_shared"],
            "z_shared_from_both": out_both["z_shared"],
            "latent_image_cls": out_both["latent_image_cls"],
            "latent_spectrum_cls": out_both["latent_spectrum_cls"],
            "latent_shared_cls": out_both["latent_shared_cls"],
            "latent_image_proj": out_both["latent_image_proj"],
            "latent_spectrum_proj": out_both["latent_spectrum_proj"],
            "latent_shared_proj": out_both["latent_shared_proj"],
            "pred_logvar_image": out_both["pred_logvar_image"],
            "pred_logvar_spectrum": out_both["pred_logvar_spectrum"],
            "input_state": state,
        }
        return [dict(pred_dict), dict(pred_dict)]
