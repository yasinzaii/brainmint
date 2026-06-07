# =========================================================================
# Adapted from https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/generation/maisi/networks/diffusion_model_unet_maisi.py
# which has the following license:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from monai.networks.blocks import Convolution
from monai.networks.nets.diffusion_model_unet import (
    get_down_block,
    get_mid_block,
    get_timestep_embedding,
    get_up_block,
    zero_module,
)
from monai.utils import ensure_tuple_rep
from monai.utils.type_conversion import convert_to_tensor
from torch import nn


class DiffusionUNet(nn.Module):
    """
    3D diffusion U-Net with timestep, optional class conditioning, and optional
    global (non-spatial) conditioning vector (e.g. demographics).

    This is adapted from:
      - MONAI's DiffusionModelUNet
      - MAISI's latent diffusion U-Net

    Changes from the original MAISI U-Net:
      * Adds an optional `demographics_embedding` input (global vector).
      * Concatenates that vector to the time+class embedding before UNet blocks.
      * Drops unused region/spacing inputs (top/bottom index, spacing).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
        with_demographics: bool = False,
        dem_embed_dim: int | None = None,
    ) -> None:
        super().__init__()

        if with_conditioning and cross_attention_dim is None:
            raise ValueError(
                "DiffusionUNet expects `cross_attention_dim` when `with_conditioning=True`."
            )
        if cross_attention_dim is not None and not with_conditioning:
            raise ValueError(
                "DiffusionUNet expects `with_conditioning=True` when specifying `cross_attention_dim`."
            )
        if not (0.0 <= dropout_cattn <= 1.0):
            raise ValueError("Dropout must be in [0, 1].")

        if any((c % norm_num_groups) != 0 for c in num_channels):
            raise ValueError(
                "All `num_channels` must be multiples of `norm_num_groups`, "
                f"got num_channels={num_channels}, norm_num_groups={norm_num_groups}."
            )

        if len(num_channels) != len(attention_levels):
            raise ValueError(
                "`num_channels` and `attention_levels` must have the same length, "
                f"got {len(num_channels)} vs {len(attention_levels)}."
            )

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` must be an int or a sequence with the same length as `num_channels`."
            )

        if use_flash_attention and not torch.cuda.is_available():
            raise ValueError(
                "Flash attention is only available on GPU (`torch.cuda.is_available()` must be True)."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # Input projection
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # Time embedding
        time_embed_dim = num_channels[0] * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = self._create_embedding_module(num_channels[0], time_embed_dim)

        # Class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # Optional demographics embedding
        self.with_demographics = with_demographics
        if with_demographics:
            if dem_embed_dim is None:
                dem_embed_dim = time_embed_dim
            self.dem_embed_dim = int(dem_embed_dim)

            # If different, project to time_embed_dim so we can concat cleanly
            if self.dem_embed_dim != time_embed_dim:
                self.dem_proj = nn.Linear(self.dem_embed_dim, time_embed_dim)
            else:
                self.dem_proj = nn.Identity()
        else:
            self.dem_embed_dim = None
            self.dem_proj = None

        # Final embedding size seen by the UNet blocks
        new_time_embed_dim = time_embed_dim
        if self.with_demographics:
            new_time_embed_dim += time_embed_dim

        # Down blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )
            self.down_blocks.append(down_block)

        # Middle block
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            temb_channels=new_time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))

        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]
            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )
            self.up_blocks.append(up_block)

        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    # Helpers
    @staticmethod
    def _create_embedding_module(input_dim: int, embed_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _get_time_and_class_embedding(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None,
    ) -> torch.Tensor:
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels must be provided when num_class_embeds > 0.")
            class_emb = self.class_embedding(class_labels).to(dtype=x.dtype)
            emb = emb + class_emb

        return emb

    def _add_demographics_embedding(
        self,
        emb: torch.Tensor,
        demographics_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.with_demographics:
            return emb

        if demographics_embedding is None:
            raise ValueError(
                "DiffusionUNet was created with with_demographics=True "
                "but no `demographics_embedding` was provided to forward()."
            )

        if demographics_embedding.ndim != 2:
            raise ValueError(
                "Expected `demographics_embedding` to have shape (B, D), "
                f"got {tuple(demographics_embedding.shape)}."
            )
        if demographics_embedding.shape[0] != emb.shape[0]:
            raise ValueError(
                "Batch size mismatch between `emb` and `demographics_embedding`: "
                f"{emb.shape[0]} vs {demographics_embedding.shape[0]}."
            )

        dem = self.dem_proj(demographics_embedding.to(emb.dtype))
        return torch.cat((emb, dem), dim=1)

    def _apply_down_blocks(
        self,
        h: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None,
        down_block_additional_residuals: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if context is not None and not self.with_conditioning:
            raise ValueError(
                "Model must be created with with_conditioning=True if `context` is provided."
            )

        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            down_block_res_samples.extend(res_samples)

        # Optional additional residuals (e.g. ControlNet)
        if down_block_additional_residuals is not None:
            if len(down_block_additional_residuals) != len(down_block_res_samples):
                raise ValueError(
                    "Mismatch between down_block_additional_residuals and down_block_res_samples "
                    f"({len(down_block_additional_residuals)} vs {len(down_block_res_samples)})."
                )
            new_res_samples: list[torch.Tensor] = []
            for base, add in zip(down_block_res_samples, down_block_additional_residuals, strict=True):
                new_res_samples.append(base + add)
            down_block_res_samples = new_res_samples

        return h, down_block_res_samples

    def _apply_up_blocks(
        self,
        h: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None,
        down_block_res_samples: list[torch.Tensor],
    ) -> torch.Tensor:
        for upsample_block in self.up_blocks:
            idx: int = -len(upsample_block.resnets)  # type: ignore
            res_samples = down_block_res_samples[idx:]
            down_block_res_samples = down_block_res_samples[:idx]
            h = upsample_block(
                hidden_states=h,
                res_hidden_states_list=res_samples,
                temb=emb,
                context=context,
            )
        return h

    # Forward
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor, ...] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        demographics_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, C, *spatial_dims)
            timesteps: diffusion timesteps (B,)
            context: optional cross-attention context (B, 1, C_ctx) if with_conditioning=True
            class_labels: Class labels of shape (B,).
            down_block_additional_residuals: Optional ControlNet-style residuals for down blocks of shape (B, C, FeatureMapsDims).
            mid_block_additional_residual: Optional ControlNet-style residual for mid block (B, C, FeatureMapsDims).
            demographics_embedding: Optional global embedding (B, D_dem) if with_demographics=True

        Returns:
            Tensor of shape (B, out_channels, *spatial_dims)
        """
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._add_demographics_embedding(emb, demographics_embedding)

        h = self.conv_in(x)
        h, _updated_down_block_res_samples = self._apply_down_blocks(
            h, emb, context, down_block_additional_residuals
        )

        h = self.middle_block(h, emb, context)

        # Additional residual connections for ControlNets
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual

        h = self._apply_up_blocks(h, emb, context, _updated_down_block_res_samples)
        h = self.out(h)
        return convert_to_tensor(h)
