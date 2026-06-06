from pathlib import Path
from typing import Any, List, Optional
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, SpatialAttentionBlock, Upsample 

from monai.networks.nets.autoencoderkl import (
    AutoencoderKL,
    AEKLDownsample,
    AEKLResBlock,
)

from brainmint.models.blocks.haar_dwt import (
    TwoStageInverseWaveletTransform,
    TwoStageWaveletTransform,
)
from brainmint.utils.state_dict_loader import load_module_state_dict


class WaveletEncoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int, # 1
        channels: Sequence[int],   # 256
        out_channels: int,  
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels
        self.wavelet_transform = TwoStageWaveletTransform(self.in_channels)
        self.wavelet_out_channels = self.wavelet_transform.out_channels

        blocks: List[nn.Module] = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.wavelet_out_channels,
                out_channels=channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels) - 1

            for _ in range(self.num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(AEKLDownsample(spatial_dims=spatial_dims, in_channels=input_channel))
        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=channels[-1],
                )
            )

            blocks.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=channels[-1],
                )
            )
        # Normalise and convert to latent size
        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels[-1], eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wavelet_transform(x)
        for block in self.blocks:
            x = block(x)
        return x

class WaveletDecoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: Sequence[int],   # 256
        in_channels: int, # 4
        out_channels: int, # 1
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels
        self.inv_wavelet_transform = TwoStageInverseWaveletTransform(self.out_channels)
        self.stage_2_out_channels = self.inv_wavelet_transform.in_channels
        

        reversed_block_out_channels = list(reversed(channels))

        blocks: List[nn.Module] = []

        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )
            blocks.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                if use_convtranspose:
                    blocks.append(
                        Upsample(
                            spatial_dims=spatial_dims, mode="deconv", in_channels=block_in_ch, out_channels=block_in_ch
                        )
                    )
                else:
                    post_conv = Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        out_channels=block_in_ch,
                        strides=1,
                        kernel_size=3,
                        padding=1,
                        conv_only=True,
                    )
                    blocks.append(
                        Upsample(
                            spatial_dims=spatial_dims,
                            mode="nontrainable",
                            in_channels=block_in_ch,
                            out_channels=block_in_ch,
                            interp_mode="nearest",
                            scale_factor=2.0,
                            post_conv=post_conv,
                            align_corners=None,
                        )
                    )

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=self.stage_2_out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.inv_wavelet_transform(x)
        return x


class WaveletAutoencoder(AutoencoderKL):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        channels: number of output channels for each block.
        attention_levels: sequence of levels to add attention.
        latent_channels: latent embedding dimension.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        with_encoder_nonlocal_attn: if True use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: if True use non-local attention block in the decoder.
        use_checkpoint: if True, use activation checkpoint to save memory.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
        include_fc: whether to include the final linear layer in the attention block. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection in the attention block, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2),
        channels: Sequence[int] = (128),
        attention_levels: Sequence[bool] = (True, True),
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_checkpoint: bool = False,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            channels=channels,
            attention_levels=attention_levels,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
            use_checkpoint=use_checkpoint,
            use_convtranspose=use_convtranspose,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in channels):
            raise ValueError("AutoencoderKL expects all channels being multiple of norm_num_groups")

        if len(channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects channels being same size of attention_levels")

        if len(num_res_blocks) != len(channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`channels`."
            )

        self.encoder = WaveletEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.decoder = WaveletDecoder(
            spatial_dims=spatial_dims,
            channels=channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_convtranspose=use_convtranspose,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )


class WaveletVAE(nn.Module):
    """Checkpointed Wavelet VAE exposed as a BrainMint compression model."""

    def __init__(
        self,
        *,
        autoencoder: nn.Module,
        ckpt_path: str | Path,
        state_key: Optional[str] = "autoencoder",
        loader: Optional[str] = None,
        strict: bool | str = True,
        freeze: bool = True,
        set_eval: bool = True,
    ) -> None:
        super().__init__()
        self.model = autoencoder
        load_module_state_dict(
            self.model,
            path=str(ckpt_path),
            state_key=state_key,
            loader=loader,
            strict=strict,
            freeze=freeze,
            set_eval=set_eval,
            target_name="wavelet_vae",
        )
        if set_eval:
            self.eval()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        reconstruction, _, _ = self.model(x, *args, **kwargs)
        return reconstruction

    def reconstruct(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(x, *args, **kwargs)

    def run_inference(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction, z_mu, z_sigma = self.model(batch["image"])
        return reconstruction, z_mu, z_sigma
