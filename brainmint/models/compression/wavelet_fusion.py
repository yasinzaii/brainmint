from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, SpatialAttentionBlock, Upsample
from monai.networks.nets.autoencoderkl import (
    AEKLDownsample,
    AEKLResBlock,
    AutoencoderKL,
)
from monai.utils import ensure_tuple_rep

from brainmint.models.blocks.haar_wavelet_fusion import (
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform3D,
)
from brainmint.utils.state_dict_loader import load_module_state_dict


class AEKLResBlockDecoder(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, norm_num_groups: int, norm_eps: float, out_channels: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.nin_shortcut: nn.Module
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)

        return x + h

class TwoStageWaveletFusion(nn.Module):

    def __init__(self, in_channels: int, out_channels:int, spatial_dims:int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stage1 = HaarWaveletTransform3D()
        self.stage2 = HaarWaveletTransform3D()
        self.stage1_out_channels = in_channels * 8
        self.stage2_out_channels = in_channels * 4 * 8 

        self.stage1_conv_out = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.stage1_out_channels,
            out_channels=out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.stage2_conv_out = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.stage2_out_channels,
            out_channels=out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )


    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        c = self.in_channels                          # [B, 8C, D/2, H/2, W/2]
        stage1_coeffs = self.stage1(x)
        stage1_first_four = stage1_coeffs[:, :4 * c]  # [B, 8*(4C), D/4, H/4, W/4]
        
        stage2_coeffs = self.stage2(stage1_first_four)
        #stage2_first_four = stage2_coeffs[:, :4]

        out_stage1 = self.stage1_conv_out(stage1_coeffs)  # in_channels = 8C
        out_stage2 = self.stage2_conv_out(stage2_coeffs)  # in_channels = 8*(4C) = 32C
        
        return out_stage1, out_stage2


class TwoStageInverseWaveletFusion(nn.Module):

    def __init__(self, out_channels: int, in_channels: int, spatial_dims:int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stage1_inverse = InverseHaarWaveletTransform3D()
        self.stage2_inverse = InverseHaarWaveletTransform3D()
        self.stage1_in_channels = out_channels * 8
        self.stage2_in_channels = out_channels * 4 * 8

        self.stage1_conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.stage1_in_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.stage2_conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.stage2_in_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

    def forward(
        self, stage1_in: torch.Tensor, stage2_in: torch.Tensor
    ) -> torch.Tensor:
        
        c = self.out_channels

        # Expect stage2_in at D/4 so that inverse() upsamples to D/2
        stage2_coeffs = self.stage2_conv_in(stage2_in)  # [B, 8*(4C), D/4, H/4, W/4]
        stage1_first_four = self.stage2_inverse(stage2_coeffs)  # [B, 4C, D/2, H/2, W/2]

        stage1_coeffs = self.stage1_conv_in(stage1_in)  # [B, 8C, D/2, H/2, W/2]
        stage1_coeffs[:, :4 * c] = stage1_coeffs[:, :4 * c]  + stage1_first_four

        reconstructed = self.stage1_inverse(stage1_coeffs)  # [B, C, D, H, W]
        
        return reconstructed 

class WaveletFusionEncoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        wavelet_out_channels: number of output channels for  TwoStageWaveletFusion.
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
        in_channels: int,
        wavelet_channels: int,
        channels: Sequence[int],
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
        
        self.wavelet_out_channels = wavelet_channels
        self.wavelet_transform = TwoStageWaveletFusion(
            in_channels=in_channels, 
            out_channels=self.wavelet_out_channels, 
            spatial_dims=spatial_dims
        ) 
        

        self.stages = nn.ModuleList()
        blocks: list[nn.Module] = []

        # Residual and downsampling blocks
        output_channel = channels[0]
        for i in range(len(channels)):
            
            input_channel = output_channel
            if i == 0:
                input_channel = self.wavelet_out_channels 
            elif i == 1:
                input_channel = input_channel + self.wavelet_out_channels 
            
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

                #self.stages.append(nn.ModuleList(blocks))
                self.stages.append(nn.Sequential(*blocks))
                blocks = [] # Restart

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
        self.stages.append(nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        stage1, stage2 = self.wavelet_transform(x)
        
        x = stage1  
        for block in self.stages[0]:
            x = block(x)
        x = torch.cat([stage2, x], dim=1)

        for idx in range(1,len(self.stages)):
            for block in self.stages[idx]:
                x = block(x)
        return x


class WaveletFusionDecoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        wavelet_in_channels: number of out input channels for TwoStageInverseWaveletFusion
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
        channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        wavelet_channels:int, 
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

        self.wavelet_in_channels = wavelet_channels
        self.inv_wavelet_transform = TwoStageInverseWaveletFusion(
            out_channels=out_channels, 
            in_channels=self.wavelet_in_channels, 
            spatial_dims=spatial_dims)

        reversed_block_out_channels = list(reversed(channels))

        self.stages = nn.ModuleList()
        
        blocks: list[nn.Module] = []

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
                AEKLResBlockDecoder(
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
                AEKLResBlockDecoder(
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
        block_out_ch_no_wavelet = block_out_ch
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = reversed_block_out_channels[i]
            block_out_ch = reversed_block_out_channels[i]

           
            is_final_block = i == len(channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    AEKLResBlockDecoder(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                
                if i == len(reversed_block_out_channels)-2:
                    block_out_ch = reversed_block_out_channels[i+1] + self.wavelet_in_channels 
                    block_out_ch_no_wavelet = reversed_block_out_channels[i+1]
                    
                elif i == len(reversed_block_out_channels)-1:
                    block_out_ch = self.wavelet_in_channels
                    block_out_ch_no_wavelet = block_out_ch
                else:
                    block_out_ch = reversed_block_out_channels[i+1]
                    block_out_ch_no_wavelet = block_out_ch
                
                
                if reversed_attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_out_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                
                self.stages.append(nn.Sequential(*blocks))
                blocks = [] # Restart
                
                if use_convtranspose:
                    blocks.append(
                        Upsample(
                            spatial_dims=spatial_dims, 
                            mode="deconv", 
                            in_channels=block_out_ch_no_wavelet, 
                            out_channels=block_out_ch_no_wavelet,
                        )
                    )
                else:
                    post_conv = Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=block_out_ch_no_wavelet,
                        out_channels=block_out_ch_no_wavelet,
                        strides=1,
                        kernel_size=3,
                        padding=1,
                        conv_only=True,
                    )
                    blocks.append(
                        Upsample(
                            spatial_dims=spatial_dims,
                            mode="nontrainable",
                            in_channels=block_out_ch_no_wavelet,
                            out_channels=block_out_ch_no_wavelet,
                            interp_mode="nearest",
                            scale_factor=2.0,
                            post_conv=post_conv,
                            align_corners=None,
                        )
                    )

                

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_out_ch, eps=norm_eps, affine=True))

        self.stages.append(nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for idx in range(len(self.stages) - 1):
            for block in self.stages[idx]:
                x = block(x)

        stage2_input = x[:, :self.wavelet_in_channels]
        x = x[:, self.wavelet_in_channels:]
        for block in self.stages[-1]:
            x = block(x)
        stage1_input = x
        x = self.inv_wavelet_transform(stage1_in=stage1_input, stage2_in=stage2_input)
        return x

class WaveletFusionAutoencoder(AutoencoderKL):
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
        wavelet_channels: int = 64,
        num_res_blocks: Sequence[int] | int = (2, 2),
        channels: Sequence[int] = (128,256),
        attention_levels: Sequence[bool] = (False, False, False),
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = False,
        with_decoder_nonlocal_attn: bool = False,
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
        )

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in channels):
            raise ValueError("AutoencoderKL expects all channels being multiple of norm_num_groups")

        if len(channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects channels being same size of attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(channels))

        if len(num_res_blocks) != len(channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`channels`."
            )

        self.encoder: nn.Module = WaveletFusionEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            wavelet_channels=wavelet_channels,
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
        self.decoder: nn.Module = WaveletFusionDecoder(
            spatial_dims=spatial_dims,
            channels=channels,
            in_channels=latent_channels,
            wavelet_channels=wavelet_channels,
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


class WaveletFusionVAE(nn.Module):
    """Checkpointed WaveletFusion VAE exposed as a BrainMint compression model."""

    def __init__(
        self,
        *,
        autoencoder: nn.Module,
        ckpt_path: str | Path,
        state_key: str | None = "autoencoder",
        loader: str | None = None,
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
            target_name="wavelet_fusion_vae",
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        reconstruction, _, _ = self.model(x, *args, **kwargs)
        return reconstruction

    def reconstruct(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(x, *args, **kwargs)

    def run_inference(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, Any, Any]:
        reconstruction, z_mu, z_sigma = self.model(batch["image"])
        return reconstruction, z_mu, z_sigma
