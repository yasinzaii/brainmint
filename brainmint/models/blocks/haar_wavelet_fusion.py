"""Corrected 3D Haar-DWT ordering used by WaveletFusion models.

Prefer this module for new WaveletFusion-style code. Keep
``brainmint.models.blocks.haar_dwt`` for Wavelet VAE and DWT checkpoint/config
compatibility.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange

class HaarWaveletTransform3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def conv():
            return nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0, bias=False)

        self.h_conv   = conv()
        self.g_conv   = conv()
        self.hh_conv  = conv()
        self.gh_conv  = conv()
        self.h_v_conv = conv()
        self.g_v_conv = conv()
        self.hh_v_conv= conv()
        self.gh_v_conv= conv()

        self._initialize_weights()

    def _initialize_weights(self):
        h = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) * 0.3536           # LLL
        g = torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]) * 0.3536       # LLH
        hh = torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]) * 0.3536      # LHL
        gh = torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]) * 0.3536      # LHH
        h_v = torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]) * 0.3536     # HLL
        g_v = torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]) * 0.3536     # HLH
        hh_v = torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]) * 0.3536    # HHL
        gh_v = torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]) * 0.3536    # HHH
        h = h.view(1, 1, 2, 2, 2)
        g = g.view(1, 1, 2, 2, 2)
        hh = hh.view(1, 1, 2, 2, 2)
        gh = gh.view(1, 1, 2, 2, 2)
        h_v = h_v.view(1, 1, 2, 2, 2)
        g_v = g_v.view(1, 1, 2, 2, 2)
        hh_v = hh_v.view(1, 1, 2, 2, 2)
        gh_v = gh_v.view(1, 1, 2, 2, 2)
        
        with torch.no_grad():
            self.h_conv.weight.copy_(h.to(self.h_conv.weight.device).to(self.h_conv.weight.dtype))
            self.g_conv.weight.copy_(g.to(self.g_conv.weight.device).to(self.g_conv.weight.dtype))
            self.hh_conv.weight.copy_(hh.to(self.hh_conv.weight.device).to(self.hh_conv.weight.dtype))
            self.gh_conv.weight.copy_(gh.to(self.gh_conv.weight.device).to(self.gh_conv.weight.dtype))
            self.h_v_conv.weight.copy_(h_v.to(self.h_v_conv.weight.device).to(self.h_v_conv.weight.dtype))
            self.g_v_conv.weight.copy_(g_v.to(self.g_v_conv.weight.device).to(self.g_v_conv.weight.dtype))
            self.hh_v_conv.weight.copy_(hh_v.to(self.hh_v_conv.weight.device).to(self.hh_v_conv.weight.dtype))
            self.gh_v_conv.weight.copy_(gh_v.to(self.gh_v_conv.weight.device).to(self.gh_v_conv.weight.dtype))
        
        self.h_conv.requires_grad_(False)
        self.g_conv.requires_grad_(False)
        self.hh_conv.requires_grad_(False)
        self.gh_conv.requires_grad_(False)
        self.h_v_conv.requires_grad_(False)
        self.g_v_conv.requires_grad_(False)
        self.hh_v_conv.requires_grad_(False)
        self.gh_v_conv.requires_grad_(False)

    def forward(self, x):
        assert x.dim() == 5
        b = x.shape[0]
        c = x.shape[1]
        
        x = rearrange(x, "b c d h w -> (b c) 1 d h w")
        n_dim = x.shape[0]
        outputs = []
        for i in range(n_dim):
            y = x[i: i+1]
            outputs.append(self.h_conv(y))
            outputs.append(self.g_conv(y))
            outputs.append(self.hh_conv(y))
            outputs.append(self.h_v_conv(y)) # Swapped - Now: HLL, before LHH
            outputs.append(self.gh_conv(y))  # Swapped - Now: LHH, before HLL
            outputs.append(self.g_v_conv(y))
            outputs.append(self.hh_v_conv(y))
            outputs.append(self.gh_v_conv(y))
        
        outputs = torch.cat(outputs, dim=0)
        
        # Order Confirmation Test.
        # Requirement:
        # I want the following order, having LLL for all channels first and so on...
        # [LLL(ch1..C),  LLH(ch1..C),  ...,  HHH(ch1..C)]

        # s1 = rearrange(outputs, "(b c k) 1 d h w -> b (k c) d h w", b=b, c=c, k=8)
        # lll1 = s1[:, :c]  # contiguous block of LLL (k=0)

        # lll2 = rearrange(outputs, "(b c k) 1 d h w -> b k c d h w", k=8, c=c)[:, 0]

        # lll3 = rearrange(s1, "b (k c) d h w -> b k c d h w", k=8, c=c)[:, 0]

        # assert torch.allclose(lll1, lll2)
        # assert torch.allclose(lll1, lll3)

        ret_outputs = rearrange(outputs, "(b c k) 1 d h w -> b (k c) d h w", b=b, c=c, k=8)
        return ret_outputs
    

class InverseHaarWaveletTransform3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer('h', 
            torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('g', 
            torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('hh', 
            torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('gh', 
            torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('h_v', 
            torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('g_v', 
            torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('hh_v', 
            torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )
        self.register_buffer('gh_v', 
            torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]).view(1, 1, 2, 2, 2) * 0.3536
        )

    def forward(self, coeffs):
        assert coeffs.dim() == 5, "coeffs must be [B, 8C, D/2, H/2, W/2]"
        b, ch, d, h, w = coeffs.shape
        assert ch % 8 == 0, "channel dim must be multiple of 8"
        #c = ch // 8
        
        # With (k, c) packing we can just chunk on channels into 8 equal blocks (each size c).
        # Order must match the FORWARD order (we used: LLL, LLH, LHL, HLL, LHH, HLH, HHL, HHH).
        (
            low_low_low,
            low_low_high,
            low_high_low,
            high_low_low,  # Swapped - Now: HLL, before LHH
            low_high_high, # Swapped - Now: LHH, before HLL
            high_low_high,
            high_high_low,
            high_high_high,
        ) = coeffs.chunk(8, dim=1)

        low_low_low = rearrange(low_low_low, "b c d h w -> (b c) 1 d h w")
        low_low_high = rearrange(low_low_high, "b c d h w -> (b c) 1 d h w")
        low_high_low = rearrange(low_high_low, "b c d h w -> (b c) 1 d h w")
        low_high_high = rearrange(low_high_high, "b c d h w -> (b c) 1 d h w")
        high_low_low = rearrange(high_low_low, "b c d h w -> (b c) 1 d h w")
        high_low_high = rearrange(high_low_high, "b c d h w -> (b c) 1 d h w")
        high_high_low = rearrange(high_high_low, "b c d h w -> (b c) 1 d h w")
        high_high_high = rearrange(high_high_high, "b c d h w -> (b c) 1 d h w")

        low_low_low = F.conv_transpose3d(low_low_low, self.h, stride=2)
        low_low_high = F.conv_transpose3d(low_low_high, self.g, stride=2)
        low_high_low = F.conv_transpose3d(low_high_low, self.hh, stride=2)
        low_high_high = F.conv_transpose3d(low_high_high, self.gh, stride=2)
        high_low_low = F.conv_transpose3d(high_low_low, self.h_v, stride=2)
        high_low_high = F.conv_transpose3d(high_low_high, self.g_v, stride=2)
        high_high_low = F.conv_transpose3d(high_high_low, self.hh_v, stride=2)
        high_high_high = F.conv_transpose3d(high_high_high, self.gh_v, stride=2)

        reconstructed = (
            low_low_low
            + low_low_high
            + low_high_low
            + low_high_high
            + high_low_low
            + high_low_high
            + high_high_low
            + high_high_high
        )
            
        reconstructed = rearrange(reconstructed, "(b c) 1 d h w -> b c d h w", b=b)
        return reconstructed
