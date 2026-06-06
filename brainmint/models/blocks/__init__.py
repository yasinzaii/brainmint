"""Reusable neural network building blocks for BrainMint models.

Import wavelet transforms from their concrete submodules so coefficient-order
compatibility remains explicit:

- ``brainmint.models.blocks.haar_dwt`` preserves the original Wavelet VAE/DWT ordering.
- ``brainmint.models.blocks.haar_wavelet_fusion`` is the recommended ordering for WaveletFusion code.
"""
