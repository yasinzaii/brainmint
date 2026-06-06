from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import (
    ConditioningBuilder,
    DiffusionPipeline,
    DiffusionSampler,
    LatentInput,
    LatentProvider,
    Postprocessor,
)


class LatentDiffusionGenerationPipeline(DiffusionPipeline):
    """Generic latent-diffusion generation pipeline.

    The pipeline owns the high-level inference flow only:

    1. ask a ``LatentProvider`` for the latent reference shape,
    2. build conditioning tensors,
    3. sample a latent with a ``DiffusionSampler``,
    4. optionally decode through ``ctx.modules['autoencoder']``,
    5. optionally postprocess the decoded sample.

    Model-family behavior belongs in the injected provider, conditioning builder,
    sampler, autoencoder, and postprocessor. Checkpoint loading and module
    construction are handled outside the pipeline by the inference runner/config.
    """

    required_modules: Set[str] = {"unet", "noise_scheduler"}

    def __init__(
        self,
        *,
        latent_provider: LatentProvider,
        conditioning: ConditioningBuilder,
        sampler: DiffusionSampler,
        postprocess: Optional[Postprocessor] = None,
        decode: bool = True,
        scale_factor_key: str = "scale_factor",
        expected_latent_channels: Optional[int] = None,
        force_float32_decode: bool = True,
    ) -> None:
        super().__init__()
        self.latent_provider = latent_provider
        self.conditioning = conditioning
        self.sampler = sampler
        self.postprocess = postprocess
        self.decode = bool(decode)
        self.scale_factor_key = str(scale_factor_key)
        self.expected_latent_channels = expected_latent_channels
        self.force_float32_decode = bool(force_float32_decode)

        if self.decode:
            self.required_modules = set(self.required_modules) | {"autoencoder"}

    def get_required_modules(self) -> Set[str]:
        required = set(self.required_modules)
        required |= self.latent_provider.get_required_modules()
        required |= self.conditioning.get_required_modules()
        required |= self.sampler.get_required_modules()
        if self.postprocess is not None:
            required |= self.postprocess.get_required_modules()
        return required

    @staticmethod
    def _decode_latent(autoencoder: Any, z: torch.Tensor, *, force_float32: bool) -> torch.Tensor:
        """Decode a latent using MONAI-LDM or generic autoencoder APIs."""

        z_in = z.float() if force_float32 else z
        if hasattr(autoencoder, "decode_stage_2_outputs"):
            return autoencoder.decode_stage_2_outputs(z_in)
        if hasattr(autoencoder, "decode"):
            return autoencoder.decode(z_in)
        raise AttributeError("Autoencoder does not expose decode_stage_2_outputs() or decode().")


    @torch.no_grad()
    def run(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> Dict[str, Any]:
        _ = ctx.get("unet", required=True)
        _ = ctx.get("noise_scheduler", required=True)
        autoencoder = ctx.get("autoencoder", required=self.decode) if self.decode else None

        latent_in = self.latent_provider.get_latent(batch, ctx=ctx)
        latent_ref = latent_in.ref
        if latent_ref is None:
            raise ValueError("LatentProvider returned LatentInput without 'ref' for shape inference.")

        if self.expected_latent_channels is not None and latent_ref.shape[1] != int(self.expected_latent_channels):
            raise ValueError(f"latent_ref has C={latent_ref.shape[1]} but expected {self.expected_latent_channels}")

        latent_seed = LatentInput(init=torch.randn_like(latent_ref, dtype=torch.float32))
        conditioning = self.conditioning.build(batch, latent_ref=latent_ref, ctx=ctx)

        sample_dict = self.sampler.sample_latent(latent=latent_seed, conditioning=conditioning, ctx=ctx)
        z = sample_dict.get("latent")
        if not isinstance(z, torch.Tensor):
            raise TypeError(
                f"Sampler must return dict with key 'latent' -> Tensor, got keys={list(sample_dict.keys())}"
            )

        out: Dict[str, Any] = {"latent": z, **{key: value for key, value in sample_dict.items() if key != "latent"}}

        if self.decode:
            scale_factor = float(ctx.scalar(self.scale_factor_key, 1.0))
            z_dec = z / scale_factor if scale_factor != 0.0 else z

            with torch.autocast(device_type=ctx.device.type, enabled=False):
                decoded = self._decode_latent(autoencoder, z_dec, force_float32=self.force_float32_decode)

            if self.postprocess is not None:
                ctx = ctx.with_updates(metadata={"modality": batch["modality"]})
                decoded = self.postprocess(decoded, ctx=ctx)

            out["sample"] = decoded

        return out
