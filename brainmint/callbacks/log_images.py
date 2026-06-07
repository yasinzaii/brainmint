import contextlib
import gc
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from monai.data import MetaTensor
from monai.transforms import SaveImage

_LOG = logging.getLogger(__name__)


def get_tb_writer(trainer):
    logs = trainer.loggers if getattr(trainer, "loggers", None) else ([trainer.logger] if getattr(trainer, "logger", None) else [])
    for lg in logs:
        if isinstance(lg, pl.loggers.TensorBoardLogger):
            return lg.experiment  # SummaryWriter
    return None


class SaveMRIImages(pl.Callback):
    """
    After each validation epoch:
      • iterate the val dataloader,
      • collect ONE sample per requested modality (e.g., T1w/T2w/T1ce/FLAIR),
      • run module inference (preferred) or forward() as fallback,
      • save paired NIfTI files into: <run_dir>/<subdir>/epoch-XXX/.

    Filenames:
      <tag>_<MODALITY>_input.nii.gz
      <tag>_<MODALITY>_<OUTNAME>.nii.gz  (for each selected model output)
    """

    def __init__(
        self,
        modalities: Sequence[str] = ("T1w", "T2w", "T1ce", "FLAIR"),
        tag: str = "val",
        subdir: str = "val_samples",
        dirpath: str | None = None,  # Defaults to trainer.log_dir
        infer_kwarg_keys: Sequence[str] = ("image",),  # Batch keys to pass to inference
        infer_method_candidates: Sequence[str] = ("_run_inference",),
        fallback_to_forward: bool = True,
        output_names: Sequence[str] = ("output", "z_mu", "z_sigma"),  # Names for outputs, matching the order the model returns.
        save_outputs: Sequence[int | bool | str] = (1, 0, 0),  # Output to save, Boolean Mask | Indices | Output Names
        input_save_list: Sequence[str] | None = None,
        save_dtype: torch.dtype = torch.float32,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        output_activation: str | None = None,
        separate_folder: bool = False,
        dataset_module: Any | None = None,
        
        
    ) -> None:
        super().__init__()
        self.modalities = [m.lower() for m in modalities]
        self.tag = tag
        self.subdir = subdir
        self.dirpath = dirpath
        self.infer_method_candidates = list(infer_method_candidates)
        self.fallback_to_forward = fallback_to_forward
        self.infer_kwarg_keys = list(infer_kwarg_keys)
        self.input_save_list = list(input_save_list) if input_save_list is not None else []

        self.output_names = list(output_names)
        self.save_indices = self._normalize_save_selector(save_outputs, self.output_names)

        self.save_dtype = save_dtype if isinstance(save_dtype, torch.dtype) else getattr(
            torch, str(save_dtype).strip().lower().removeprefix("torch.")
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.output_activation = output_activation.lower() if isinstance(output_activation, str) and output_activation else None
        if self.output_activation not in (None, "sigmoid", "tanh"):
            raise ValueError(f"SaveMRIImages: unsupported output_activation={output_activation}")
        self.separate_folder = separate_folder
        self.dataset_module = dataset_module

    @staticmethod
    def _normalize_save_selector(selector: Sequence[int | bool | str], names: Sequence[str]) -> list[int]:
        """Normalize save_outputs → sorted unique indices."""
        n = len(names)
        lower_names = {nm.lower() for nm in names}
        name_map = {nm.lower(): i for i, nm in enumerate(names)}

        is_bool_mask = (n > 0 and len(selector) == n) and all(
            isinstance(x, bool) or (isinstance(x, int) and x in (0, 1)) for x in selector
        )
        if is_bool_mask:
            idxs = [i for i, flag in enumerate(selector) if bool(flag)]
            return sorted(set(idxs))

        is_indices = len(selector) <= n and all(isinstance(x, int) and 0 <= x < n for x in selector)
        if is_indices:
            return sorted(set(selector))

        is_names = len(selector) <= n and all(isinstance(x, str) and x.lower() in lower_names for x in selector)
        idxs: list[int] = []
        if is_names:
            for s in selector:
                j = name_map.get(s.lower())
                if j is not None:
                    idxs.append(j)

        if not idxs:
            _LOG.error(f"SaveMRIImages Callback - Invalid output selector after filtering: {selector}")
        return sorted(set(idxs))

    def _resolve_infer_fn(self, pl_module: pl.LightningModule) -> Callable | None:
        for name in self.infer_method_candidates:
            fn = getattr(pl_module, name, None)
            if callable(fn):
                _LOG.info(f"SaveMRIImages Callback - Using inference method '{name}'.")
                return fn
        if self.fallback_to_forward and hasattr(pl_module, "forward"):
            _LOG.warning(
                "SaveMRIImages Callback - None of %s found; falling back to forward().",
                self.infer_method_candidates,
            )
            return pl_module.forward
        _LOG.warning("SaveMRIImages Callback - No inference method and fallback disabled; will skip saving.")
        return None

    def _slice_like_batch(self, val: Any, batch_size: int, i: int):
        """Return the i-th sample if batched along dim 0; else return as-is."""
        if torch.is_tensor(val):
            if val.dim() > 0 and val.size(0) == batch_size:
                return val[i:i + 1]
            return val
        if isinstance(val, (list, tuple)) and len(val) == batch_size:
            return val[i]
        return val

    def _build_infer_kwargs(
        self,
        pl_module: pl.LightningModule,
        batch: dict[str, Any],
        batch_size: int,
        i: int,
    ) -> dict[str, Any]:
        """Build kwargs from configured keys; first entry is treated as input."""
        if not self.infer_kwarg_keys:
            return {}
        kwargs: dict[str, Any] = {}
        for k in self.infer_kwarg_keys:
            if k in batch:
                v = self._slice_like_batch(batch[k], batch_size, i)
                if torch.is_tensor(v):
                    v = v.to(pl_module.device, non_blocking=True)
                kwargs[k] = v
            else:
                raise KeyError(f"Key:{k} missing in batch, Available keys:{list(batch.keys())}")
        return {kk: vv for kk, vv in kwargs.items() if vv is not None}

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Only the main/global process writes
        if not trainer.is_global_zero:
            return

        run_dir = Path(self.dirpath) if self.dirpath else Path(trainer.log_dir or trainer.default_root_dir)
        epoch_dir = run_dir / self.subdir / f"epoch-{int(pl_module.current_epoch):03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        infer_fn = self._resolve_infer_fn(pl_module)
        if infer_fn is None:
            return

        # Choose which datamodule/dataloader to use
        if self.dataset_module is not None:
            dm = self.dataset_module
            try:
                if hasattr(dm, "setup"):
                    dm.setup()
            except Exception as e:
                _LOG.warning(f"... prepare/setup custom dataset module: {e}")
            if dm is None:
                _LOG.warning("SaveMRIImages Callback - No datamodule; skipping.")
                return
            vloader = dm.val_dataloader()
        else:
            dm = trainer.datamodule
            if dm is None:
                _LOG.warning("SaveMRIImages Callback - No datamodule; skipping.")
                return
            vloader = dm.val_dataloader()
        
        want = set(self.modalities)
        got: set[str] = set()
        _LOG.info(
            f"SaveMRIImages Callback - Collecting modalities {sorted(want)} for epoch {int(pl_module.current_epoch)}"
        )

        required = set(self.infer_kwarg_keys)

        for batch in vloader:
            if want == got:
                break

            missing = required - set(batch.keys())
            if missing:
                raise KeyError(
                    f"SaveMRIImages: missing required keys: {', '.join(sorted(missing))}. Present: {', '.join(sorted(batch.keys()))}"
                )

            mods = batch.get("modality", None)
            batch_size = int(batch[self.infer_kwarg_keys[0]].shape[0])  # Shape of First kwarg
            missing_modality = mods is None
            if missing_modality:
                _LOG.warning(
                    "SaveMRIImages Callback - Batch missing 'modality'; saving first available samples."
                )
                mod_list = [None] * batch_size
            else:
                mod_list = [m.lower() for m in batch["modality"]]

            remaining = len(want - got)
            saved_without_modality = 0
            for i in range(batch_size):
                if want == got:
                    break

                mod_i = mod_list[i]
                if mod_i is not None:
                    mod_i = mod_i.lower()
                    if (mod_i not in want) or (mod_i in got):
                        continue
                    mod_label = mod_i
                else:
                    if saved_without_modality >= remaining:
                        break
                    mod_label = f"sample{i:03d}"
                
                infer_kwargs = self._build_infer_kwargs(
                    pl_module=pl_module,
                    batch=batch,
                    batch_size=batch_size,
                    i=i,
                )

                plugin = getattr(trainer.strategy, "precision_plugin", None)
                ctx = plugin.forward_context() if plugin else contextlib.nullcontext()
                try:
                    with ctx, torch.no_grad():
                        result = infer_fn(infer_kwargs)
                except Exception as e:
                    kw_types = {k: type(v).__name__ for k, v in infer_kwargs.items()}
                    raise RuntimeError(
                        f"SaveMRIImages: inference failed for modality '{mod_i}'. "
                        f"Kwarg types: {kw_types}"
                    ) from e

                if isinstance(result, dict):
                    outputs = [result.get(nm, None) for nm in self.output_names]
                elif isinstance(result, (tuple, list)):
                    outputs = list(result)
                else:  # single tensor
                    outputs = [result]

                # Prepare & save input once
                save_inputs = []
                if not self.input_save_list:
                    save_inputs.append(self.infer_kwarg_keys[0])  # Fist Key (infer_kwarg_keys) is the input
                else:
                    save_inputs = self.input_save_list
                
                for inp_name in save_inputs:
                    inp = infer_kwargs[inp_name]
                    if inp.shape[0] != 1:
                        raise ValueError(f"SaveMRIImages - Input dimension error, Length of 1st Dim should be 1. Got:{inp.shape} ")
                    inp_vol = inp[0].detach().to(dtype=self.save_dtype).clamp(self.clamp_min, self.clamp_max).cpu()
                    if not isinstance(inp_vol, MetaTensor):
                        inp_vol = MetaTensor(inp_vol, meta={"filename_or_obj": "_.nii.gz"})
                    input_saver = SaveImage(
                        output_dir=str(epoch_dir),
                        output_postfix=f"{self.tag}_{mod_label}_input_{inp_name}",
                        output_ext=".nii.gz",
                        separate_folder=self.separate_folder,
                    )
                    input_saver(inp_vol)

                    del input_saver, inp_vol
                    gc.collect()

                for j in self.save_indices:
                    if j >= len(outputs) or outputs[j] is None:
                        continue
                    out = outputs[j]
                    out_vol = out[0].detach()
                    if self.output_activation == "sigmoid":
                        out_vol = torch.sigmoid(out_vol)
                    elif self.output_activation == "tanh":
                        out_vol = torch.tanh(out_vol)
                    out_vol = out_vol.to(self.save_dtype).clamp(self.clamp_min, self.clamp_max).cpu()

                    if isinstance(out_vol, MetaTensor):
                        out_meta_vol = out_vol
                    else:
                        
                        #TODO: FIX PASSING OF INPUT META INFO
                        out_meta = {}#getattr(inp_vol, "meta", None) or {}
                        out_meta_vol = MetaTensor(out_vol, meta=out_meta)

                    out_name = self.output_names[j] if j < len(self.output_names) else f"out{j}"
                    saver = SaveImage(
                        output_dir=str(epoch_dir),
                        output_postfix=f"{self.tag}_{mod_label}_output_{out_name}",
                        output_ext=".nii.gz",
                        separate_folder=self.separate_folder,
                    )
                    saver(out_meta_vol)

                    del saver, out_meta_vol
                    gc.collect()

                if mod_i is None:
                    saved_without_modality += 1
                    if saved_without_modality >= remaining:
                        got = set(want)
                else:
                    got.add(mod_i)
                _LOG.info(
                    f"SaveMRIImages Callback - Saved {mod_label} (input & {','.join(self.output_names[k] for k in self.save_indices if k < len(self.output_names))}) → {epoch_dir}"
                )

                del infer_kwargs, outputs
                gc.collect()

        missing = sorted(want - got)
        if missing:
            _LOG.warning(f"SaveMRIImages Callback - Missing modalities this epoch: {missing}")
        else:
            _LOG.info(f"SaveMRIImages Callback - Saved all requested modalities: {sorted(got)}")


        # After saving, if we used a custom dataset, tear it down and free memory
        if self.dataset_module is not None:
            del vloader, dm
            gc.collect()
            torch.cuda.empty_cache()
