"""
Exponential Moving Average (EMA) utilities.

EMA maintains a *shadow* copy of model parameters and updates it after each
optimizer step:

    shadow = decay * shadow + (1 - decay) * param

During sampling/inference you typically *swap* the live parameters to the EMA
version (then restore them afterwards).

This version supports tracking MULTIPLE modules at once by passing a mapping,
for example:
    ExponentialMovingAverage({"unet": unet, "demographics_encoder": enc}, cfg)
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import nn

ModuleOrDict = nn.Module | Mapping[str, nn.Module]


@dataclass
class EMAConfig:
    decay: float = 0.9999
    update_every: int = 1
    start_step: int = 0
    store_on_cpu: bool = False
    use_fp32_shadow: bool = True

    # If True: include requires_grad=False params in the shadow so the produced
    # state dict is fully loadable into the module.
    track_all_params: bool = True

    # If True: copy non-trainable params each update. Usually unnecessary
    # (frozen params don't change), but kept as a knob for edge cases.
    update_frozen: bool = False


class ExponentialMovingAverage:
    """Track EMA of one or more modules' parameters (and copy buffers)."""

    def __init__(self, module_or_modules: ModuleOrDict, cfg: EMAConfig):
        self.cfg = cfg
        self.num_updates: int = 0

        # Tracked modules (name -> module)
        self._modules: dict[str, nn.Module] = self._normalize_modules(module_or_modules)
        if not self._modules:
            raise ValueError("EMA initialized with no modules")

        # Fast reverse lookup for legacy single-module calls
        self._module_id_to_name: dict[int, str] = {id(m): name for name, m in self._modules.items()}

        # Shadow (EMA) state
        self._shadow_params: dict[str, dict[str, torch.Tensor]] = {}
        self._shadow_buffers: dict[str, dict[str, torch.Tensor]] = {}

        # Stored live state for swap/restore
        self._stored_params: dict[str, dict[str, torch.Tensor]] = {}
        self._stored_buffers: dict[str, dict[str, torch.Tensor]] = {}

        self._init_from(self._modules)

    def _normalize_modules(self, module_or_modules: ModuleOrDict) -> dict[str, nn.Module]:
        if isinstance(module_or_modules, nn.Module):
            return {"model": module_or_modules}
        if isinstance(module_or_modules, Mapping):
            out: dict[str, nn.Module] = {}
            for k, v in module_or_modules.items():
                if v is None:
                    continue
                if not isinstance(k, str):
                    raise TypeError(f"EMA module dict keys must be str, got {type(k)}")
                if not isinstance(v, nn.Module):
                    raise TypeError(f"EMA module dict values must be nn.Module, got {type(v)}")
                out[k] = v
            return out
        raise TypeError(f"Expected nn.Module or Mapping[str, nn.Module], got {type(module_or_modules)}")

    def _select_modules(self, module_or_modules: ModuleOrDict | None = None) -> dict[str, nn.Module]:
        """Resolve optional user input to a {name: module} mapping.

        Backward compatibility: if you pass a single module (e.g. self.unet) and
        it is one of the tracked modules, we map it back to its tracked name.
        """
        if module_or_modules is None:
            return self._modules

        if isinstance(module_or_modules, nn.Module):
            name = self._module_id_to_name.get(id(module_or_modules))
            if name is not None:
                return {name: self._modules[name]}
            return {"model": module_or_modules}

        return self._normalize_modules(module_or_modules)

    def _params(self, module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
        for name, p in module.named_parameters():
            if p is None:
                continue
            if (not self.cfg.track_all_params) and (not p.requires_grad):
                continue
            yield name, p

    def _buffers(self, module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
        for name, b in module.named_buffers():
            if torch.is_tensor(b):
                yield name, b

    def _shadow_device_for(self, ref: torch.Tensor) -> torch.device:
        return torch.device("cpu") if self.cfg.store_on_cpu else ref.device

    def _shadow_param_dtype_for(self, p: torch.Tensor) -> torch.dtype:
        if self.cfg.use_fp32_shadow and p.dtype.is_floating_point:
            return torch.float32
        return p.dtype

    def _clone_param(self, p: torch.Tensor) -> torch.Tensor:
        dev = self._shadow_device_for(p)
        dt = self._shadow_param_dtype_for(p)
        return p.detach().to(device=dev, dtype=dt).clone()

    def _clone_buffer(self, b: torch.Tensor) -> torch.Tensor:
        dev = self._shadow_device_for(b)
        return b.detach().to(device=dev, dtype=b.dtype).clone()

    def _init_from(self, modules: dict[str, nn.Module]) -> None:
        self._shadow_params.clear()
        self._shadow_buffers.clear()
        for mod_name, mod in modules.items():
            pmap: dict[str, torch.Tensor] = {}
            bmap: dict[str, torch.Tensor] = {}
            for name, p in self._params(mod):
                pmap[name] = self._clone_param(p)
            for name, b in self._buffers(mod):
                bmap[name] = self._clone_buffer(b)
            self._shadow_params[mod_name] = pmap
            self._shadow_buffers[mod_name] = bmap


    def reset(self, module_or_modules: ModuleOrDict | None = None) -> None:
        """Reset EMA to match the module(s) exactly (useful at start of fine-tune)."""
        self.num_updates = 0
        mods = self._select_modules(module_or_modules)
        if mods.keys() == self._modules.keys():
            self._init_from(mods)
        else:
            for mod_name, mod in mods.items():
                pmap: dict[str, torch.Tensor] = {}
                bmap: dict[str, torch.Tensor] = {}
                for name, p in self._params(mod):
                    pmap[name] = self._clone_param(p)
                for name, b in self._buffers(mod):
                    bmap[name] = self._clone_buffer(b)
                self._shadow_params[mod_name] = pmap
                self._shadow_buffers[mod_name] = bmap

    @torch.no_grad()
    def update(self, module_or_modules: ModuleOrDict | None = None, *, step: int) -> None:
        """Update EMA from module parameters and copy buffers.

        If ``module_or_modules`` is None, updates all tracked modules.
        """
        if step < int(self.cfg.start_step):
            return
        if int(self.cfg.update_every) > 1 and (step % int(self.cfg.update_every) != 0):
            return

        decay = float(self.cfg.decay)
        one_minus = 1.0 - decay

        mods = self._select_modules(module_or_modules)

        for mod_name, mod in mods.items():
            shadow_p = self._shadow_params.setdefault(mod_name, {})
            shadow_b = self._shadow_buffers.setdefault(mod_name, {})

            # EMA parameters
            for name, p in self._params(mod):
                if name not in shadow_p:
                    shadow_p[name] = self._clone_param(p)
                    continue

                shadow = shadow_p[name]
                if p.requires_grad:
                    src = p.detach().to(device=shadow.device, dtype=shadow.dtype)
                    shadow.mul_(decay).add_(src, alpha=one_minus)
                elif self.cfg.update_frozen:
                    src = p.detach().to(device=shadow.device, dtype=shadow.dtype)
                    shadow.copy_(src)

            # Copy buffers (no EMA)
            for name, b in self._buffers(mod):
                if name not in shadow_b:
                    shadow_b[name] = self._clone_buffer(b)
                    continue
                src_b = b.detach().to(device=shadow_b[name].device, dtype=shadow_b[name].dtype)
                shadow_b[name].copy_(src_b)

        self.num_updates += 1

    def ema_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return {module_name: module_state_dict_with_ema_tensors}."""
        out: dict[str, dict[str, torch.Tensor]] = {}
        for mod_name in self._modules.keys():
            combined: dict[str, torch.Tensor] = {}
            combined.update({k: v.detach().cpu() for k, v in self._shadow_params.get(mod_name).items()})
            combined.update({k: v.detach().cpu() for k, v in self._shadow_buffers.get(mod_name).items()})
            out[mod_name] = combined
        return out

    def state_dict(self) -> dict[str, object]:
        """Serializable EMA tracker state (includes cfg + ema_state_dict)."""
        return {
            "cfg": {
                "decay": float(self.cfg.decay),
                "update_every": int(self.cfg.update_every),
                "start_step": int(self.cfg.start_step),
                "store_on_cpu": bool(self.cfg.store_on_cpu),
                "use_fp32_shadow": bool(self.cfg.use_fp32_shadow),
                "track_all_params": bool(self.cfg.track_all_params),
                "update_frozen": bool(self.cfg.update_frozen),
            },
            "num_updates": int(self.num_updates),
            "ema_state_dict": self.ema_state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load EMA tracker state produced by :meth:`state_dict`."""
        cfg = state.get("cfg", {}) if isinstance(state, dict) else {}
        if isinstance(cfg, dict):
            self.cfg = EMAConfig(
                decay=float(cfg.get("decay", self.cfg.decay)),
                update_every=int(cfg.get("update_every", self.cfg.update_every)),
                start_step=int(cfg.get("start_step", self.cfg.start_step)),
                store_on_cpu=bool(cfg.get("store_on_cpu", self.cfg.store_on_cpu)),
                use_fp32_shadow=bool(cfg.get("use_fp32_shadow", self.cfg.use_fp32_shadow)),
                track_all_params=bool(cfg.get("track_all_params", self.cfg.track_all_params)),
                update_frozen=bool(cfg.get("update_frozen", self.cfg.update_frozen)),
            )

        self.num_updates = int(state.get("num_updates", 0)) if isinstance(state, dict) else 0

        ema_sd = state.get("ema_state_dict", {}) if isinstance(state, dict) else {}
        if not isinstance(ema_sd, Mapping):
            raise TypeError("EMA state['ema_state_dict'] must be a mapping")

        for mod_name, mod in self._modules.items():
            msd = ema_sd.get(mod_name)
            if msd is None:
                continue
            if not isinstance(msd, Mapping):
                raise TypeError(f"EMA state for module '{mod_name}' must be a mapping")

            buffer_names = {n for n, _ in self._buffers(mod)}
            param_names = {n for n, _ in self._params(mod)}

            pmap: dict[str, torch.Tensor] = {}
            bmap: dict[str, torch.Tensor] = {}
            for k, v in msd.items():
                if not isinstance(k, str) or (not torch.is_tensor(v)):
                    continue
                if k in buffer_names and k not in param_names:
                    bmap[k] = v.detach().clone()
                else:
                    pmap[k] = v.detach().clone()

            self._shadow_params[mod_name] = pmap
            self._shadow_buffers[mod_name] = bmap

    def to(self, device: torch.device) -> None:
        """Move shadow tensors to a device (no-op if store_on_cpu=True)."""
        if self.cfg.store_on_cpu:
            return
        for mod_name in list(self._shadow_params.keys()):
            for k, v in list(self._shadow_params[mod_name].items()):
                self._shadow_params[mod_name][k] = v.to(device=device)
        for mod_name in list(self._shadow_buffers.keys()):
            for k, v in list(self._shadow_buffers[mod_name].items()):
                self._shadow_buffers[mod_name][k] = v.to(device=device)


    @torch.no_grad()
    def store(self, module_or_modules: ModuleOrDict | None = None) -> None:
        """Store current params/buffers so they can be restored after a swap."""
        mods = self._select_modules(module_or_modules)
        for mod_name, mod in mods.items():
            self._stored_params[mod_name] = {name: p.detach().clone() for name, p in self._params(mod)}
            self._stored_buffers[mod_name] = {name: b.detach().clone() for name, b in self._buffers(mod)}

    @torch.no_grad()
    def copy_to(self, module_or_modules: ModuleOrDict | None = None) -> None:
        """Copy EMA params/buffers into the given module(s)."""
        mods = self._select_modules(module_or_modules)
        for mod_name, mod in mods.items():
            shadow_p = self._shadow_params.get(mod_name, {})
            shadow_b = self._shadow_buffers.get(mod_name, {})

            for name, p in self._params(mod):
                if name in shadow_p:
                    p.data.copy_(shadow_p[name].to(device=p.device, dtype=p.dtype))

            for name, b in self._buffers(mod):
                if name in shadow_b:
                    b.data.copy_(shadow_b[name].to(device=b.device, dtype=b.dtype))

    @torch.no_grad()
    def restore(self, module_or_modules: ModuleOrDict | None = None) -> None:
        """Restore params/buffers previously saved with :meth:`store`."""
        mods = self._select_modules(module_or_modules)
        for mod_name, mod in mods.items():
            sp = self._stored_params.get(mod_name, {})
            sb = self._stored_buffers.get(mod_name, {})

            for name, p in self._params(mod):
                if name in sp:
                    p.data.copy_(sp[name].to(device=p.device, dtype=p.dtype))

            for name, b in self._buffers(mod):
                if name in sb:
                    b.data.copy_(sb[name].to(device=b.device, dtype=b.dtype))

        for mod_name in list(mods.keys()):
            self._stored_params.pop(mod_name, None)
            self._stored_buffers.pop(mod_name, None)
