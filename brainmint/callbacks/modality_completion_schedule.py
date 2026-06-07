from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pytorch_lightning as pl

from brainmint.utils.schedules import PiecewiseSchedule

logger = logging.getLogger(__name__)


def _normalize_probs(probs: Mapping[str, float]) -> dict[str, float]:
    out = {str(k): float(v) for k, v in dict(probs).items()}
    s = float(sum(max(0.0, v) for v in out.values()))
    if s <= 0.0:
        raise ValueError(f"Cannot normalize probabilities with non-positive sum: {out}")
    return {k: max(0.0, v) / s for k, v in out.items()}


def _iter_transforms(root: Any) -> Iterable[Any]:
    """Depth-first walk over transform graphs (matches tests/data/test_translation_configs.py)."""
    if root is None:
        return
    stack = [root]
    seen = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        children: list[Any] = []
        nested = getattr(current, "_transform", None)
        if nested is not None and nested is not current:
            children.append(nested)

        transforms = getattr(current, "transforms", None)
        if transforms:
            children.extend(list(transforms))

        extra_start = getattr(current, "extra_xforms_start", None)
        if extra_start:
            children.extend(list(extra_start))

        extra_end = getattr(current, "extra_xforms_end", None)
        if extra_end:
            children.extend(list(extra_end))

        stack.extend(reversed(children))


def _find_choice_states(dm: Any) -> list[Any]:
    st: list[Any] = []
    tf = getattr(dm, "train_tf", None)
    root = getattr(tf, "transform", None) or tf
    for t in _iter_transforms(root):
        state = getattr(t, "state", None)
        if state is not None and hasattr(state, "get_choices") and hasattr(state, "set_choices"):
            st.append(state)
    return st


def _find_sampling_states(dm: Any) -> list[Any]:
    st: list[Any] = []
    tf = getattr(dm, "train_tf", None)
    root = getattr(tf, "transform", None) or tf
    for t in _iter_transforms(root):
        ss = getattr(t, "sampling_state", None)
        if ss is not None and hasattr(ss, "get_config") and hasattr(ss, "set_config"):
            st.append(ss)
    return st


class ModalityCompletionScheduleCallback(pl.Callback):
    """
    Generic, config-driven epoch-boundary scheduler for modality completion experiments.

    What it can update:
      1) Bucket sampler probabilities via DataModule.set_bucket_probs(...)
      2) Stream selection probabilities via SharedChoiceState.set_choices(...)
      3) Partial sampling config via SharedSamplingState.set_config(...)

    All behavior is driven by YAML config (step/linear schedules). No stage hardcoding.
    """

    def __init__(
        self,
        *,
        bucket_schedule: Sequence[Mapping[str, Any]] | None = None,
        choice_schedule: Sequence[Mapping[str, Any]] | None = None,
        sampling_schedule: Sequence[Mapping[str, Any]] | None = None,
        update_on: str = "epoch_start",
        strict: bool = True,
        normalize_bucket_probs: bool = True,
        normalize_choice_probs: bool = True,
        auto_complement_two_way: bool = True,
        log_updates: bool = True,
    ) -> None:
        super().__init__()
        self.update_on = str(update_on).lower().strip()
        if self.update_on not in ("epoch_start", "epoch_end"):
            raise ValueError("update_on must be one of: 'epoch_start', 'epoch_end'")

        self.strict = bool(strict)
        self.normalize_bucket_probs = bool(normalize_bucket_probs)
        self.normalize_choice_probs = bool(normalize_choice_probs)
        self.auto_complement_two_way = bool(auto_complement_two_way)
        self.log_updates = bool(log_updates)

        self.bucket_sched = PiecewiseSchedule(bucket_schedule, name="bucket_schedule")
        self.choice_sched = PiecewiseSchedule(choice_schedule, name="choice_schedule")
        self.sampling_sched = PiecewiseSchedule(sampling_schedule, name="sampling_schedule")

        self._choice_states: list[Any] = []
        self._sampling_states: list[Any] = []

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            return
        dm = trainer.datamodule
        self._choice_states = _find_choice_states(dm)
        self._sampling_states = _find_sampling_states(dm)

        if self.strict:
            if (self.choice_sched.steps or self.choice_sched.lines) and not self._choice_states:
                raise RuntimeError("choice_schedule provided but no SharedChoiceState found in train transforms.")
            if (self.sampling_sched.steps or self.sampling_sched.lines) and not self._sampling_states:
                raise RuntimeError("sampling_schedule provided but no SharedSamplingState found in train transforms.")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.update_on == "epoch_start":
            self._apply(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.update_on == "epoch_end":
            self._apply(trainer)

    def _apply(self, trainer: pl.Trainer) -> None:
        epoch = int(trainer.current_epoch)
        dm = trainer.datamodule

        bucket_probs = self.bucket_sched.value_at(epoch)
        choice_updates = self.choice_sched.value_at(epoch)
        sampling_updates = self.sampling_sched.value_at(epoch)

        if bucket_probs is not None:
            self._apply_bucket_probs(dm, bucket_probs, epoch)

        if choice_updates is not None:
            self._apply_choice_probs(choice_updates, epoch)

        if sampling_updates is not None:
            self._apply_sampling_config(sampling_updates)

    # ---------------------------
    # Bucket probs
    # ---------------------------
    def _apply_bucket_probs(self, dm: Any, value: Any, epoch: int) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(f"bucket_schedule value must be a mapping bucket->prob, got {type(value)}")

        probs = {str(k): float(v) for k, v in dict(value).items()}
        if self.normalize_bucket_probs:
            probs = _normalize_probs(probs)

        setter = getattr(dm, "set_bucket_probs", None)
        if not callable(setter):
            msg = "DataModule has no set_bucket_probs(); cannot apply bucket_schedule."
            if self.strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            return

        setter(probs)

        if self.log_updates:
            logger.info("[Schedule] bucket_probs(epoch=%d): %s", int(epoch), probs)

    # ---------------------------
    # Choice probs (SharedChoiceState)
    # ---------------------------
    def _apply_choice_probs(self, value: Any, epoch: int) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(f"choice_schedule value must be a nested mapping, got {type(value)}")

        updates: dict[str, Any] = dict(value)

        for state in self._choice_states:
            cur = dict(state.get_choices())
            changed = False

            for bucket, mods in updates.items():
                bucket = str(bucket)
                if bucket not in cur:
                    if self.strict:
                        raise KeyError(f"choice_schedule refers to unknown bucket '{bucket}'")
                    logger.warning("choice_schedule: unknown bucket '%s' (ignored)", bucket)
                    continue
                if not isinstance(mods, Mapping):
                    if self.strict:
                        raise TypeError(f"choice_schedule[{bucket}] must be a mapping modality->alias_probs")
                    continue

                bcfg = cur[bucket]
                if not isinstance(bcfg, Mapping):
                    if self.strict:
                        raise TypeError(f"choices[{bucket}] must be mapping")
                    continue

                for mod, alias_probs in mods.items():
                    mod_key = str(mod).lower()
                    if not isinstance(alias_probs, Mapping):
                        if self.strict:
                            raise TypeError(f"choice_schedule[{bucket}][{mod_key}] must be mapping alias->prob")
                        continue

                    # Find modality config; fall back to "*"
                    mcfg = bcfg.get(mod_key, None)
                    wildcard = False
                    if mcfg is None:
                        mcfg = bcfg.get("*", None)
                        wildcard = True
                    if mcfg is None:
                        if self.strict:
                            raise KeyError(f"choices missing config for bucket='{bucket}' modality='{mod_key}' (and no '*')")
                        logger.warning("choices missing config for bucket='%s' modality='%s' (ignored)", bucket, mod_key)
                        continue
                    if not isinstance(mcfg, Mapping):
                        if self.strict:
                            raise TypeError(f"choices[{bucket}][{mod_key if not wildcard else '*'}] must be mapping")
                        continue

                    streams = dict(mcfg.get("streams") or {})
                    if not streams:
                        if self.strict:
                            raise KeyError(f"choices[{bucket}][{mod_key if not wildcard else '*'}].streams is empty")
                        continue

                    probs = dict(mcfg.get("probs") or {})

                    # apply updates (set)
                    for alias, p in alias_probs.items():
                        probs[str(alias)] = float(p)

                    if self.strict:
                        unknown_aliases = set(str(a) for a in alias_probs.keys()) - set(streams.keys())
                        if unknown_aliases:
                            raise KeyError(
                                f"choice_schedule[{bucket}][{mod_key}] refers to unknown stream aliases: "
                                f"{sorted(unknown_aliases)}"
                            )

                    # Optional complement for 2-way choices when only one alias was specified
                    if self.auto_complement_two_way:
                        defined_aliases = list(streams.keys())
                        specified = [str(a) for a in alias_probs.keys() if str(a) in defined_aliases]
                        if len(defined_aliases) == 2 and len(specified) == 1:
                            a0 = specified[0]
                            other = defined_aliases[0] if defined_aliases[1] == a0 else defined_aliases[1]
                            p0 = float(probs.get(a0, 0.0))
                            probs[other] = max(0.0, 1.0 - p0)

                    # Keep only aliases defined in streams
                    probs = {a: float(probs.get(a, 0.0)) for a in streams.keys()}

                    if self.normalize_choice_probs:
                        probs = _normalize_probs(probs)

                    new_mcfg = dict(mcfg)
                    new_mcfg["probs"] = probs

                    new_bucket_cfg = dict(cur[bucket])
                    if wildcard and "*" in new_bucket_cfg:
                        new_bucket_cfg["*"] = new_mcfg
                    else:
                        new_bucket_cfg[mod_key] = new_mcfg
                    cur[bucket] = new_bucket_cfg

                    changed = True

            if changed:
                try:
                    state.set_epoch(int(epoch))
                except Exception:
                    pass
                state.set_choices(cur)
                if self.log_updates:
                    logger.info("[Schedule] choice_probs(epoch=%d): updated SharedChoiceState", epoch)

    # ---------------------------
    # Sampling config (SharedSamplingState)
    # ---------------------------
    def _apply_sampling_config(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(f"sampling_schedule value must be a mapping, got {type(value)}")
        upd = {str(k): v for k, v in dict(value).items()}
        if self.strict:
            if "sigma_prob" in upd:
                upd["sigma_prob"] = float(upd["sigma_prob"])
                if not (0.0 <= upd["sigma_prob"] <= 1.0):
                    raise ValueError("sampling_schedule sigma_prob must be between 0 and 1.")
            if "sigma_alpha" in upd:
                upd["sigma_alpha"] = float(upd["sigma_alpha"])
                if upd["sigma_alpha"] < 0.0:
                    raise ValueError("sampling_schedule sigma_alpha must be >= 0.")

        for ss in self._sampling_states:
            cur = dict(ss.get_config())
            cur.update(upd)
            ss.set_config(cur)

        if self.log_updates:
            logger.info("[Schedule] sampling_config: %s", upd)
