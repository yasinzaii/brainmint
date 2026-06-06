from __future__ import annotations

"""BraSyn / BrainLesion MissingMRI runtime helpers.

This module owns the HPC-specific BraSyn runtime behavior. The upstream
``brats`` package drives MissingMRI inference through Singularity/Apptainer;
BrainMint patches that runner so container images, temporary files, and writable
overlays land in configured scratch/cache locations instead of transient node
``/tmp`` paths.
"""

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_LOG = logging.getLogger(__name__)


def _ensure_brats_installed() -> None:
    try:
        import brats  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The BrainLesion/BraTS python package (`brats`) must be installed to run "
            "BraSyn MissingMRI baselines. Install it (pip/conda) in your env."
        ) from exc


def _resolve_container_cli(preferred: Optional[str] = None) -> str:
    """Return the first available Apptainer/Singularity binary."""

    candidates: list[str] = []
    if preferred:
        candidates.append(str(preferred))
    candidates.extend(["apptainer", "singularity"])

    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        if shutil.which(name):
            return name

    raise RuntimeError(
        "Neither 'apptainer' nor 'singularity' was found on PATH. "
        "Please install one or update PATH."
    )


def _run_singularity_cmd(cmd: list[str]) -> list[str]:
    """Run a container command and stream merged stdout/stderr through logging."""

    _LOG.info("EXEC: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )

    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        output_lines.append(line)
        _LOG.info("%s", line)

    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, "\n".join(output_lines))

    return output_lines


def resolve_tmp_root(cfg_value: Optional[str | Path] = None) -> Path:
    """Resolve the host temporary root for BraSyn container execution."""

    if cfg_value is not None:
        return Path(str(cfg_value)).expanduser().resolve()

    for key in ("APPTAINER_TMPDIR", "SINGULARITY_TMPDIR", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()

    return Path(tempfile.gettempdir()).resolve()


def resolve_singularity_image_dir(cfg_value: Optional[str | Path] = None) -> Optional[Path]:
    """Resolve the persistent Singularity/Apptainer image cache directory."""

    if cfg_value is not None:
        return Path(str(cfg_value)).expanduser().resolve()

    for key in ("APPTAINER_CACHEDIR", "SINGULARITY_CACHEDIR"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve() / "brats_singularity_images"

    return None


@dataclass
class BraSynRuntime:
    tmp_root: Optional[str] = None
    overlay_mode: str = "tmpfs"  # tmpfs | overlay | none
    overlay_size_mb: int = 256
    overlay_readonly: bool = True
    keep_overlay: bool = True
    use_fakeroot: bool = False
    singularity_image_dir: Optional[str] = None


def _patch_brats_singularity_runner(runtime: BraSynRuntime) -> None:
    """Patch upstream ``brats`` Singularity execution for HPC use.

    Upstream uses node-local temp paths and creates writable overlay resources in
    ways that are expensive or brittle on shared HPC filesystems. This patch keeps
    images/additional files persistent and lets callers choose tmpfs, persistent
    overlay, or no writable layer.
    """

    _ensure_brats_installed()
    from brats.core import singularity as brats_singularity

    sing_img_dir = resolve_singularity_image_dir(runtime.singularity_image_dir)
    runtime.singularity_image_dir = str(sing_img_dir) if sing_img_dir else None

    if getattr(brats_singularity, "__brainmint_patched__", False):
        return

    if runtime.tmp_root:
        tmp = Path(runtime.tmp_root)
        tmp.mkdir(parents=True, exist_ok=True)

        session_dir = tmp / "apptainer_session"
        session_dir.mkdir(parents=True, exist_ok=True)

        os.environ["APPTAINER_SESSIONDIR"] = str(session_dir)
        os.environ["SINGULARITY_SESSIONDIR"] = str(session_dir)
        os.environ["APPTAINER_TMPDIR"] = str(tmp)
        os.environ["SINGULARITY_TMPDIR"] = str(tmp)
        os.environ["TMPDIR"] = str(tmp)
        os.environ["APPTAINERENV_TMPDIR"] = "/tmp"
        os.environ["APPTAINERENV_RAY_TMPDIR"] = "/tmp/ray"
        tempfile.tempdir = str(tmp)

    container_cli = _resolve_container_cli(preferred="apptainer" if runtime.use_fakeroot else None)

    if runtime.singularity_image_dir:
        image_root = Path(runtime.singularity_image_dir)
        image_root.mkdir(parents=True, exist_ok=True)

        def _ensure_image_patched(image: str) -> str:
            image_path = image_root / image.replace(":", "_")
            if not image_path.exists():
                _LOG.info("Building BraSyn Singularity sandbox: %s", image_path)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [container_cli, "build", "--sandbox"]
                if runtime.use_fakeroot:
                    cmd.append("--fakeroot")
                cmd += [str(image_path), f"docker://{image}"]
                subprocess.run(cmd, check=True)
            _LOG.info("Using BraSyn Singularity sandbox: %s", image_path)
            return str(image_path)

        brats_singularity._ensure_image = _ensure_image_patched  # type: ignore[assignment]

    try:
        from brats import constants as brats_constants
        from brats.utils import zenodo as brats_zenodo
        import brats.core.docker as brats_docker

        if runtime.singularity_image_dir:
            add_root = Path(runtime.singularity_image_dir) / "brats_additional_files"
            add_root.mkdir(parents=True, exist_ok=True)

            brats_constants.ADDITIONAL_FILES_FOLDER = add_root
            brats_zenodo.ADDITIONAL_FILES_FOLDER = add_root

            original_check = brats_zenodo.check_additional_files_path
            memo: Dict[str, Path] = {}

            def check_additional_files_path_local_first(record_id: str) -> Path:
                if record_id in memo:
                    return memo[record_id]

                pattern = f"{record_id}_v*.*.*"
                matching = [path for path in add_root.glob(pattern) if path.is_dir()]
                if not matching:
                    matching = [path for path in add_root.glob(f"{record_id}*") if path.is_dir()]

                latest = brats_zenodo._get_latest_version_folder_name(matching)  # type: ignore[attr-defined]
                if latest:
                    path = add_root / latest
                    memo[record_id] = path
                    _LOG.info("Using cached BraSyn additional files: %s", path)
                    return path

                path = original_check(record_id)
                memo[record_id] = path
                return path

            brats_zenodo.check_additional_files_path = check_additional_files_path_local_first  # type: ignore[assignment]
            brats_docker.check_additional_files_path = check_additional_files_path_local_first  # type: ignore[assignment]
            if hasattr(brats_singularity, "check_additional_files_path"):
                brats_singularity.check_additional_files_path = check_additional_files_path_local_first  # type: ignore[assignment]
    except Exception as exc:
        _LOG.debug("Could not patch BraSyn additional-file cache handling: %s", exc)

    def run_container_patched(
        algorithm: Any,
        data_path: Path,
        output_path: Path,
        cuda_devices: str,
        force_cpu: bool,
        internal_external_name_map: Optional[Dict[str, str]] = None,
        overlay_size: int = 1024,
    ) -> None:
        """Drop-in replacement for ``brats.core.singularity.run_container``."""

        overlay_mode = str(runtime.overlay_mode).lower()
        if overlay_mode not in {"tmpfs", "overlay", "none", "false", "0"}:
            raise ValueError(f"Unknown overlay_mode {runtime.overlay_mode!r}")
        if overlay_size <= 0:
            raise ValueError("Overlay size must be greater than 0.")

        get_additional_files_path = getattr(brats_singularity, "_get_additional_files_path")
        get_volume_mappings_mlcube = getattr(brats_singularity, "_get_volume_mappings_mlcube")
        get_volume_mappings_docker_only = getattr(brats_singularity, "_get_volume_mappings_docker_only")
        handle_device_requests = getattr(brats_singularity, "_handle_device_requests")
        sanity_check_output = getattr(brats_singularity, "_sanity_check_output")
        parameters_dir = getattr(brats_singularity, "PARAMETERS_DIR")
        time_mod = __import__("time")

        image = brats_singularity._ensure_image(image=algorithm.run_args.docker_image)  # type: ignore[misc]
        additional_files_path = get_additional_files_path(algorithm)
        output_path.mkdir(parents=True, exist_ok=True)

        command_args = brats_singularity._build_command_args(algorithm=algorithm)  # type: ignore[attr-defined]
        if algorithm.meta.year <= 2024:
            volume_mappings = get_volume_mappings_mlcube(
                data_path=data_path,
                additional_files_path=additional_files_path,
                output_path=output_path,
                parameters_path=parameters_dir,
            )
            args = ["infer", *command_args]
        else:
            volume_mappings = get_volume_mappings_docker_only(
                data_path=data_path,
                output_path=output_path,
            )
            args = None

        device_requests = handle_device_requests(
            algorithm=algorithm,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )
        singularity_bindings = brats_singularity._convert_volume_mappings_to_singularity_format(volume_mappings)  # type: ignore[attr-defined]

        options: list[str] = []
        if runtime.use_fakeroot:
            options.append("--fakeroot")
        if len(device_requests) > 0 and not force_cpu:
            options.append("--nv")
            _LOG.info("Using CUDA devices: %s", cuda_devices)

        docker_working_dir = brats_singularity._get_docker_working_dir(algorithm.run_args.docker_image)  # type: ignore[attr-defined]
        if docker_working_dir is not None:
            options += ["--cwd", str(docker_working_dir)]
        else:
            brats_singularity.logger.warning(
                "Docker working directory not found. Using default working directory."
            )

        overlay_path: Optional[Path] = None
        overlay_created = False
        if overlay_mode == "tmpfs":
            options.append("--writable-tmpfs")
            _LOG.info("Using BraSyn tmpfs overlay: --writable-tmpfs")
        elif overlay_mode == "overlay":
            size_mb = int(runtime.overlay_size_mb) if runtime.overlay_size_mb else int(overlay_size)
            if size_mb <= 0:
                size_mb = 256
            image_path = Path(image)
            overlay_path = image_path.parent / f"{image_path.name}_overlay.img"
            overlay_spec = f"{overlay_path}:ro" if runtime.overlay_readonly else str(overlay_path)
            options += ["--overlay", overlay_spec]
            _LOG.info("Using BraSyn persistent overlay: %s", overlay_spec)
            if not overlay_path.exists():
                cmd = [container_cli, "overlay", "create"]
                if runtime.use_fakeroot:
                    cmd.append("--fakeroot")
                cmd += ["--size", str(size_mb), str(overlay_path)]
                subprocess.run(cmd, check=True)
                overlay_created = True

        brats_singularity.logger.info("Starting inference")
        start_time = time_mod.time()
        try:
            cmd = [container_cli, "run"]
            for binding in singularity_bindings:
                cmd += ["--bind", binding]
            cmd += options
            cmd.append(image)
            if args:
                cmd += args

            container_output = _run_singularity_cmd(cmd)
            sanity_check_output(
                data_path=data_path,
                output_path=output_path,
                container_output="\n".join(container_output),
                internal_external_name_map=internal_external_name_map,
            )
        finally:
            if overlay_mode == "overlay" and overlay_created and overlay_path is not None and not runtime.keep_overlay:
                try:
                    overlay_path.unlink()
                except FileNotFoundError:
                    pass

        brats_singularity.logger.info(
            "Finished inference in %.2f seconds", time_mod.time() - start_time
        )

    brats_singularity.run_container = run_container_patched  # type: ignore[assignment]
    try:
        import brats.core.brats_algorithm as brats_algorithm

        brats_algorithm.run_singularity_container = run_container_patched  # type: ignore[attr-defined]
    except Exception as exc:
        _LOG.debug("Could not patch brats_algorithm Singularity runner: %s", exc)

    setattr(brats_singularity, "__brainmint_patched__", True)
