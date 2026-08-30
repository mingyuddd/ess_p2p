#!/usr/bin/env python3
"""Build TensorRT engines from NVIDIA ESS ONNX models.

Expected portable model directory layout::

    models/
    ├── ess.onnx
    ├── light_ess.onnx
    └── plugins/
        ├── aarch64/ess_plugins.so
        └── x86_64/ess_plugins.so

Typical use on the target computer::

    python3 build_ess_engines.py --model-dir /path/to/models

The engine is intentionally built on the target computer because TensorRT
plans are sensitive to the GPU, TensorRT, CUDA, and plugin environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SUPPORTED_ARCHITECTURES = {"aarch64", "x86_64"}
DEFAULT_MODEL_NAMES = ("ess.onnx", "light_ess.onnx")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")
ELF_MACHINE_NAMES = {
    62: "x86_64",
    183: "aarch64",
}


@dataclass(frozen=True)
class BuildJob:
    onnx: Path
    plugin: Path
    engine: Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ess.onnx and/or light_ess.onnx into TensorRT engines "
            "with the ESS custom plugin for this computer."
        )
    )
    parser.add_argument(
        "onnx",
        nargs="*",
        type=Path,
        help=(
            "ONNX files to build. If omitted, ess.onnx and light_ess.onnx "
            "are discovered under --model-dir."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=SCRIPT_DIR,
        help=(
            "Directory containing the ONNX files and plugins/ "
            f"(default: {SCRIPT_DIR})."
        ),
    )
    parser.add_argument(
        "--plugin",
        type=Path,
        help=(
            "Explicit ess_plugins.so path. By default each ONNX uses "
            "<onnx-dir>/plugins/<machine>/ess_plugins.so."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Engine output directory (default: next to each ONNX file).",
    )
    parser.add_argument(
        "--trtexec",
        type=Path,
        help=(
            "Path to trtexec. Default search order: PATH, then "
            f"{DEFAULT_TRTEXEC}."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="TensorRT engine precision (default: fp16).",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Build the engine without running trtexec's post-build benchmark.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing engine only after a new build succeeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without building engines.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to trtexec.",
    )
    return parser.parse_args()


def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def resolve_trtexec(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    else:
        candidates.append(DEFAULT_TRTEXEC)
        from_path = shutil.which("trtexec")
        if from_path:
            candidates.append(Path(from_path))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved

    rendered = ", ".join(str(path) for path in candidates) or "none"
    fail(f"trtexec was not found or is not executable; checked: {rendered}")


def discover_onnx_files(args: argparse.Namespace) -> list[Path]:
    if args.onnx:
        files = [path.expanduser().resolve() for path in args.onnx]
    else:
        model_dir = args.model_dir.expanduser().resolve()
        files = [model_dir / name for name in DEFAULT_MODEL_NAMES]
        files = [path for path in files if path.is_file()]

    if not files:
        fail(
            "no ONNX model was found; pass ONNX paths or place ess.onnx and/or "
            "light_ess.onnx under --model-dir"
        )

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        if not path.is_file():
            fail(f"ONNX file does not exist: {path}")
        if path.suffix.lower() != ".onnx":
            fail(f"model must have an .onnx suffix: {path}")
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def elf_architecture(path: Path) -> str | None:
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None

    byte_order = header[5]
    if byte_order == 1:
        machine = struct.unpack_from("<H", header, 18)[0]
    elif byte_order == 2:
        machine = struct.unpack_from(">H", header, 18)[0]
    else:
        return None
    return ELF_MACHINE_NAMES.get(machine, f"ELF-machine-{machine}")


def check_plugin_dependencies(plugin: Path) -> None:
    ldd = shutil.which("ldd")
    if not ldd:
        print("WARNING: ldd is unavailable; plugin dependencies were not checked.")
        return

    result = subprocess.run(
        [ldd, str(plugin)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    missing = [line.strip() for line in result.stdout.splitlines() if "not found" in line]
    if missing:
        fail(
            "ESS plugin has missing runtime dependencies:\n  "
            + "\n  ".join(missing)
            + "\nInstall a matching CUDA/TensorRT/cuDNN environment first."
        )


def make_jobs(args: argparse.Namespace, onnx_files: Sequence[Path]) -> list[BuildJob]:
    machine = platform.machine()
    if machine not in SUPPORTED_ARCHITECTURES:
        fail(
            f"unsupported machine architecture {machine!r}; expected one of "
            f"{sorted(SUPPORTED_ARCHITECTURES)}"
        )

    explicit_plugin = args.plugin.expanduser().resolve() if args.plugin else None
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    jobs: list[BuildJob] = []

    for onnx in onnx_files:
        plugin = explicit_plugin or (onnx.parent / "plugins" / machine / "ess_plugins.so")
        plugin = plugin.resolve()
        if not plugin.is_file():
            fail(
                f"ESS plugin does not exist: {plugin}\n"
                f"Copy plugins/{machine}/ess_plugins.so beside the ONNX models "
                "or pass --plugin."
            )

        plugin_arch = elf_architecture(plugin)
        if plugin_arch is None:
            fail(f"ESS plugin is not a valid ELF shared library: {plugin}")
        if plugin_arch != machine:
            fail(
                f"ESS plugin architecture is {plugin_arch}, but this computer is "
                f"{machine}: {plugin}"
            )

        destination = output_dir or onnx.parent
        engine = destination / f"{onnx.stem}.engine"
        if engine.exists() and not args.force:
            fail(f"engine already exists: {engine}; use --force to replace it safely")
        jobs.append(BuildJob(onnx=onnx, plugin=plugin, engine=engine))

    for plugin in sorted({job.plugin for job in jobs}):
        check_plugin_dependencies(plugin)
    return jobs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_command(
    trtexec: Path,
    job: BuildJob,
    temporary_engine: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(trtexec),
        f"--onnx={job.onnx}",
        f"--saveEngine={temporary_engine}",
        f"--staticPlugins={job.plugin}",
    ]
    if args.precision == "fp16":
        command.append("--fp16")
    if args.skip_inference:
        command.append("--skipInference")
    if args.verbose:
        command.append("--verbose")
    return command


def write_metadata(
    job: BuildJob,
    trtexec: Path,
    command: Sequence[str],
    precision: str,
) -> Path:
    metadata_path = job.engine.with_suffix(job.engine.suffix + ".build.json")
    metadata = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "precision": precision,
        "trtexec": str(trtexec),
        "onnx": str(job.onnx),
        "onnx_sha256": sha256(job.onnx),
        "plugin": str(job.plugin),
        "plugin_sha256": sha256(job.plugin),
        "engine": str(job.engine),
        "engine_sha256": sha256(job.engine),
        "command": list(command),
    }
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{os.getpid()}"
    )
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    return metadata_path


def run_job(
    index: int,
    total: int,
    job: BuildJob,
    trtexec: Path,
    args: argparse.Namespace,
) -> None:
    temporary_engine = job.engine.with_name(
        f".{job.engine.name}.tmp-{os.getpid()}"
    )
    command = build_command(trtexec, job, temporary_engine, args)

    print(f"[{index}/{total}] ONNX   : {job.onnx}")
    print(f"[{index}/{total}] Plugin : {job.plugin}")
    print(f"[{index}/{total}] Engine : {job.engine}")
    print(f"[{index}/{total}] Command: {shlex.join(command)}")
    if args.dry_run:
        return

    job.engine.parent.mkdir(parents=True, exist_ok=True)
    temporary_engine.unlink(missing_ok=True)
    try:
        subprocess.run(command, check=True)
        if not temporary_engine.is_file() or temporary_engine.stat().st_size == 0:
            fail(f"trtexec succeeded but did not create a valid engine: {temporary_engine}")
        os.replace(temporary_engine, job.engine)
        metadata_path = write_metadata(job, trtexec, command, args.precision)
    except Exception:
        temporary_engine.unlink(missing_ok=True)
        raise

    print(f"[{index}/{total}] PASS: {job.engine}")
    print(f"[{index}/{total}] Metadata: {metadata_path}")


def main() -> int:
    try:
        args = parse_arguments()
        trtexec = resolve_trtexec(args.trtexec)
        onnx_files = discover_onnx_files(args)
        jobs = make_jobs(args, onnx_files)

        print(f"Target architecture: {platform.machine()}")
        print(f"TensorRT tool: {trtexec}")
        print(f"Precision: {args.precision}")
        for index, job in enumerate(jobs, start=1):
            run_job(index, len(jobs), job, trtexec, args)
        if args.dry_run:
            print("DRY RUN PASS: no engine was written.")
        else:
            print(f"All {len(jobs)} ESS engine build(s) completed successfully.")
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
