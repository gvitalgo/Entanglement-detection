"""Machine-readable provenance helpers for thesis experiment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "tensorflow",
    "keras",
    "qutip",
    "matplotlib",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_logical_fingerprint(path: Path) -> str:
    """Hash array names, dtypes, shapes and values independent of NPZ timestamps."""

    digest = hashlib.sha256()
    with np.load(path) as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            if array.dtype.kind in {"U", "S", "O"}:
                digest.update(json.dumps(array.tolist(), ensure_ascii=False).encode("utf-8"))
            else:
                digest.update(array.tobytes())
    return digest.hexdigest()


def collect_environment() -> dict[str, Any]:
    package_versions: dict[str, str] = {}
    for package_name in PACKAGE_NAMES:
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = "NOT INSTALLED"

    tensorflow_devices: list[str] = []
    try:
        import tensorflow as tf

        tensorflow_devices = [device.name for device in tf.config.list_physical_devices()]
    except Exception as exc:  # pragma: no cover - environment-specific diagnostics
        tensorflow_devices = [f"UNAVAILABLE: {type(exc).__name__}: {exc}"]

    host_hardware: dict[str, Any] = {
        "cpu_model": "NOT DETERMINED",
        "physical_memory": "NOT DETERMINED",
        "machine_model": "NOT DETERMINED",
        "gpu_inventory": "NOT DETERMINED",
    }
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if not cpu:
                cpu = subprocess.run(
                    ["sysctl", "-n", "hw.model"],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            if not cpu:
                hardware = subprocess.run(
                    ["system_profiler", "SPHardwareDataType", "-json"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if hardware.returncode == 0 and hardware.stdout.strip():
                    hardware_payload = json.loads(hardware.stdout)["SPHardwareDataType"][0]
                    cpu = hardware_payload.get("chip_type", "")
                    host_hardware["physical_memory"] = hardware_payload.get("physical_memory", "NOT DETERMINED")
                    host_hardware["machine_model"] = hardware_payload.get("machine_model", "NOT DETERMINED")
            host_hardware["cpu_model"] = cpu or "NOT DETERMINED"
            displays = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                check=False,
                capture_output=True,
                text=True,
            )
            if displays.returncode == 0 and displays.stdout.strip():
                host_hardware["gpu_inventory"] = json.loads(displays.stdout)
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": sys.version, "executable": sys.executable},
        "packages": package_versions,
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {"logical_count": os.cpu_count(), "model": host_hardware["cpu_model"]},
        "host": {
            "machine_model": host_hardware["machine_model"],
            "physical_memory": host_hardware["physical_memory"],
        },
        "host_gpu_inventory": host_hardware["gpu_inventory"],
        "tensorflow_physical_devices": tensorflow_devices,
    }


def collect_git_state(repository_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "branch": run_git("branch", "--show-current"),
            "is_dirty": bool(run_git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"status": f"NOT DETERMINED: {type(exc).__name__}: {exc}"}
