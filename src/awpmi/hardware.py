"""The machine as detected at run time (Phase 7A, decision 0014): the runtime planner's inputs and every benchmark's
hardware record. Nothing here is assumed: what cannot be detected is reported as None, never filled with a default.

Host memory comes from the operating system (GlobalMemoryStatusEx on Windows, /proc/meminfo on Linux); the GPU from
nvidia-smi when it is installed (name, memory, driver, and the PCIe link: its current generation and width, which an
idle GPU lowers, and the maximum the GPU and the platform allow), else from PyTorch's device properties; storage from
the volume that holds a given path.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_NVIDIA_FIELDS = (
    "name",
    "memory.total",
    "memory.free",
    "driver_version",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
)


def host_memory() -> dict[str, int | None]:
    """Total and available physical memory, in bytes."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total_bytes": int(status.ullTotalPhys), "available_bytes": int(status.ullAvailPhys)}
        return {"total_bytes": None, "available_bytes": None}
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            fields[key] = int(value.split()[0]) * 1024
        return {"total_bytes": fields.get("MemTotal"), "available_bytes": fields.get("MemAvailable")}
    except (OSError, ValueError):
        return {"total_bytes": None, "available_bytes": None}


def _number(value: str) -> int | str | None:
    value = value.strip()
    if not value or value.startswith("[") or value.lower() in {"n/a", "not supported"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def gpus() -> list[dict[str, Any]]:
    """Every NVIDIA GPU nvidia-smi sees (memory in bytes), or PyTorch's view of the CUDA devices, or nothing."""
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            output = subprocess.run(
                [executable, f"--query-gpu={','.join(_NVIDIA_FIELDS)}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            devices = []
            for line in output.strip().splitlines():
                values = dict(zip(_NVIDIA_FIELDS, (_number(v) for v in line.split(","))))
                devices.append(
                    {
                        "name": values["name"],
                        "memory_total_bytes": values["memory.total"] * (1 << 20)
                        if isinstance(values["memory.total"], int) else None,
                        "memory_free_bytes": values["memory.free"] * (1 << 20)
                        if isinstance(values["memory.free"], int) else None,
                        "driver": values["driver_version"],
                        "pcie_generation_current": values["pcie.link.gen.current"],
                        "pcie_generation_max": values["pcie.link.gen.max"],
                        "pcie_width_current": values["pcie.link.width.current"],
                        "pcie_width_max": values["pcie.link.width.max"],
                        "source": "nvidia-smi",
                    }
                )
            if devices:
                return devices
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    try:
        import torch

        if torch.cuda.is_available():
            return [
                {
                    "name": torch.cuda.get_device_name(k),
                    "memory_total_bytes": torch.cuda.get_device_properties(k).total_memory,
                    "memory_free_bytes": None,
                    "driver": None,
                    "pcie_generation_current": None,
                    "pcie_generation_max": None,
                    "pcie_width_current": None,
                    "pcie_width_max": None,
                    "source": "torch",
                }
                for k in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return []


def storage(path: str | Path) -> dict[str, Any]:
    """The volume holding `path` (its nearest existing ancestor): total and free bytes."""
    target = Path(path).resolve()
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(target)
    return {"path": str(target), "anchor": target.anchor, "total_bytes": usage.total, "free_bytes": usage.free}


def detect(storage_path: str | Path | None = None) -> dict[str, Any]:
    """The machine, as detected now."""
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "host_memory": host_memory(),
        "gpus": gpus(),
        "storage": storage(storage_path) if storage_path is not None else None,
    }
