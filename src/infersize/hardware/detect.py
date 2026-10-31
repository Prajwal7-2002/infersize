"""Live local hardware silicon detection and zero-OOM fit verification."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Optional

from pydantic import BaseModel, Field

from infersize.analytics.vram import estimate_vram_usage
from infersize.hardware.specs import GPU_PRESETS, GPUProfile
from infersize.models.specs import ModelArch


class DetectedHardware(BaseModel):
    """Information about detected local accelerator or CPU compute environment."""

    device_type: str = Field(description="'cuda', 'apple_silicon', or 'cpu'")
    device_name: str = Field(description="Commercial name of the detected device")
    total_memory_gb: float = Field(description="Total addressable device memory in GB")
    free_memory_gb: float = Field(description="Currently free/available memory in GB")
    matched_gpu_id: Optional[str] = Field(
        default=None, description="Closest matched infersize GPU preset ID if applicable"
    )
    cuda_driver_version: Optional[str] = Field(default=None, description="NVIDIA Driver version if CUDA")
    cuda_compute_capability: Optional[str] = Field(default=None, description="Compute capability (e.g. 8.6, 8.9)")


class LocalFitResult(BaseModel):
    """Result of evaluating a model workload against local physical silicon."""

    model_name: str
    hardware: DetectedHardware
    precision: str
    kv_cache_dtype: str
    batch_size: int
    context_tokens: int

    required_memory_gb: float
    weights_memory_gb: float
    kv_cache_memory_gb: float
    activations_memory_gb: float

    headroom_total_gb: float = Field(description="Memory headroom against total physical memory")
    headroom_free_gb: float = Field(description="Memory headroom against currently free available memory")
    fits_total: bool = Field(description="Whether the workload fits inside total physical capacity")
    fits_currently: bool = Field(description="Whether the workload fits in currently free memory without evicting apps")
    verdict: str = Field(description="Clear human-readable engineering verdict")


def _get_system_ram_gb() -> tuple[float, float]:
    """Retrieves total and available system RAM across Windows, Linux, and macOS in GB."""
    total_gb = 16.0
    free_gb = 8.0

    system = platform.system()
    if system == "Windows":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_gb = round(stat.ullTotalPhys / (1024**3), 2)
                free_gb = round(stat.ullAvailPhys / (1024**3), 2)
        except Exception:
            pass
    elif system == "Linux":
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = int(parts[1].split()[0])  # in kB
            if "MemTotal" in meminfo:
                total_gb = round(meminfo["MemTotal"] / (1024**2), 2)
            if "MemAvailable" in meminfo:
                free_gb = round(meminfo["MemAvailable"] / (1024**2), 2)
            elif "MemFree" in meminfo:
                free_gb = round(meminfo["MemFree"] / (1024**2), 2)
        except Exception:
            pass
    elif system == "Darwin":  # macOS
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_gb = round(int(out) / (1024**3), 2)
            free_gb = round(total_gb * 0.5, 2)  # Estimation
        except Exception:
            pass

    return total_gb, free_gb


def detect_local_hardware() -> DetectedHardware:
    """Inspects local silicon environment and returns detected compute and memory specifications."""
    # 1. Try PyTorch CUDA if installed
    try:
        import torch

        if torch.cuda.is_available():
            dev_idx = 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_vram = round(props.total_memory / (1024**3), 2)

            free_vram = total_vram
            try:
                free_b, total_b = torch.cuda.mem_get_info(dev_idx)
                free_vram = round(free_b / (1024**3), 2)
                total_vram = round(total_b / (1024**3), 2)
            except Exception:
                pass

            cap = f"{props.major}.{props.minor}"
            matched_id = _match_gpu_preset(dev_name, total_vram)

            return DetectedHardware(
                device_type="cuda",
                device_name=dev_name,
                total_memory_gb=total_vram,
                free_memory_gb=free_vram,
                matched_gpu_id=matched_id,
                cuda_compute_capability=cap,
            )
    except Exception:
        pass

    # 2. Try nvidia-smi CLI if torch is not installed or lacks CUDA
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            ).decode().strip()

            lines = out.splitlines()
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 3:
                    name = parts[0]
                    total_mb = float(parts[1])
                    free_mb = float(parts[2])
                    driver = parts[3] if len(parts) > 3 else None
                    total_gb = round(total_mb / 1024.0, 2)
                    free_gb = round(free_mb / 1024.0, 2)
                    matched_id = _match_gpu_preset(name, total_gb)

                    return DetectedHardware(
                        device_type="cuda",
                        device_name=name,
                        total_memory_gb=total_gb,
                        free_memory_gb=free_gb,
                        matched_gpu_id=matched_id,
                        cuda_driver_version=driver,
                    )
        except Exception:
            pass

    # 3. Fallback to System CPU & RAM
    total_ram, free_ram = _get_system_ram_gb()
    cpu_model = platform.processor() or "Generic Host CPU"

    return DetectedHardware(
        device_type="cpu",
        device_name=f"{platform.machine()} CPU ({cpu_model})",
        total_memory_gb=total_ram,
        free_memory_gb=free_ram,
        matched_gpu_id=None,
    )


def _match_gpu_preset(name: str, total_gb: float) -> Optional[str]:
    """Finds closest matching preset GPU in GPU_PRESETS."""
    clean_name = name.lower()
    for gid in GPU_PRESETS.keys():
        if gid in clean_name or clean_name.replace(" ", "") in gid:
            return gid

    # Match based on common names
    if "3050" in clean_name:
        return "rtx3050"
    if "4090" in clean_name:
        return "rtx4090"
    if "4080" in clean_name:
        return "rtx4080"
    if "4070" in clean_name:
        return "rtx4070"
    if "4060" in clean_name:
        return "rtx4060"
    if "a100" in clean_name:
        return "a100-80gb" if total_gb > 50 else "a100-40gb"
    if "h100" in clean_name:
        return "h100-sxm"
    if "a10g" in clean_name or "a10" in clean_name:
        return "a10g"
    if "t4" in clean_name:
        return "t4"
    if "l40" in clean_name:
        return "l40s"
    if "l4" in clean_name:
        return "l4"

    return None


def verify_local_fit(
    arch: ModelArch,
    precision: str = "fp16",
    kv_cache_dtype: str = "fp16",
    batch_size: int = 1,
    context_tokens: int = 2048,
    hardware: Optional[DetectedHardware] = None,
) -> LocalFitResult:
    """Verifies whether the workload fits on the detected physical hardware without OOM."""
    hw = hardware or detect_local_hardware()

    # Construct temporary GPUProfile based on detected memory
    if hw.matched_gpu_id and hw.matched_gpu_id in GPU_PRESETS:
        target_gpu = GPU_PRESETS[hw.matched_gpu_id]
    else:
        target_gpu = GPUProfile(
            name=hw.device_name,
            vram_gb=hw.total_memory_gb,
            memory_bandwidth_gb_s=150.0,  # Generic baseline
            memory_bus_width_bits=128,
            boost_clock_mhz=1500,
            fp16_tflops=20.0,
            architecture="Detected Silicon",
        )

    vram_result = estimate_vram_usage(
        model=arch,
        gpu=target_gpu,
        precision=precision,
        kv_cache_dtype=kv_cache_dtype,
        batch_size=batch_size,
        context_tokens=context_tokens,
        tensor_parallel=1,
    )

    req_mem = vram_result.total_required_vram_gb
    headroom_total = round(hw.total_memory_gb - req_mem, 2)
    headroom_free = round(hw.free_memory_gb - req_mem, 2)

    fits_total = headroom_total >= 0.0
    fits_free = headroom_free >= 0.0

    if fits_free:
        verdict = f"✔ FITS WITH COMFORTABLE HEADROOM ({headroom_free} GB free remaining)"
    elif fits_total:
        deficit = round(-headroom_free, 2)
        verdict = f"⚠ FITS ON HARDWARE, BUT REQUIRES CLOSING APPS ({deficit} GB occupied by other processes)"
    else:
        deficit = round(-headroom_total, 2)
        verdict = f"✘ OUT-OF-MEMORY (OOM) DEFICIT OF {deficit} GB ON {hw.device_name.upper()}"

    return LocalFitResult(
        model_name=arch.name,
        hardware=hw,
        precision=precision,
        kv_cache_dtype=kv_cache_dtype,
        batch_size=batch_size,
        context_tokens=context_tokens,
        required_memory_gb=req_mem,
        weights_memory_gb=vram_result.weight_memory_gb,
        kv_cache_memory_gb=vram_result.kv_cache_memory_gb,
        activations_memory_gb=vram_result.activation_memory_gb,
        headroom_total_gb=headroom_total,
        headroom_free_gb=headroom_free,
        fits_total=fits_total,
        fits_currently=fits_free,
        verdict=verdict,
    )
