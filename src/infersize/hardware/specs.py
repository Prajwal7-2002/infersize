"""Hardware specs, VRAM limits, memory bandwidths, and peak compute profiles."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GPUProfile(BaseModel):
    """Physical hardware specifications of a GPU or Unified Memory Accelerator."""

    name: str = Field(description="Unique device identifier")
    display_name: str = Field(description="Human-readable device name")
    vram_gb: float = Field(description="Total addressable physical VRAM in Gigabytes")
    memory_bandwidth_gb_s: float = Field(description="Peak theoretical memory bandwidth in GB/s")
    fp16_tflops: float = Field(description="Peak FP16 / BF16 Tensor Core compute in TFLOPS")
    int8_tflops: float = Field(description="Peak INT8 Tensor Core compute in TOPS")
    int4_tflops: float = Field(description="Peak INT4 Tensor Core compute in TOPS")
    cuda_overhead_gb: float = Field(
        default=0.8, description="Standard CUDA driver & context memory reservation in GB"
    )

    @property
    def memory_bandwidth_tb_s(self) -> float:
        """Memory bandwidth in Terabytes per second."""
        return self.memory_bandwidth_gb_s / 1000.0

    @property
    def ridge_point_fp16(self) -> float:
        """Arithmetic intensity ridge point (FLOPs/Byte) for FP16 math."""
        # TFLOPS / (TB/s) = FLOPs / Byte
        return self.fp16_tflops / self.memory_bandwidth_tb_s

    @property
    def usable_vram_gb(self) -> float:
        """Usable physical VRAM after baseline driver/context overhead."""
        return max(0.0, self.vram_gb - self.cuda_overhead_gb)


# Standard Industry Hardware Presets
GPU_PRESETS: dict[str, GPUProfile] = {
    # Consumer & Edge GPUs
    "rtx3050": GPUProfile(
        name="rtx3050",
        display_name="NVIDIA GeForce RTX 3050 Laptop (6GB)",
        vram_gb=6.0,
        memory_bandwidth_gb_s=288.0,
        fp16_tflops=72.0,
        int8_tflops=144.0,
        int4_tflops=288.0,
        cuda_overhead_gb=0.6,
    ),
    "rtx3060": GPUProfile(
        name="rtx3060",
        display_name="NVIDIA GeForce RTX 3060 (12GB)",
        vram_gb=12.0,
        memory_bandwidth_gb_s=360.0,
        fp16_tflops=52.0,
        int8_tflops=104.0,
        int4_tflops=208.0,
        cuda_overhead_gb=0.7,
    ),
    "rtx4060": GPUProfile(
        name="rtx4060",
        display_name="NVIDIA GeForce RTX 4060 (8GB)",
        vram_gb=8.0,
        memory_bandwidth_gb_s=272.0,
        fp16_tflops=120.0,
        int8_tflops=240.0,
        int4_tflops=480.0,
        cuda_overhead_gb=0.7,
    ),
    "rtx4080": GPUProfile(
        name="rtx4080",
        display_name="NVIDIA GeForce RTX 4080 (16GB)",
        vram_gb=16.0,
        memory_bandwidth_gb_s=716.8,
        fp16_tflops=195.0,
        int8_tflops=390.0,
        int4_tflops=780.0,
        cuda_overhead_gb=0.8,
    ),
    "rtx4090": GPUProfile(
        name="rtx4090",
        display_name="NVIDIA GeForce RTX 4090 (24GB)",
        vram_gb=24.0,
        memory_bandwidth_gb_s=1008.0,
        fp16_tflops=330.0,
        int8_tflops=660.0,
        int4_tflops=1320.0,
        cuda_overhead_gb=0.9,
    ),
    # Datacenter GPUs
    "t4": GPUProfile(
        name="t4",
        display_name="NVIDIA Tesla T4 (16GB)",
        vram_gb=16.0,
        memory_bandwidth_gb_s=320.0,
        fp16_tflops=65.0,
        int8_tflops=130.0,
        int4_tflops=260.0,
        cuda_overhead_gb=0.7,
    ),
    "l4": GPUProfile(
        name="l4",
        display_name="NVIDIA L4 (24GB)",
        vram_gb=24.0,
        memory_bandwidth_gb_s=300.0,
        fp16_tflops=120.0,
        int8_tflops=240.0,
        int4_tflops=480.0,
        cuda_overhead_gb=0.8,
    ),
    "a100-40gb": GPUProfile(
        name="a100-40gb",
        display_name="NVIDIA A100 PCIe (40GB)",
        vram_gb=40.0,
        memory_bandwidth_gb_s=1555.0,
        fp16_tflops=312.0,
        int8_tflops=624.0,
        int4_tflops=1248.0,
        cuda_overhead_gb=1.1,
    ),
    "a100-80gb": GPUProfile(
        name="a100-80gb",
        display_name="NVIDIA A100 SXM4 (80GB HBM2e)",
        vram_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        fp16_tflops=312.0,
        int8_tflops=624.0,
        int4_tflops=1248.0,
        cuda_overhead_gb=1.2,
    ),
    "h100-80gb": GPUProfile(
        name="h100-80gb",
        display_name="NVIDIA H100 SXM5 (80GB HBM3)",
        vram_gb=80.0,
        memory_bandwidth_gb_s=3350.0,
        fp16_tflops=989.0,
        int8_tflops=1978.0,
        int4_tflops=3956.0,
        cuda_overhead_gb=1.4,
    ),
    "h200-141gb": GPUProfile(
        name="h200-141gb",
        display_name="NVIDIA H200 SXM (141GB HBM3e)",
        vram_gb=141.0,
        memory_bandwidth_gb_s=4800.0,
        fp16_tflops=989.0,
        int8_tflops=1978.0,
        int4_tflops=3956.0,
        cuda_overhead_gb=1.5,
    ),
    # Apple Silicon Unified Memory
    "m3-max-36gb": GPUProfile(
        name="m3-max-36gb",
        display_name="Apple M3 Max Unified Memory (36GB)",
        vram_gb=36.0,
        memory_bandwidth_gb_s=300.0,
        fp16_tflops=50.0,
        int8_tflops=100.0,
        int4_tflops=200.0,
        cuda_overhead_gb=1.0,
    ),
    "m3-max-128gb": GPUProfile(
        name="m3-max-128gb",
        display_name="Apple M3 Max Unified Memory (128GB)",
        vram_gb=128.0,
        memory_bandwidth_gb_s=400.0,
        fp16_tflops=100.0,
        int8_tflops=200.0,
        int4_tflops=400.0,
        cuda_overhead_gb=1.5,
    ),
}


def get_gpu_profile(gpu_name: str) -> GPUProfile:
    """Retrieve hardware profile specification by name or alias."""
    key = gpu_name.lower().strip().replace(" ", "").replace("_", "-")
    if key in GPU_PRESETS:
        return GPU_PRESETS[key]

    # Fuzzy match
    for name, spec in GPU_PRESETS.items():
        if key in name or name in key:
            return spec

    raise ValueError(
        f"Unknown hardware profile '{gpu_name}'. Available presets: {list(GPU_PRESETS.keys())}"
    )


def list_gpus() -> list[str]:
    """Return all available preset GPU profile names."""
    return list(GPU_PRESETS.keys())
