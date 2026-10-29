"""Command-line interface for infersize."""

from __future__ import annotations

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from infersize import __version__
from infersize.analytics.vram import (
    estimate_vram_usage,
    find_max_concurrent_batch_size,
    find_max_context_length,
)
from infersize.hardware.specs import GPU_PRESETS, get_gpu_profile
from infersize.models.specs import MODEL_PRESETS, get_model_spec

app = typer.Typer(
    name="infersize",
    help="⚡ Analytical Performance Modeling & VRAM Capacity Planner for LLM Serving.",
    add_completion=False,
)
console = Console()


def _format_memory_bar(res) -> str:
    """Generate an ASCII visual representation of GPU VRAM distribution."""
    width = 40
    total = res.gpu_total_vram_gb
    w_frac = min(1.0, res.weight_memory_gb / total)
    kv_frac = min(1.0 - w_frac, res.kv_cache_memory_gb / total)
    act_frac = min(1.0 - w_frac - kv_frac, res.activation_memory_gb / total)
    over_frac = min(1.0 - w_frac - kv_frac - act_frac, res.cuda_overhead_gb / total)

    w_chars = int(w_frac * width)
    kv_chars = int(kv_frac * width)
    act_chars = int(act_frac * width)
    over_chars = int(over_frac * width)
    free_chars = max(0, width - (w_chars + kv_chars + act_chars + over_chars))

    bar = (
        f"[cyan]{'█' * w_chars}[/]"
        f"[magenta]{'█' * kv_chars}[/]"
        f"[yellow]{'█' * act_chars}[/]"
        f"[blue]{'█' * over_chars}[/]"
        f"[dim]{'░' * free_chars}[/]"
    )
    return bar


@app.command()
def vram(
    model: str = typer.Option(
        "qwen2.5-7b", "--model", "-m", help="Model name or architecture preset"
    ),
    gpu: str = typer.Option("rtx3050", "--gpu", "-g", help="GPU hardware profile preset"),
    quant: str = typer.Option(
        "fp16", "--quant", "-q", help="Weight precision (fp16, int8, fp8, int4, bitnet)"
    ),
    kv_cache: str = typer.Option(
        "fp16", "--kv-cache", "-k", help="KV cache dtype (fp16, int8, fp8, int4)"
    ),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Concurrent serving batch size"),
    context: int = typer.Option(2048, "--context", "-c", help="Sequence context length in tokens"),
):
    """Predict exact physical VRAM footprint and capacity headroom before downloading or serving."""
    try:
        model_spec = get_model_spec(model)
        gpu_spec = get_gpu_profile(gpu)
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)

    res = estimate_vram_usage(
        model=model_spec,
        gpu=gpu_spec,
        precision=quant,
        kv_cache_dtype=kv_cache,
        batch_size=batch_size,
        context_tokens=context,
    )

    status_str = (
        "[bold green]✔ WORKLOAD FITS IN VRAM[/]"
        if res.fits_in_vram
        else "[bold red]✘ OUT OF MEMORY (OOM) RISK[/]"
    )

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Model:[/] {res.model_name} ({res.total_parameters / 1e9:.2f}B params) | "
            f"[bold magenta]GPU:[/] {res.gpu_name} ({res.gpu_total_vram_gb:.1f} GB)\n"
            f"[bold yellow]Precision:[/] Weights: {res.precision.upper()} | KV-Cache: {res.kv_cache_dtype.upper()} | "
            f"Batch: {res.batch_size} | Context: {res.context_tokens} tokens\n"
            f"[bold]Verdict:[/] {status_str} ({res.vram_utilization_pct:.1f}% utilized)",
            title="[bold yellow]⚡ LLM-PERFMODEL: VRAM CAPACITY ANALYSIS[/]",
            box=box.ROUNDED,
        )
    )

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("Component", style="bold")
    table.add_column("Memory (GB)", justify="right")
    table.add_column("Share (%)", justify="right")
    table.add_column("Key Physical Insight")

    total = res.gpu_total_vram_gb
    table.add_row(
        "[cyan]Model Weights[/]",
        f"{res.weight_memory_gb:.2f} GB",
        f"{(res.weight_memory_gb / total) * 100:.1f}%",
        f"Compressed {res.weight_compression_ratio:.1f}x from FP16 baseline",
    )
    table.add_row(
        "[magenta]KV Cache[/]",
        f"{res.kv_cache_memory_gb:.2f} GB",
        f"{(res.kv_cache_memory_gb / total) * 100:.1f}%",
        f"{res.kv_bytes_per_token:,} bytes/token (B={res.batch_size}, S={res.context_tokens})",
    )
    table.add_row(
        "[yellow]Activations[/]",
        f"{res.activation_memory_gb:.2f} GB",
        f"{(res.activation_memory_gb / total) * 100:.1f}%",
        "FlashAttention linear peak memory buffer",
    )
    table.add_row(
        "[blue]CUDA Context[/]",
        f"{res.cuda_overhead_gb:.2f} GB",
        f"{(res.cuda_overhead_gb / total) * 100:.1f}%",
        "Driver runtime reservation and cuBLAS workspace",
    )
    table.add_section()
    table.add_row(
        "[bold white]TOTAL REQUIRED[/]",
        f"[bold white]{res.total_required_vram_gb:.2f} GB[/]",
        f"[bold white]{res.vram_utilization_pct:.1f}%[/]",
        "[bold white]Total physical VRAM demand[/]",
    )

    headroom_style = "green" if res.fits_in_vram else "red"
    headroom_text = (
        f"+{res.free_vram_headroom_gb:.2f} GB free"
        if res.fits_in_vram
        else f"-{abs(res.free_vram_headroom_gb):.2f} GB deficit"
    )
    table.add_row(
        f"[bold {headroom_style}]HEADROOM[/]",
        f"[bold {headroom_style}]{headroom_text}[/]",
        f"[{headroom_style}]{100.0 - res.vram_utilization_pct:.1f}%[/]",
        f"[{headroom_style}]VRAM remaining for additional batch streams[/]",
    )

    console.print(table)

    bar = _format_memory_bar(res)
    console.print(f"[bold]Memory Allocation:[/] [{bar}]")
    console.print(
        "Legend: [cyan]■[/] Weights  [magenta]■[/] KV-Cache  [yellow]■[/] Activations  [blue]■[/] Driver  [dim]░[/] Free\n"
    )


@app.command()
def max_batch(
    model: str = typer.Option("qwen2.5-7b", "--model", "-m"),
    gpu: str = typer.Option("rtx3050", "--gpu", "-g"),
    quant: str = typer.Option("int4", "--quant", "-q"),
    context: int = typer.Option(2048, "--context", "-c"),
    kv_cache: str = typer.Option("fp16", "--kv-cache", "-k"),
):
    """Calculate the maximum concurrent request batch size before running out of VRAM."""
    model_spec = get_model_spec(model)
    gpu_spec = get_gpu_profile(gpu)

    limit = find_max_concurrent_batch_size(
        model=model_spec,
        gpu=gpu_spec,
        precision=quant,
        context_tokens=context,
        kv_cache_dtype=kv_cache,
    )

    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/] {model_spec.name} ({quant.upper()}) | [bold]GPU:[/] {gpu_spec.display_name}\n"
            f"[bold]Target Context Window:[/] {context:,} tokens per stream\n\n"
            f"🎯 [bold green]Maximum Safe Concurrent Streams:[/] [bold white]{limit}[/] requests\n"
            f"[dim](Exceeding this batch limit will trigger Out-Of-Memory preemption or swapping).[/]",
            title="[bold yellow]🚀 CONCURRENT BATCH SIZING[/]",
            box=box.ROUNDED,
        )
    )
    console.print()


@app.command()
def max_context(
    model: str = typer.Option("qwen2.5-7b", "--model", "-m"),
    gpu: str = typer.Option("rtx3050", "--gpu", "-g"),
    quant: str = typer.Option("int4", "--quant", "-q"),
    batch_size: int = typer.Option(1, "--batch-size", "-b"),
    kv_cache: str = typer.Option("fp16", "--kv-cache", "-k"),
):
    """Calculate the maximum context window (prompt + decode) before running out of VRAM."""
    model_spec = get_model_spec(model)
    gpu_spec = get_gpu_profile(gpu)

    limit = find_max_context_length(
        model=model_spec,
        gpu=gpu_spec,
        precision=quant,
        batch_size=batch_size,
        kv_cache_dtype=kv_cache,
    )

    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/] {model_spec.name} ({quant.upper()}) | [bold]GPU:[/] {gpu_spec.display_name}\n"
            f"[bold]Concurrent Streams:[/] {batch_size}\n\n"
            f"🎯 [bold green]Maximum Supported Context Length:[/] [bold white]{limit:,}[/] tokens\n"
            f"[dim](Longer prompts will cause GPU memory allocation failure).[/]",
            title="[bold yellow]📏 CONTEXT LENGTH CAPACITY[/]",
            box=box.ROUNDED,
        )
    )
    console.print()


@app.command()
def models():
    """List all available open-source model architecture presets."""
    table = Table(title="Available Model Presets", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Model Identifier", style="bold green")
    table.add_column("Layers", justify="right")
    table.add_column("Hidden Dim", justify="right")
    table.add_column("KV Heads", justify="right")
    table.add_column("GQA Ratio", justify="right")
    table.add_column("Total Parameters", justify="right")

    for name, spec in MODEL_PRESETS.items():
        params = spec.calculate_total_parameters()
        table.add_row(
            name,
            str(spec.num_layers),
            str(spec.hidden_size),
            str(spec.num_key_value_heads),
            f"{spec.gqa_ratio:.1f}x",
            f"{params / 1e9:.2f}B",
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def gpus():
    """List all available GPU hardware profiles and Silicon roofline metrics."""
    table = Table(title="Available GPU Profiles", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Identifier", style="bold green")
    table.add_column("Device Name")
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("Bandwidth (GB/s)", justify="right")
    table.add_column("FP16 TFLOPS", justify="right")
    table.add_column("Ridge Point (I*)", justify="right")

    for name, spec in GPU_PRESETS.items():
        table.add_row(
            name,
            spec.display_name,
            f"{spec.vram_gb:.1f} GB",
            f"{spec.memory_bandwidth_gb_s:.0f} GB/s",
            f"{spec.fp16_tflops:.0f}",
            f"{spec.ridge_point_fp16:.1f} FLOPs/B",
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def version():
    """Display infersize package version."""
    console.print(f"[bold cyan]infersize[/] version [bold green]{__version__}[/]")


if __name__ == "__main__":
    app()
