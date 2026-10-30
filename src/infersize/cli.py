"""Command-line interface for infersize."""

from __future__ import annotations

import sys

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure UTF-8 encoding on Windows console to prevent charmap UnicodeEncodeError
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from infersize import __version__
from infersize.analytics.comparison import (
    compare_gpus_for_workload,
)
from infersize.analytics.vram import (
    estimate_vram_usage,
    find_max_concurrent_batch_size,
    find_max_context_length,
)
from infersize.hardware.specs import GPU_PRESETS, get_gpu_profile
from infersize.models.specs import MODEL_PRESETS, get_model_spec

app = typer.Typer(
    name="infersize",
    help="Analytical Performance Modeling & VRAM Capacity Planner for LLM Serving.",
    add_completion=False,
)
console = Console(legacy_windows=False)


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
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallelism degree (TP)"),
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
        tensor_parallel=tensor_parallel,
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
            f"Batch: {res.batch_size} | Context: {res.context_tokens} tokens | TP: {res.tensor_parallel}\n"
            f"[bold]Verdict:[/] {status_str} ({res.vram_utilization_pct:.1f}% utilized)",
            title="[bold yellow]⚡ INFERSIZE: VRAM CAPACITY ANALYSIS[/]",
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
def compare(
    model: str = typer.Option(
        "qwen2.5-7b", "--model", "-m", help="Model name or architecture preset"
    ),
    context: int = typer.Option(2048, "--context", "-c", help="Sequence context length in tokens"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Concurrent serving batch size"),
    quant: str = typer.Option(
        "fp16", "--quant", "-q", help="Weight precision (fp16, int8, fp8, int4, bitnet)"
    ),
    kv_cache: str = typer.Option(
        "fp16", "--kv-cache", "-k", help="KV cache dtype (fp16, int8, fp8, int4)"
    ),
    gpus: str = typer.Option(
        "",
        "--gpus",
        "-g",
        help="Comma-separated GPU IDs to compare (e.g. 'rtx4090,a100-80gb,l4'), or empty for standard set",
    ),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallelism degree (TP)"),
    json_output: bool = typer.Option(
        False, "--json", help="Output comparison result as JSON for automated pipelines and CI"
    ),
):
    """Compare workload capacity, VRAM fit, headroom, and batch limits across multiple GPUs."""
    try:
        model_spec = get_model_spec(model)
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)

    candidate_gpus = None
    if gpus.strip():
        gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
        try:
            candidate_gpus = [get_gpu_profile(gid) for gid in gpu_ids]
        except ValueError as e:
            console.print(f"[bold red]Error:[/] {e}")
            raise typer.Exit(code=1)

    report = compare_gpus_for_workload(
        model=model_spec,
        candidate_gpus=candidate_gpus,
        precision=quant,
        kv_cache_dtype=kv_cache,
        batch_size=batch_size,
        context_tokens=context,
        tensor_parallel=tensor_parallel,
    )

    if json_output:
        console.print_json(report.model_dump_json(indent=2))
        return

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Model:[/] {report.model_name} | "
            f"[bold yellow]Weights:[/] {report.precision.upper()} | "
            f"[bold magenta]KV-Cache:[/] {report.kv_cache_dtype.upper()} | "
            f"[bold white]Batch:[/] {report.batch_size} | "
            f"[bold white]Context:[/] {report.context_tokens:,} tokens | "
            f"[bold green]TP:[/] {report.tensor_parallel}",
            title="[bold yellow]⚡ INFERSIZE: MULTI-GPU HARDWARE COMPARISON MATRIX[/]",
            box=box.ROUNDED,
        )
    )

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("GPU Profile", style="bold")
    table.add_column("VRAM", justify="right")
    table.add_column("Required", justify="right")
    table.add_column("Util (%)", justify="right")
    table.add_column("Headroom", justify="right")
    table.add_column("Max Batch", justify="right")
    table.add_column("Bandwidth", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Sizing Verdict & Recommendation")

    status_styles = {
        "OPTIMAL": ("[bold green]⭐ OPTIMAL[/]", "green"),
        "FITS": ("[green]✔ FITS[/]", "green"),
        "TIGHT": ("[bold yellow]⚠ TIGHT[/]", "yellow"),
        "UNDERUTILIZED": ("[cyan]ℹ OVERKILL[/]", "cyan"),
        "OOM": ("[bold red]✘ OOM[/]", "red"),
    }

    for item in report.items:
        badge, color = status_styles.get(item.status, (item.status, "white"))
        sign = "+" if item.headroom_gb >= 0 else ""
        headroom_str = f"[{color}]{sign}{item.headroom_gb:.1f} GB[/]"

        table.add_row(
            f"[{color}]{item.gpu_id}[/]",
            f"{item.vram_total_gb:.1f} GB",
            f"{item.required_vram_gb:.1f} GB",
            f"{item.utilization_pct:.1f}%",
            headroom_str,
            f"{item.max_concurrent_batch} reqs" if item.fits else "0",
            f"{item.memory_bandwidth_gb_s:.0f} GB/s",
            badge,
            item.recommendation_note,
        )

    console.print(table)

    # Recommendations panel
    rec_lines = []
    if report.recommended_optimal_gpu:
        rec_lines.append(
            f"⭐ [bold green]Recommended (Right-Sized / Cost-Effective):[/] [bold white]{report.recommended_optimal_gpu}[/] "
            "- Safest fit with minimal idle VRAM waste and lowest hardware rental cost."
        )
    if report.recommended_performance_gpu:
        rec_lines.append(
            f"🚀 [bold cyan]Recommended (Peak Throughput):[/] [bold white]{report.recommended_performance_gpu}[/] "
            "- Highest memory bandwidth among fitting GPUs; delivers lowest time-per-output-token."
        )

    if rec_lines:
        console.print(
            Panel(
                "\n".join(rec_lines),
                title="[bold green]🎯 PROCUREMENT & SIZING RECOMMENDATION[/]",
                box=box.ROUNDED,
            )
        )
    console.print()


@app.command()
def max_batch(
    model: str = typer.Option("qwen2.5-7b", "--model", "-m"),
    gpu: str = typer.Option("rtx3050", "--gpu", "-g"),
    quant: str = typer.Option("int4", "--quant", "-q"),
    context: int = typer.Option(2048, "--context", "-c"),
    kv_cache: str = typer.Option("fp16", "--kv-cache", "-k"),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallelism degree"),
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
        tensor_parallel=tensor_parallel,
    )

    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/] {model_spec.name} ({quant.upper()}) | [bold]GPU:[/] {gpu_spec.display_name}\n"
            f"[bold]Target Context Window:[/] {context:,} tokens per stream | [bold]TP:[/] {tensor_parallel}\n\n"
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
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallelism degree"),
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
        tensor_parallel=tensor_parallel,
    )

    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/] {model_spec.name} ({quant.upper()}) | [bold]GPU:[/] {gpu_spec.display_name}\n"
            f"[bold]Concurrent Streams:[/] {batch_size} | [bold]TP:[/] {tensor_parallel}\n\n"
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
