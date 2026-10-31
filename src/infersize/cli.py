"""Command-line interface for infersize."""

from __future__ import annotations

import sys
from pathlib import Path

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

from rich.markdown import Markdown

from infersize import __version__
from infersize.analytics.comparison import (
    compare_gpus_for_workload,
)
from infersize.analytics.cost import (
    compare_cloud_costs_for_workload,
    estimate_serving_cost,
)
from infersize.analytics.report import generate_architecture_report
from infersize.analytics.speculative import estimate_speculative_decoding
from infersize.analytics.vram import (
    estimate_vram_usage,
    find_max_concurrent_batch_size,
    find_max_context_length,
)
from infersize.hardware.detect import detect_local_hardware, verify_local_fit
from infersize.hardware.pricing import CLOUD_GPU_PRICING
from infersize.hardware.specs import GPU_PRESETS, get_gpu_profile
from infersize.models.loader import resolve_model_arch
from infersize.models.specs import MODEL_PRESETS, ModelArch

app = typer.Typer(
    name="infersize",
    help="Analytical Performance Modeling & VRAM Capacity Planner for LLM Serving.",
    add_completion=False,
)
console = Console(legacy_windows=False)


def _resolve_model(
    model: str,
    hf: bool = False,
    config: str | None = None,
    hf_token: str | None = None,
) -> ModelArch:
    """Resolve model architecture from preset, local config file, or Hugging Face hub."""
    try:
        return resolve_model_arch(
            model_identifier=model,
            hf_token=hf_token,
            is_hf_flag=hf,
            config_path=config,
        )
    except Exception as e:
        console.print(f"[bold red]Error resolving model:[/] {e}")
        raise typer.Exit(code=1)


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
    hf: bool = typer.Option(
        False, "--hf", help="Fetch model architecture directly from Hugging Face Hub"
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to local Hugging Face config.json file"
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", help="Hugging Face API token for gated or private repositories"
    ),
):
    """Predict exact physical VRAM footprint and capacity headroom before downloading or serving."""
    model_spec = _resolve_model(model=model, hf=hf, config=config, hf_token=hf_token)
    try:
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
    hf: bool = typer.Option(
        False, "--hf", help="Fetch model architecture directly from Hugging Face Hub"
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to local Hugging Face config.json file"
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", help="Hugging Face API token for gated or private repositories"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output comparison result as JSON for automated pipelines and CI"
    ),
):
    """Compare workload capacity, VRAM fit, headroom, and batch limits across multiple GPUs."""
    model_spec = _resolve_model(model=model, hf=hf, config=config, hf_token=hf_token)

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
def cost(
    model: str = typer.Option(
        "qwen2.5-7b", "--model", "-m", help="Model name or architecture preset"
    ),
    gpu: str = typer.Option("rtx4090", "--gpu", "-g", help="GPU hardware profile preset"),
    quant: str = typer.Option(
        "fp16", "--quant", "-q", help="Weight precision (fp16, int8, fp8, int4, bitnet)"
    ),
    kv_cache: str = typer.Option(
        "fp16", "--kv-cache", "-k", help="KV cache dtype (fp16, int8, fp8, int4)"
    ),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Concurrent serving batch size"),
    context: int = typer.Option(2048, "--context", "-c", help="Sequence context length in tokens"),
    provider: str = typer.Option(
        "runpod", "--provider", "-p", help="Cloud provider preset (runpod, lambda, aws, gcp)"
    ),
    rate: float | None = typer.Option(
        None, "--rate", "-r", help="Custom hourly rental rate override in USD per GPU"
    ),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallelism degree (TP)"),
    compare_all: bool = typer.Option(
        False, "--compare-all", help="Compare cost and throughput across standard candidate GPUs"
    ),
    compare_clouds: bool = typer.Option(
        False,
        "--compare-clouds",
        help="Compare hourly cost and $/1M tokens across all cloud providers",
    ),
    hf: bool = typer.Option(
        False, "--hf", help="Fetch model architecture directly from Hugging Face Hub"
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to local Hugging Face config.json file"
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", help="Hugging Face API token for gated or private repositories"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output result as JSON for automated pipelines and CI"
    ),
):
    """Predict memory-bandwidth decode throughput, TPOT latency, and $/1M tokens serving cost."""
    model_spec = _resolve_model(model=model, hf=hf, config=config, hf_token=hf_token)
    try:
        gpu_spec = get_gpu_profile(gpu)
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)

    if compare_all:
        multi_report = compare_cloud_costs_for_workload(
            model=model_spec,
            precision=quant,
            kv_cache_dtype=kv_cache,
            batch_size=batch_size,
            context_tokens=context,
            provider=provider,
            tensor_parallel=tensor_parallel,
        )
        if json_output:
            console.print_json(multi_report.model_dump_json(indent=2))
            return

        console.print()
        console.print(
            Panel(
                f"[bold cyan]Model:[/] {multi_report.model_name} | "
                f"[bold yellow]Precision:[/] {multi_report.precision.upper()} | "
                f"[bold white]Batch:[/] {multi_report.batch_size} | "
                f"[bold white]Context:[/] {multi_report.context_tokens:,} tokens | "
                f"[bold magenta]Provider:[/] {provider.upper()}",
                title="[bold yellow]⚡ INFERSIZE: CLOUD HARDWARE COST & THROUGHPUT MATRIX[/]",
                box=box.ROUNDED,
            )
        )
        t = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        t.add_column("GPU Profile", style="bold")
        t.add_column("Rate ($/hr)", justify="right")
        t.add_column("Throughput (tok/s)", justify="right")
        t.add_column("TPOT (ms)", justify="right")
        t.add_column("Capacity (tok/hr)", justify="right")
        t.add_column("Cost / 1M Tokens", justify="right", style="bold green")
        t.add_column("Status", justify="center")

        for r in multi_report.results:
            if not r.fits_in_vram:
                t.add_row(
                    r.gpu_id,
                    f"${r.total_cluster_hourly_usd:.2f}",
                    "--",
                    "--",
                    "--",
                    "--",
                    "[bold red]✘ OOM[/]",
                )
            else:
                t.add_row(
                    r.gpu_id,
                    f"${r.total_cluster_hourly_usd:.2f}",
                    f"{r.aggregate_tokens_per_sec:.1f}",
                    f"{r.time_per_output_token_ms:.1f} ms",
                    f"{r.tokens_per_hour:,.0f}",
                    f"${r.cost_per_million_tokens_usd:.4f}",
                    "[green]✔ FITS[/]",
                )
        console.print(t)
        if multi_report.lowest_cost_setup:
            console.print(
                Panel(
                    f"🏆 [bold green]Lowest Unit Cost Setup:[/] [bold white]{multi_report.lowest_cost_setup.gpu_id}[/] at "
                    f"[bold green]${multi_report.lowest_cost_setup.cost_per_million_tokens_usd:.4f} / 1M tokens[/] "
                    f"(${multi_report.lowest_cost_setup.total_cluster_hourly_usd:.2f}/hr)",
                    box=box.ROUNDED,
                )
            )
        console.print()
        return

    res = estimate_serving_cost(
        model=model_spec,
        gpu=gpu_spec,
        precision=quant,
        kv_cache_dtype=kv_cache,
        batch_size=batch_size,
        context_tokens=context,
        tensor_parallel=tensor_parallel,
        provider=provider,
        custom_hourly_rate=rate,
    )

    if json_output:
        console.print_json(res.model_dump_json(indent=2))
        return

    status_str = (
        "[bold green]✔ FIT - CAPACITY NORMAL[/]"
        if res.fits_in_vram
        else f"[bold red]✘ OOM DEFICIT (-{res.oom_deficit_gb:.1f} GB)[/]"
    )

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Model:[/] {res.model_name} | [bold magenta]GPU:[/] {res.gpu_name} (TP={res.tensor_parallel})\n"
            f"[bold yellow]Provider:[/] {res.provider.upper()} (${res.total_cluster_hourly_usd:.2f}/hr cluster) | "
            f"[bold white]Batch:[/] {res.batch_size} | [bold white]Context:[/] {res.context_tokens:,} tokens\n"
            f"[bold]Status:[/] {status_str}",
            title="[bold yellow]⚡ INFERSIZE: CLOUD SERVING ROOFLINE & TCO ANALYSIS[/]",
            box=box.ROUNDED,
        )
    )

    if not res.fits_in_vram:
        console.print(
            f"[bold red]Error:[/] Workload exceeds physical VRAM by {res.oom_deficit_gb:.1f} GB. "
            "Cannot compute throughput without quantization or tensor parallelism.\n"
        )
        return

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Engineering Insight")

    table.add_row(
        "[cyan]Memory Bandwidth[/]",
        f"{res.memory_bandwidth_gb_s:.0f} GB/s (Eff: {res.effective_bandwidth_gb_s:.1f} GB/s)",
        f"Assumes {res.bus_efficiency * 100:.0f}% memory bus saturation",
    )
    table.add_row(
        "[cyan]DRAM Read / Step[/]",
        f"{res.total_dram_bytes_per_step / 1e9:.2f} GB",
        f"Weights: {res.weight_bytes_per_step / 1e9:.2f} GB, KV-Cache: {res.kv_bytes_per_step / 1e9:.2f} GB",
    )
    table.add_row(
        "[yellow]Time Per Output Token (TPOT)[/]",
        f"{res.time_per_output_token_ms:.1f} ms",
        f"{res.tokens_per_sec_per_user:.1f} tok/s per user stream",
    )
    table.add_row(
        "[yellow]Aggregate Throughput[/]",
        f"{res.aggregate_tokens_per_sec:.1f} tok/s",
        f"Total generation across B={res.batch_size} concurrent streams",
    )
    table.add_row(
        "[magenta]Hourly Token Capacity[/]",
        f"{res.tokens_per_hour:,.0f} tokens/hr",
        "Maximum tokens generated per running hour",
    )
    table.add_section()
    table.add_row(
        "[bold white]Cluster Rental Rate[/]",
        f"[bold white]${res.total_cluster_hourly_usd:.2f} / hr[/]",
        f"${res.hourly_rate_usd:.2f}/hr per GPU x {res.tensor_parallel} GPUs",
    )
    table.add_row(
        "[bold green]COST PER 1M TOKENS[/]",
        f"[bold green]${res.cost_per_million_tokens_usd:.4f}[/]",
        "Direct operational unit cost per million generated tokens",
    )
    table.add_row(
        "[bold blue]24/7 Dedicated Monthly[/]",
        f"[bold blue]${res.monthly_dedicated_cost_usd:.2f} / mo[/]",
        "Baseline idle cluster rental (730 continuous hours)",
    )

    console.print(table)

    if compare_clouds:
        gpu_id_key = gpu_spec.name
        cloud_table = Table(
            title=f"Cloud Provider Price Comparison for {gpu_spec.display_name}",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        cloud_table.add_column("Provider", style="bold")
        cloud_table.add_column("Hourly / GPU", justify="right")
        cloud_table.add_column("Total Cluster / Hr", justify="right")
        cloud_table.add_column("Cost / 1M Tokens", justify="right", style="bold green")
        cloud_table.add_column("Monthly (730h)", justify="right")

        for p_name in ["runpod", "lambda", "aws", "gcp"]:
            p_rate = CLOUD_GPU_PRICING.get(p_name, {}).get(gpu_id_key)
            if p_rate is not None:
                p_cluster_hr = p_rate * res.tensor_parallel
                p_cost_1m = (
                    (p_cluster_hr / res.tokens_per_hour) * 1e6 if res.tokens_per_hour > 0 else 0.0
                )
                cloud_table.add_row(
                    p_name.upper(),
                    f"${p_rate:.2f}",
                    f"${p_cluster_hr:.2f}",
                    f"${p_cost_1m:.4f}",
                    f"${p_cluster_hr * 730:.2f}",
                )
        console.print(cloud_table)

    console.print()


@app.command()
def report(
    model: str = typer.Option(
        "qwen2.5-7b", "--model", "-m", help="Model name or architecture preset"
    ),
    gpu: str = typer.Option("rtx4090", "--gpu", "-g", help="Target GPU hardware profile preset"),
    quant: str = typer.Option(
        "fp16", "--quant", "-q", help="Weight precision (fp16, int8, fp8, int4, bitnet)"
    ),
    kv_cache: str = typer.Option(
        "fp16", "--kv-cache", "-k", help="KV cache dtype (fp16, int8, fp8, int4)"
    ),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Concurrent serving batch size"),
    context: int = typer.Option(2048, "--context", "-c", help="Sequence context length in tokens"),
    provider: str = typer.Option(
        "runpod", "--provider", "-p", help="Cloud provider preset (runpod, lambda, aws, gcp)"
    ),
    rate: float | None = typer.Option(
        None, "--rate", "-r", help="Custom hourly rental rate override in USD per GPU"
    ),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallelism degree (TP)"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="File path to save the generated Markdown RFC report"
    ),
    hf: bool = typer.Option(
        False, "--hf", help="Fetch model architecture directly from Hugging Face Hub"
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to local Hugging Face config.json file"
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", help="Hugging Face API token for gated or private repositories"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output complete report specification as JSON"
    ),
):
    """Generate and export a publication-ready Architecture RFC and Sizing Specification."""
    model_spec = _resolve_model(model=model, hf=hf, config=config, hf_token=hf_token)
    try:
        gpu_spec = get_gpu_profile(gpu)
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)

    rep = generate_architecture_report(
        model=model_spec,
        gpu=gpu_spec,
        precision=quant,
        kv_cache_dtype=kv_cache,
        batch_size=batch_size,
        context_tokens=context,
        tensor_parallel=tensor_parallel,
        provider=provider,
        custom_hourly_rate=rate,
    )

    if json_output:
        console.print_json(rep.model_dump_json(indent=2))
        return

    md_content = rep.to_markdown()

    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                f.write(md_content)
            console.print()
            console.print(
                Panel(
                    f"✔ [bold green]Architecture RFC Report exported successfully![/]\n\n"
                    f"📁 [bold white]Destination:[/] {output}\n"
                    f"📊 [bold cyan]Model:[/] {rep.model_name} | [bold magenta]GPU:[/] {rep.gpu_name} (TP={rep.tensor_parallel})\n"
                    f"💰 [bold yellow]Unit Economics:[/] ${rep.cost_estimate.cost_per_million_tokens_usd:.4f} / 1M tokens\n\n"
                    f"[dim]Ready to drop directly into GitHub PRs, Engineering RFCs, or infrastructure tickets.[/]",
                    title="[bold green]⚡ INFERSIZE: SPECIFICATION EXPORT COMPLETE[/]",
                    box=box.ROUNDED,
                )
            )
            console.print()
        except OSError as e:
            console.print(f"[bold red]Error writing output file:[/] {e}")
            raise typer.Exit(code=1)
    else:
        # Render markdown to terminal
        console.print()
        console.print(Markdown(md_content))
        console.print()
        console.print(
            "[dim]Tip: Use [bold white]--output deployment_rfc.md[/] to export this report to a Markdown file.[/]\n"
        )


@app.command()
def max_batch(
    model: str = typer.Option("qwen2.5-7b", "--model", "-m"),
    gpu: str = typer.Option("rtx3050", "--gpu", "-g"),
    quant: str = typer.Option("int4", "--quant", "-q"),
    context: int = typer.Option(2048, "--context", "-c"),
    kv_cache: str = typer.Option("fp16", "--kv-cache", "-k"),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallelism degree"),
    hf: bool = typer.Option(
        False, "--hf", help="Fetch model architecture directly from Hugging Face Hub"
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to local Hugging Face config.json file"
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", help="Hugging Face API token for gated or private repositories"
    ),
):
    """Calculate the maximum concurrent request batch size before running out of VRAM."""
    model_spec = _resolve_model(model=model, hf=hf, config=config, hf_token=hf_token)
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
    hf: bool = typer.Option(
        False, "--hf", help="Fetch model architecture directly from Hugging Face Hub"
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to local Hugging Face config.json file"
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", help="Hugging Face API token for gated or private repositories"
    ),
):
    """Calculate the maximum context window (prompt + decode) before running out of VRAM."""
    model_spec = _resolve_model(model=model, hf=hf, config=config, hf_token=hf_token)
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


@app.command()
def spec(
    target: str = typer.Option(
        "qwen2.5-32b", "--target", "-t", help="Target model preset name or HF repo ID"
    ),
    draft: str = typer.Option(
        "qwen2.5-0.5b", "--draft", "-d", help="Draft model preset name or HF repo ID"
    ),
    gpu: str = typer.Option("a100-80gb", "--gpu", "-g", help="Target GPU hardware profile"),
    target_quant: str = typer.Option(
        "fp16", "--target-quant", help="Quantization for target model weights"
    ),
    draft_quant: str = typer.Option(
        "fp16", "--draft-quant", help="Quantization for draft model weights"
    ),
    kv_cache: str = typer.Option("fp16", "--kv-cache", help="KV-cache data type"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Concurrent request streams"),
    context: int = typer.Option(2048, "--context", "-c", help="Sequence length (prompt + output)"),
    lookahead: int = typer.Option(4, "--lookahead", "-k", help="Speculative proposal length K"),
    acceptance: float = typer.Option(
        0.70, "--acceptance", "-a", help="Empirical token verification acceptance rate (alpha)"
    ),
    tp: int = typer.Option(1, "--tp", help="Tensor Parallelism degree across GPUs"),
):
    """Analytically model Speculative Decoding joint VRAM and Leviathan speedup."""
    target_arch = _resolve_model(target)
    draft_arch = _resolve_model(draft)
    gpu_profile = get_gpu_profile(gpu)

    res = estimate_speculative_decoding(
        target_arch=target_arch,
        draft_arch=draft_arch,
        gpu=gpu_profile,
        target_precision=target_quant,
        draft_precision=draft_quant,
        kv_cache_dtype=kv_cache,
        batch_size=batch_size,
        context_tokens=context,
        lookahead_k=lookahead,
        acceptance_rate=acceptance,
        tensor_parallel=tp,
    )

    console.print()
    status_style = "bold green" if res.fits_in_vram else "bold red"
    status_icon = "✔ FIT" if res.fits_in_vram else "✘ OOM"
    console.print(
        Panel(
            f"[bold cyan]Target:[/] {res.target_model_name} ({target_quant.upper()}) | "
            f"[bold cyan]Draft:[/] {res.draft_model_name} ({draft_quant.upper()}) | "
            f"[bold cyan]GPU:[/] {res.gpu_name} (TP={tp})\n"
            f"[bold cyan]Workload:[/] Batch: {res.batch_size} | Context: {res.context_tokens:,} tokens | "
            f"K: {res.lookahead_k} | Alpha: {res.acceptance_rate:.2f}\n"
            f"[bold cyan]Status:[/] [{status_style}]{status_icon} - {'CAPACITY NORMAL' if res.fits_in_vram else 'OUT-OF-MEMORY'}[/]",
            title="[bold yellow]⚡ INFERSIZE: SPECULATIVE DECODING CAPACITY & ROOFLINE SIZER[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Speculative Sizing Metric", style="bold")
    table.add_column("Value", justify="right", style="bold green")
    table.add_column("Engineering Analysis & Verification Physics")

    table.add_row(
        "Target Model VRAM",
        f"{res.target_vram_gb:.2f} GB",
        f"Weights + KV-Cache ({kv_cache}) + Activation buffers",
    )
    table.add_row(
        "Draft Model VRAM",
        f"{res.draft_vram_gb:.2f} GB",
        f"Weights + KV-Cache ({kv_cache}) + Activation buffers",
    )
    table.add_row(
        "Total Joint VRAM",
        f"{res.total_vram_gb:.2f} GB",
        "Target + Draft + CUDA context held concurrently in VRAM",
    )
    table.add_row(
        "GPU Total VRAM",
        f"{res.gpu_total_vram_gb:.1f} GB",
        f"Physical VRAM limit of {res.gpu_name}",
    )
    headroom_style = "green" if res.headroom_gb >= 0 else "red"
    table.add_row(
        "VRAM Headroom",
        f"[{headroom_style}]{res.headroom_gb:+.2f} GB[/]",
        "Safe operational headroom"
        if res.headroom_gb >= 0
        else "VRAM deficit: Will crash with CUDA OOM",
    )
    table.add_section()
    table.add_row(
        "Draft Step Latency",
        f"{res.draft_step_latency_ms:.1f} ms",
        "Time to generate 1 draft token at memory bandwidth limits",
    )
    table.add_row(
        "Target Verify Pass",
        f"{res.target_verify_latency_ms:.1f} ms",
        f"Target parallel forward pass verifying all {res.lookahead_k} candidates",
    )
    table.add_row(
        "Target Baseline TPOT",
        f"{res.baseline_autoregressive_latency_ms:.1f} ms",
        f"Standard autoregressive decode without speculation ({res.baseline_tokens_per_sec:.1f} tok/s)",
    )
    table.add_row(
        "Expected Accepted / Cycle",
        f"{res.expected_accepted_tokens:.2f} tokens",
        f"Guaranteed 1 token + (alpha {res.acceptance_rate} * K {res.lookahead_k})",
    )
    table.add_row(
        "Effective Speculative TPOT",
        f"{res.effective_tpot_ms:.1f} ms",
        "Net latency per emitted token across speculative cycles",
    )
    table.add_row(
        "Effective Generation Rate",
        f"{res.effective_tokens_per_sec:.1f} tok/s",
        "Effective throughput per concurrent user stream",
    )
    table.add_row(
        "NET SPECULATIVE SPEEDUP",
        f"[bold yellow]{res.speedup_ratio:.2f}x[/]",
        "Net acceleration over standard autoregressive decoding",
    )

    console.print(table)
    console.print()


@app.command()
def fit(
    model: str = typer.Option(
        "qwen2.5-7b", "--model", "-m", help="Model preset name or Hugging Face repository ID"
    ),
    quant: str = typer.Option("fp16", "--quant", "-q", help="Weight quantization precision"),
    kv_cache: str = typer.Option("fp16", "--kv-cache", help="KV-cache data type"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Concurrent request streams"),
    context: int = typer.Option(2048, "--context", "-c", help="Sequence length (prompt + output)"),
):
    """Detect local accelerator/CPU hardware and verify zero-OOM execution fit."""
    arch = _resolve_model(model)
    hw = detect_local_hardware()
    fit_res = verify_local_fit(
        arch=arch,
        precision=quant,
        kv_cache_dtype=kv_cache,
        batch_size=batch_size,
        context_tokens=context,
        hardware=hw,
    )

    console.print()
    verdict_style = (
        "bold green"
        if fit_res.fits_currently
        else ("bold yellow" if fit_res.fits_total else "bold red")
    )
    console.print(
        Panel(
            f"[bold cyan]Detected Silicon:[/] {hw.device_name} ({hw.total_memory_gb:.1f} GB Total, {hw.free_memory_gb:.1f} GB Currently Free)\n"
            f"[bold cyan]Model Workload:[/] {arch.name} ({quant.upper()}) | Batch: {batch_size} | Context: {context:,} tokens\n"
            f"[bold cyan]Memory Demand:[/] {fit_res.required_memory_gb:.2f} GB Required\n"
            f"[bold cyan]Verdict:[/] [{verdict_style}]{fit_res.verdict}[/]",
            title="[bold green]🖥️ INFERSIZE: LIVE HARDWARE ZERO-OOM VERIFIER[/]",
            border_style="green",
            box=box.ROUNDED,
        )
    )

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Memory Component", style="bold")
    table.add_column("Footprint", justify="right")
    table.add_column("Context")

    table.add_row(
        "Model Weights",
        f"{fit_res.weights_memory_gb:.2f} GB",
        f"Static memory for {arch.name} in {quant.upper()}",
    )
    table.add_row(
        "KV-Cache Buffer",
        f"{fit_res.kv_cache_memory_gb:.2f} GB",
        f"Batch {batch_size} x {context:,} tokens in {kv_cache.upper()}",
    )
    table.add_row(
        "Activation Overhead",
        f"{fit_res.activations_memory_gb:.2f} GB",
        "Transient tensor allocations",
    )
    table.add_section()
    table.add_row(
        "Total Required",
        f"{fit_res.required_memory_gb:.2f} GB",
        "Minimum physical memory necessary",
    )
    table.add_row(
        "Total Silicon Memory",
        f"{hw.total_memory_gb:.1f} GB",
        f"Physical capacity of {hw.device_name}",
    )
    table.add_row(
        "Currently Free Memory",
        f"{hw.free_memory_gb:.1f} GB",
        "Free memory without closing existing processes",
    )
    console.print(table)
    console.print()


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", "-p", help="Port to bind the Streamlit web studio"),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="Open browser automatically"),
):
    """Launch the interactive Streamlit LLM Capacity Planner web studio."""
    import subprocess

    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    if not app_path.exists():
        app_path = Path(__file__).resolve().parent.parent.parent / "app.py"

    console.print("[bold cyan]Launching infersize Interactive Sizing Studio...[/]")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    if not browser:
        cmd.extend(["--server.headless", "true"])

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("[bold yellow]Infersize UI stopped.[/]")
    except Exception as e:
        console.print(f"[bold red]Failed to start Streamlit:[/] {e}")
        console.print("[yellow]Make sure streamlit is installed: pip install 'infersize[ui]'[/]")


if __name__ == "__main__":
    app()
