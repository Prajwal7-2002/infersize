"""Comprehensive tests for architecture report generation, RFC export, and CLI."""

import json

from typer.testing import CliRunner

from infersize.analytics.report import generate_architecture_report
from infersize.cli import app
from infersize.hardware.specs import get_gpu_profile
from infersize.models.specs import get_model_spec

runner = CliRunner()


def test_generate_architecture_report_structure():
    """Verify generated ArchitectureReport contains all required sections and metrics."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    rtx4090 = get_gpu_profile("rtx4090")

    rep = generate_architecture_report(
        model=qwen_7b,
        gpu=rtx4090,
        precision="fp16",
        batch_size=8,
        context_tokens=2048,
        provider="runpod",
    )

    assert rep.model_name == "qwen2.5-7b"
    assert rep.gpu_id == "rtx4090"
    assert rep.batch_size == 8
    assert rep.context_tokens == 2048
    assert rep.total_parameters > 7e9
    assert rep.num_layers == 28
    assert rep.gqa_ratio == 7.0

    # Sub-reports exist
    assert rep.vram_estimate.fits_in_vram
    assert rep.cost_estimate.aggregate_tokens_per_sec > 0
    assert len(rep.gpu_comparison.items) > 0
    assert len(rep.cloud_cost_comparison.results) > 0


def test_architecture_report_to_markdown():
    """Verify markdown output contains formatted sections, tables, and recommendations."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    rtx4090 = get_gpu_profile("rtx4090")

    rep = generate_architecture_report(
        model=qwen_7b,
        gpu=rtx4090,
        precision="fp16",
        batch_size=16,
        context_tokens=2048,
    )
    md = rep.to_markdown()

    assert "# Architecture RFC: LLM Capacity Planning & Sizing Specification" in md
    assert "## 1. Executive Summary & Workload Definition" in md
    assert "## 2. Transformer Architecture & Parameter Accounting" in md
    assert "## 3. Physical VRAM Allocation Breakdown" in md
    assert "## 4. Multi-Accelerator Comparison Matrix" in md
    assert "## 5. Decode Roofline Throughput & Cloud Unit Economics" in md
    assert "## 6. Actionable Procurement & Sizing Recommendation" in md
    assert "## 7. Mathematical & Analytical Foundations" in md
    assert "Optimal Right-Sized GPU" in md
    assert "COST PER 1M TOKENS" in md


def test_cli_report_output_file(tmp_path):
    """Verify CLI `report --output <file>` generates and writes file on disk."""
    out_file = tmp_path / "deployment_plan.md"
    res = runner.invoke(
        app,
        [
            "report",
            "--model",
            "qwen2.5-7b",
            "--gpu",
            "rtx4090",
            "--batch-size",
            "4",
            "--context",
            "1024",
            "--output",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert "SPECIFICATION EXPORT COMPLETE" in res.output
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "# Architecture RFC:" in content
    assert "qwen2.5-7b" in content


def test_cli_report_stdout():
    """Verify CLI `report` without --output renders markdown to console."""
    res = runner.invoke(app, ["report", "--model", "qwen2.5-7b", "--gpu", "rtx4090"])
    assert res.exit_code == 0
    assert "Executive Summary" in res.output
    assert "Tip: Use" in res.output


def test_cli_report_json():
    """Verify CLI `report --json` emits valid JSON with full report schema."""
    res = runner.invoke(app, ["report", "--model", "qwen2.5-7b", "--gpu", "rtx4090", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["model_name"] == "qwen2.5-7b"
    assert "vram_estimate" in data
    assert "cost_estimate" in data
    assert "gpu_comparison" in data
    assert "cloud_cost_comparison" in data


def test_cli_report_invalid_inputs():
    """Verify CLI handles invalid model or GPU names gracefully with exit code 1."""
    res = runner.invoke(app, ["report", "--model", "invalid-model"])
    assert res.exit_code == 1
    assert "Error" in res.output
