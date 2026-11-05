"""Check predictions against decode speeds measured on real hardware.

The default prediction is a bandwidth *ceiling*: it must never be below what a real
engine achieves. The overhead model is checked out of sample (leave-one-out): the
per-token overhead is fitted on two models and used to predict the third.
"""

import json
from pathlib import Path

import pytest

from infersize.analytics.cost import estimate_serving_cost
from infersize.hardware.specs import get_gpu_profile
from infersize.models.specs import get_model_spec

DATA = json.loads((Path(__file__).parent / "data" / "measured_rtx3050_ollama.json").read_text())
GPU = get_gpu_profile("rtx3050")


def _step_ms(preset: str, overhead_ms: float = 0.0) -> float:
    res = estimate_serving_cost(
        get_model_spec(preset),
        GPU,
        precision=DATA["precision"],
        context_tokens=DATA["context_tokens"],
        decode_overhead_ms=overhead_ms,
    )
    return res.time_per_step_ms


@pytest.mark.parametrize("run", DATA["runs"], ids=lambda r: r["preset"])
def test_ceiling_is_never_below_measured(run):
    ceiling_tok_s = 1000.0 / _step_ms(run["preset"])
    assert ceiling_tok_s >= run["decode_tok_s"]


@pytest.mark.parametrize("held_out", range(len(DATA["runs"])))
def test_overhead_model_predicts_held_out_model(held_out):
    runs = DATA["runs"]
    train = [r for i, r in enumerate(runs) if i != held_out]
    overhead = sum(1000.0 / r["decode_tok_s"] - _step_ms(r["preset"]) for r in train) / len(train)
    target = runs[held_out]
    predicted = 1000.0 / _step_ms(target["preset"], overhead)
    rel_err = abs(predicted - target["decode_tok_s"]) / target["decode_tok_s"]
    assert rel_err < 0.15, (
        f"{target['preset']}: predicted {predicted:.1f}, measured {target['decode_tok_s']}"
    )
