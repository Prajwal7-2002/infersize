"""User-supplied hourly hardware rate helper.

Hardcoded cloud provider pricing tables were removed because cloud GPU rental rates
change frequently and cannot be verified statically. Pass ``custom_rate`` (e.g. via
``--rate`` on the CLI) when computing dollar unit economics.
"""

from __future__ import annotations


def get_hourly_rate(
    gpu_id: str,
    provider: str = "custom",
    custom_rate: float | None = None,
) -> float:
    """Return a user-supplied hourly rate in USD, or 0.0 when no rate is provided."""
    if custom_rate is not None and custom_rate >= 0.0:
        return float(custom_rate)
    return 0.0
