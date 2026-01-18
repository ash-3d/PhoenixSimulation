"""Environment-specific resource targets for dev and prod."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Target:
    bucket: str
    service: str
    batch_job_prefix: str


DEV = Target(
    bucket="citb4-projects-dev",
    service="citb4-dev",
    batch_job_prefix="dev",
)

PROD = Target(
    bucket="prod-projects",
    service="citb4",
    batch_job_prefix="prod",
)

TARGETS: Dict[str, Target] = {
    "DEV": DEV,
    "PROD": PROD,
}


def get_targets(env: str | None = None) -> Target:
    """Return the target config for the requested environment."""
    if env:
        key = env.upper()
        if key in TARGETS:
            return TARGETS[key]
    return PROD
