"""
Shared rendering environment configuration.

This module ensures HOME is set, configures offscreen rendering, and centralizes
GPU detection/logging so every component (Flask app, Batch jobs, standalone
scripts) shares the same behavior and messaging.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from env_targets import get_targets


@dataclass(frozen=True)
class RenderingConfig:
    is_cloud: bool
    use_gpu: bool
    gpu_count: int


@dataclass(frozen=True)
class BatchStageConfig:
    machine_type: str
    cpu_count: int
    memory_gb: int
    gpu_type: str | None
    gpu_count: int
    description: str


@dataclass(frozen=True)
class BatchSettings:
    stages: Dict[str, BatchStageConfig]
    gcs_bucket: str
    region: str
    max_duration: str


def _ensure_home(is_cloud: bool) -> None:
    if "HOME" not in os.environ:
        default_home = "/root" if is_cloud else os.path.expanduser("~")
        os.environ["HOME"] = default_home


def _count_nvidia_gpus() -> int:
    dev_path = Path("/dev")
    if not dev_path.exists():
        return 0
    matches = list(dev_path.glob("nvidia[0-9]*"))
    return len(matches)


def _determine_gpu_mode() -> Tuple[bool, int]:
    override = os.environ.get("USE_GPU")
    gpu_count = _count_nvidia_gpus()

    if override == "1":
        return True, gpu_count or 1  # assume at least 1 GPU if forced on
    if override == "0":
        return False, 0

    return (gpu_count > 0), gpu_count


def _configure_offscreen_env(is_cloud: bool, use_gpu: bool) -> None:
    if is_cloud:
        os.environ.setdefault("ETS_TOOLKIT", "null")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Force VTK/PyVista to use windowless backends everywhere
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")

    if use_gpu:
        # Prefer EGL on GPU nodes to avoid X server dependency
        os.environ.setdefault("PYVISTA_EGL", "true")
        os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
        os.environ.pop("GALLIUM_DRIVER", None)
    else:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["GALLIUM_DRIVER"] = "llvmpipe"


def _log_status(is_cloud: bool, use_gpu: bool, gpu_count: int) -> None:
    location = "Cloud" if is_cloud else "Local"
    home = os.environ.get("HOME")

    prefix = f"✓ {location} environment detected - HOME={home}"
    print(prefix)

    if use_gpu:
        gpu_label = "GPU" if gpu_count == 1 else "GPUs"
        print(
            f"✓ GPU rendering enabled ({gpu_count} {gpu_label if gpu_count else 'GPU'})"
        )
    else:
        print("✓ Software rendering configured (CPU-only)")


def configure_rendering() -> RenderingConfig:
    is_cloud = Path("/app").exists() or os.environ.get("K_SERVICE") is not None
    _ensure_home(is_cloud)

    use_gpu, gpu_count = _determine_gpu_mode()
    _configure_offscreen_env(is_cloud, use_gpu)
    _log_status(is_cloud, use_gpu, gpu_count)

    return RenderingConfig(is_cloud=is_cloud, use_gpu=use_gpu, gpu_count=gpu_count)


# Configure immediately so imports see a consistent environment.
CONFIG = configure_rendering()
IS_CLOUD = CONFIG.is_cloud
USE_GPU = CONFIG.use_gpu
GPU_COUNT = CONFIG.gpu_count


def _get_git_branch() -> str | None:
    """Get current git branch name, return None if not in a git repo"""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# Determine environment
# Priority: ENV environment variable > git branch detection > default to DEV
env_override = os.environ.get("ENV")
if env_override:
    # Explicitly set via environment variable (Cloud Run, Cloud Batch)
    ENV = env_override.strip().upper()
    if ENV not in ("DEV", "PROD"):
        ENV = "PROD"
else:
    # Local development: auto-detect from git branch
    # Branch "release" → PROD, all other branches → DEV
    branch = _get_git_branch()
    if branch == "release":
        ENV = "PROD"
    else:
        ENV = "DEV"

ENV_TARGETS = get_targets(ENV)

# Shared project/data configuration
PROJECTS_ROOT = Path("/app/projects") if IS_CLOUD else Path("projects")
ALLOWED_EXTENSIONS = frozenset({"csv", "txt", "dat", "gcode"})
PIPELINE_FOLDERS = {
    "rust_code_input": "Input files for Rust thermal simulation",
    "frame_generator_input": "FEA data for visualization",
    "frames_filtered_active_only": "Generated visualization frames",
}


def resolve_project_dir(project_id: str, prefer_existing: bool = True) -> Path:
    """
    Return the canonical directory for a project.

    Prefer the flat layout (/projects/<id>) but fall back to the legacy
    env-specific path (/projects/<env>/<id>) if it already exists. When no existing
    folder is found, fall back to the flat layout so callers can create it.
    """
    normalized_id = project_id.strip().strip("/\\")
    primary = PROJECTS_ROOT / normalized_id
    legacy = PROJECTS_ROOT / ENV.lower() / normalized_id

    if prefer_existing:
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy

    return primary


def get_project_path(project_id: str) -> Path:
    """Convenience alias so callers can stay agnostic about layout quirks."""
    return resolve_project_dir(project_id)

# Cloud Batch defaults (centralized here so CLI, Flask, and Batch processors stay aligned)
# Use environment-aware bucket from ENV_TARGETS
BATCH_SETTINGS = BatchSettings(
    stages={
        "rust_simulation": BatchStageConfig(
            machine_type="n2-highmem-128",
            cpu_count=128,
            memory_gb=512,
            gpu_type=None,
            gpu_count=0,
            description="CPU-intensive thermal simulation",
        ),
        "visualization": BatchStageConfig(
            machine_type="n2-highmem-128",
            cpu_count=128,
            memory_gb=512,
            gpu_type=None,
            gpu_count=0,
            description="CPU-intensive visualization (software rendering)",
        ),
        "full_pipeline": BatchStageConfig(
            machine_type="n2-highmem-128",
            cpu_count=128,
            memory_gb=512,
            gpu_type=None,
            gpu_count=0,
            description="Full pipeline execution",
        ),
        "mask_generation": BatchStageConfig(
            machine_type="n2-highmem-32",
            cpu_count=32,
            memory_gb=128,
            gpu_type=None,
            gpu_count=0,
            description="Hot/cold mask generation (single frame)",
        ),
    },
    gcs_bucket=ENV_TARGETS.bucket,
    region="europe-west4",
    max_duration="86400s",
)

BATCH_CONFIG = BATCH_SETTINGS.stages
BATCH_GCS_BUCKET = BATCH_SETTINGS.gcs_bucket
BATCH_REGION = BATCH_SETTINGS.region
# Batch Job configuration
if ENV == "PROD":
    BATCH_MAX_DURATION = "86400s"  # 24 hours
else:
    BATCH_MAX_DURATION = "14400s"  # 4 hours (DEV)

# OAuth Configuration for Google Sign-In
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(24).hex())
