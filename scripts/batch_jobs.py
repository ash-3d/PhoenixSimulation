#!/usr/bin/env python3
"""Minimal command-line viewer for Google Cloud Batch jobs.

It wraps ``gcloud batch jobs list`` to produce a compact table that is easier to
scan than the default output and optionally refreshes the table in-place.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence
import re

DEFAULT_PROJECT = os.environ.get("GCP_PROJECT_ID")
DEFAULT_LOCATION = os.environ.get("GCP_LOCATION", "europe-west4")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
STATE_COLORS = {
    "RUNNING": "\033[33m",  # yellow
    "FAILED": "\033[31m",
    "SUCCEEDED": "\033[32m",
    "QUEUED": "\033[36m",
}
RESET = "\033[0m"
STARTED_STATES = {"STATE_RUNNING", "STATE_SUCCEEDED", "STATE_FAILED", "RUNNING", "SUCCEEDED", "FAILED"}


def run_cmd(args: Sequence[str]) -> str:
    """Run a command and return stdout on success."""
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("gcloud CLI not found. Install it or add it to $PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc
    return result.stdout


def parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return "<1s"
    minutes, secs = divmod(int(seconds), 60)
    hours, mins = divmod(minutes, 60)
    days, hrs = divmod(hours, 24)
    parts: List[str] = []
    if days:
        parts.append(f"{days}d")
    if hrs or (days and (mins or secs)):
        parts.append(f"{hrs}h")
    if mins or (parts and secs):
        parts.append(f"{mins}m")
    if not parts or secs:
        parts.append(f"{secs}s")
    return " ".join(parts)


def job_times(job: Dict[str, Any]) -> tuple[datetime | None, datetime | None, datetime | None]:
    status = job.get("status") or {}
    create = parse_rfc3339(job.get("createTime"))
    start = parse_rfc3339(status.get("startTime"))
    end = parse_rfc3339(status.get("completionTime"))
    return create, start, end


def transition_time(job: Dict[str, Any], target_states: set[str]) -> datetime | None:
    status = job.get("status") or {}
    events = status.get("statusEvents") or []
    # Events usually arrive newest-first; sort so we can pick the earliest relevant transition.
    sorted_events = sorted(
        (
            (parse_rfc3339(event.get("time")), (event.get("state") or "").upper())
            for event in events
        ),
        key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc),
    )
    for ts, state in sorted_events:
        if ts and state in target_states:
            return ts
    return None


def effective_start_time(job: Dict[str, Any]) -> datetime | None:
    _, start, _ = job_times(job)
    if start:
        return start
    return transition_time(job, STARTED_STATES)


def compute_duration(job: Dict[str, Any]) -> str:
    create, _, end = job_times(job)
    reference_start = effective_start_time(job) or create
    if not reference_start:
        return "-"
    if end:
        seconds = (end - reference_start).total_seconds()
    else:
        seconds = (datetime.now(timezone.utc) - reference_start).total_seconds()
    return human_duration(max(seconds, 0))


def compute_wait_time(job: Dict[str, Any]) -> str:
    create, _, _ = job_times(job)
    start = effective_start_time(job)
    if not create:
        return "-"
    if start:
        seconds = (start - create).total_seconds()
    else:
        seconds = (datetime.now(timezone.utc) - create).total_seconds()
    return human_duration(max(seconds, 0))


def _to_number(value: Any) -> float | None:
    """Convert gcloud numeric fields that sometimes arrive as strings."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def format_memory(task_group: Dict[str, Any]) -> str:
    resources = ((task_group.get("taskSpec") or {}).get("computeResource") or {})
    memory_mib = _to_number(resources.get("memoryMib"))
    if memory_mib is None or memory_mib <= 0:
        return "-"
    # Round to whole GiB to match the UI table.
    return f"{int(round(memory_mib / 1024))} GB"


def format_cpu(task_group: Dict[str, Any]) -> str:
    resources = ((task_group.get("taskSpec") or {}).get("computeResource") or {})
    cpu_milli = _to_number(resources.get("cpuMilli"))
    if cpu_milli is None or cpu_milli <= 0:
        return "-"
    return f"{cpu_milli / 1000:.0f} vCPU"


def machine_type(job: Dict[str, Any]) -> str:
    policy = (job.get("allocationPolicy") or {}).get("instances") or []
    if not policy:
        return "-"
    machine = (
        (policy[0].get("policy") or {}).get("machineType")
        or policy[0].get("machineType")
    )
    return machine or "-"


def pick_task_group(job: Dict[str, Any]) -> Dict[str, Any]:
    groups = job.get("taskGroups") or []
    return groups[0] if groups else {}


def load_jobs(project: str, location: str, limit: int) -> List[Dict[str, Any]]:
    args = [
        "gcloud",
        "batch",
        "jobs",
        "list",
        "--project",
        project,
        "--location",
        location,
        "--limit",
        str(limit),
        "--format",
        "json",
        "--sort-by",
        "~createTime",
    ]
    output = run_cmd(args).strip()
    return json.loads(output) if output else []


def job_to_row(job: Dict[str, Any]) -> List[str]:
    name = job.get("name", "").split("/")[-1]
    status = (job.get("status") or {}).get("state", "STATE_UNSPECIFIED")
    state = status.replace("STATE_", "").replace("_", " ").title()
    colored_state = colorize_state(state, (job.get("status") or {}).get("state", ""))
    task_group = pick_task_group(job)
    rows = [
        name or "-",
        colored_state,
        str(task_group.get("taskCount", "-")),
        format_memory(task_group),
        format_cpu(task_group),
        machine_type(job),
        format_time(job.get("createTime")),
        compute_wait_time(job),
        compute_duration(job),
    ]
    return rows


def format_time(value: str | None) -> str:
    ts = parse_rfc3339(value)
    if not ts:
        return "-"
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def colorize_state(display: str, state_raw: str) -> str:
    if not COLOR_ENABLED:
        return display
    color = STATE_COLORS.get(state_raw.upper())
    if not color:
        return display
    return f"{color}{display}{RESET}"


def render_table(rows: List[List[str]], headers: Sequence[str]) -> str:
    if not rows:
        return "No jobs found."
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], visible_len(cell))

    def fmt(line: Sequence[str]) -> str:
        padded = []
        for idx, cell in enumerate(line):
            pad = widths[idx] - visible_len(cell)
            padded.append(f"{cell}{' ' * pad}")
        return "  ".join(padded)

    lines = [fmt(headers), fmt(["-" * len(h) for h in headers])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def filter_jobs(
    jobs: Iterable[Dict[str, Any]],
    states: set[str] | None,
    name_filter: str | None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for job in jobs:
        state = ((job.get("status") or {}).get("state") or "").upper()
        if states and state not in states:
            continue
        if name_filter and name_filter.lower() not in job.get("name", "").lower():
            continue
        filtered.append(job)
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show Cloud Batch job statuses in a compact table."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="GCP project ID")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Batch region")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of jobs to fetch (default: 20)",
    )
    parser.add_argument(
        "--state",
        help="Comma-separated list of job states to keep (e.g. RUNNING,FAILED)",
    )
    parser.add_argument(
        "--match",
        help="Only show jobs whose names contain this substring (case-insensitive)",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Refresh every N seconds (set to 0 to print once). Default: 5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.project:
        print("Error: GCP project ID is required", file=sys.stderr)
        print("Set GCP_PROJECT_ID environment variable or use --project flag", file=sys.stderr)
        print("Example: export GCP_PROJECT_ID=your-project-id", file=sys.stderr)
        sys.exit(1)

    states = (
        {state.strip().upper() for state in args.state.split(",")} if args.state else None
    )

    headers = [
        "Job name",
        "Status",
        "# Tasks",
        "Memory/task",
        "Core/task",
        "Machine type",
        "Created",
        "Wait",
        "Runtime",
    ]

    interval = args.watch or 0
    while True:
        try:
            jobs = load_jobs(args.project, args.location, args.limit)
            jobs = filter_jobs(jobs, states, args.match)
            jobs.sort(
                key=lambda job: parse_rfc3339(job.get("createTime"))
                or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            rows = [job_to_row(job) for job in jobs]
            output = render_table(rows, headers)
        except RuntimeError as exc:
            output = f"Error fetching jobs: {exc}"
        if interval > 0:
            print("\033c", end="")  # Clear screen for a dynamic feel.
        print(output)
        if interval <= 0:
            break
        time.sleep(max(interval, 1.0))


if __name__ == "__main__":
    main()
