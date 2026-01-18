#!/usr/bin/env python3
"""Quick textual viewer for Google Cloud Build history."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

DEFAULT_PROJECT = os.environ.get("GCP_PROJECT_ID")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
STATUS_COLORS = {
    "QUEUED": "\033[36m",
    "WORKING": "\033[33m",
    "SUCCESS": "\033[32m",
    "FAILURE": "\033[31m",
    "INTERNAL_ERROR": "\033[31m",
    "TIMEOUT": "\033[31m",
    "CANCELLED": "\033[35m",
}
RESET = "\033[0m"


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


def build_times(build: Dict[str, Any]) -> tuple[datetime | None, datetime | None, datetime | None]:
    return (
        parse_rfc3339(build.get("createTime")),
        parse_rfc3339(build.get("startTime")),
        parse_rfc3339(build.get("finishTime")),
    )


def compute_wait_time(build: Dict[str, Any]) -> str:
    created, started, _ = build_times(build)
    if not created:
        return "-"
    reference = started or datetime.now(timezone.utc)
    return human_duration(max((reference - created).total_seconds(), 0))


def compute_duration(build: Dict[str, Any]) -> str:
    _, started, finished = build_times(build)
    if not started:
        return "-"
    if finished:
        seconds = (finished - started).total_seconds()
    else:
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
    return human_duration(max(seconds, 0))


def format_time(value: str | None) -> str:
    ts = parse_rfc3339(value)
    if not ts:
        return "-"
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def colorize_status(display: str, status_raw: str) -> str:
    if not COLOR_ENABLED:
        return display
    color = STATUS_COLORS.get(status_raw.upper())
    if not color:
        return display
    return f"{color}{display}{RESET}"


def format_trigger(build: Dict[str, Any]) -> str:
    trigger = build.get("buildTriggerId") or build.get("triggerId")
    if trigger:
        return trigger
    repo = ((build.get("source") or {}).get("repoSource") or {}).get("repoName")
    if repo:
        return repo
    storage_source = ((build.get("source") or {}).get("storageSource") or {}).get("object")
    if storage_source:
        return storage_source
    return "-"


def format_images(build: Dict[str, Any]) -> str:
    images = build.get("images") or []
    if not images:
        return "-"
    first = images[0]
    remaining = len(images) - 1
    return f"{first} (+{remaining})" if remaining > 0 else first


def format_commit(build: Dict[str, Any]) -> str:
    substitutions = build.get("substitutions") or {}
    commit = substitutions.get("COMMIT_SHA") or substitutions.get("REVISION_ID")
    if not commit:
        repo_source = ((build.get("source") or {}).get("repoSource") or {})
        commit = repo_source.get("commitSha")
    if not commit:
        resolved = ((build.get("resolvedSource") or {}).get("repoSource") or {})
        commit = resolved.get("commitSha")
    if not commit:
        provenance = (
            (build.get("sourceProvenance") or {}).get("resolvedRepoSource") or {}
        )
        commit = provenance.get("commitSha")
    if not commit:
        return "-"
    return commit[:7]


def render_table(rows: List[List[str]], headers: Sequence[str]) -> str:
    if not rows:
        return "No builds found."
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


def filter_builds(
    builds: Iterable[Dict[str, Any]],
    states: set[str] | None,
    name_filter: str | None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for build in builds:
        status = (build.get("status") or "").upper()
        if states and status not in states:
            continue
        if name_filter:
            haystack = " ".join(
                [
                    build.get("id", ""),
                    format_trigger(build),
                    " ".join(build.get("images") or []),
                ]
            ).lower()
            if name_filter.lower() not in haystack:
                continue
        filtered.append(build)
    return filtered


def load_builds(project: str, limit: int) -> List[Dict[str, Any]]:
    args = [
        "gcloud",
        "builds",
        "list",
        "--project",
        project,
        "--limit",
        str(limit),
        "--sort-by",
        "~createTime",
        "--format",
        "json",
    ]
    output = run_cmd(args).strip()
    return json.loads(output) if output else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show Cloud Build history in a tight table.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="GCP project ID")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of builds to fetch (default: 20)",
    )
    parser.add_argument(
        "--state",
        help="Comma-separated list of build states to keep (e.g. WORKING,SUCCESS)",
    )
    parser.add_argument(
        "--match",
        help="Only show builds whose ID/trigger/image contains this substring (case-insensitive)",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Refresh every N seconds (set to 0 to print once). Default: 5",
    )
    return parser.parse_args()


def build_to_row(build: Dict[str, Any]) -> List[str]:
    status = build.get("status", "STATUS_UNKNOWN")
    human_status = status.replace("_", " ").title()
    return [
        build.get("id", "-"),
        colorize_status(human_status, status),
        format_trigger(build),
        format_commit(build),
        format_images(build),
        format_time(build.get("createTime")),
        compute_wait_time(build),
        compute_duration(build),
    ]


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
        "Build ID",
        "Status",
        "Trigger/Source",
        "Commit",
        "Images",
        "Created",
        "Wait",
        "Runtime",
    ]

    interval = args.watch or 0
    while True:
        try:
            builds = load_builds(args.project, args.limit)
            builds = filter_builds(builds, states, args.match)
            builds.sort(
                key=lambda build: parse_rfc3339(build.get("createTime"))
                or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            rows = [build_to_row(build) for build in builds]
            output = render_table(rows, headers)
        except RuntimeError as exc:
            output = f"Error fetching builds: {exc}"
        if interval > 0:
            print("\033c", end="")
        print(output)
        if interval <= 0:
            break
        time.sleep(max(interval, 1.0))


if __name__ == "__main__":
    main()
