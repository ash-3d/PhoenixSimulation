#!/usr/bin/env python3
"""
Fetch the failure logs for a specific Cloud Batch job and show a compact context
around error events.

By default, searches backwards from the end of logs to find the most recent/final
error (which is usually the actual cause of batch job failures).

Usage examples:
  # Default: Show context around the final error (most common use case)
  python fetch_batch_failure_logs.py JOB_ID

  # Show last 30 log entries (quick check)
  python fetch_batch_failure_logs.py JOB_ID --tail 30

  # Show both first and last errors (comprehensive view)
  python fetch_batch_failure_logs.py JOB_ID --show-both-ends

  # Fetch more logs if job has extensive output
  python fetch_batch_failure_logs.py JOB_ID --limit 500
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple


def run_cmd(args: List[str]) -> str:
    """Run a command and return stdout, raising a helpful error on failure."""
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"Command failed: {' '.join(args)}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}",
            file=sys.stderr,
        )
        raise
    return result.stdout


PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
if not PROJECT_ID:
    print("Error: Set GCP_PROJECT_ID environment variable", file=sys.stderr)
    print("Example: export GCP_PROJECT_ID=your-project-id", file=sys.stderr)
    sys.exit(1)

LOCATION = os.environ.get("GCP_LOCATION", "europe-west4")


def describe_job(job_id: str) -> Dict[str, Any]:
    cmd = [
        "gcloud",
        "batch",
        "jobs",
        "describe",
        job_id,
        "--location",
        LOCATION,
        "--format",
        "json",
    ]
    cmd.extend(["--project", PROJECT_ID])
    return json.loads(run_cmd(cmd))


def read_job_logs(job_uid: str, limit: int, order: str = "asc") -> List[Dict[str, Any]]:
    # Try batch_task_logs first (application output), fall back to batch_agent_logs (system events)
    filter_expr = (
        f'(logName="projects/{PROJECT_ID}/logs/batch_task_logs" '
        f'OR logName="projects/{PROJECT_ID}/logs/batch_agent_logs") '
        f'AND labels.job_uid="{job_uid}"'
    )

    cmd = [
        "gcloud",
        "logging",
        "read",
        filter_expr,
        "--limit",
        str(limit),
        "--format",
        "json",
        "--order",
        order,
        "--project",
        PROJECT_ID,
    ]
    output = run_cmd(cmd).strip()
    if not output:
        return []
    return json.loads(output)


# Failure keyword ranking: (keyword, priority_score)
# Higher score = higher priority
FAILURE_KEYWORDS = [
    # High priority (actual failures)
    ("job failed", 10),
    ("❌", 10),
    ("not found", 9),
    ("exception", 9),
    ("traceback", 9),
    ("file not found", 9),
    ("directory not found", 9),
    # Medium priority
    ("failed", 7),
    ("error:", 6),
    ("error ", 5),
    # Low priority (often just warnings)
    ("warn", 2),
]


def _score_entry(entry: Dict[str, Any]) -> int:
    """Score an entry based on severity and keyword matching."""
    severity = (entry.get("severity") or "").upper()
    text = (entry.get("textPayload", "") or "").lower()

    score = 0

    # Severity scoring
    if severity in {"ERROR", "CRITICAL", "ALERT", "EMERGENCY"}:
        score += 8
    elif severity == "WARNING":
        score += 2

    # Keyword scoring
    for keyword, keyword_score in FAILURE_KEYWORDS:
        if keyword.lower() in text:
            score += keyword_score

    return score


def find_failure_context(
    entries: List[Dict[str, Any]],
    context: int,
    search_backwards: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return a slice of log entries centered around the failure entry.

    Args:
        entries: List of log entries to search
        context: Number of entries to show before/after failure
        search_backwards: If True, search from end (most recent) backwards

    Returns:
        Tuple of (context_entries, failure_index)
    """
    if not entries:
        return [], -1

    # Score all entries
    scored_entries = [(idx, _score_entry(entry)) for idx, entry in enumerate(entries)]

    # Find highest scoring entry
    if search_backwards:
        # Search from end backwards (prioritize recent failures)
        scored_entries.reverse()

    failure_idx = -1
    max_score = 0

    for idx, score in scored_entries:
        if score > max_score:
            max_score = score
            failure_idx = idx
            if score >= 10:  # High priority match, stop searching
                break

    # If no failure found, use last entry
    if failure_idx == -1:
        failure_idx = len(entries) - 1

    start = max(0, failure_idx - context)
    end = min(len(entries), failure_idx + context + 1)
    return entries[start:end], failure_idx


def pretty_print(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        ts = entry.get("timestamp", "?")
        severity = entry.get("severity", "INFO")
        text = (
            entry.get("textPayload")
            or json.dumps(entry.get("jsonPayload", {}), indent=2)
            or ""
        )
        print(f"[{ts}] {severity}")
        print(text.rstrip())
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show failure context for a Cloud Batch job."
    )
    parser.add_argument("job_id", help="Batch job ID (e.g. citb4-foo-123456)")
    parser.add_argument(
        "--context",
        type=int,
        default=5,
        help="Number of log entries to show before/after the failure entry.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of log entries to pull from Cloud Logging.",
    )
    parser.add_argument(
        "--search-backwards",
        action="store_true",
        default=True,
        help="Search from end of logs backwards (default: True). Use --no-search-backwards to search from beginning.",
    )
    parser.add_argument(
        "--no-search-backwards",
        action="store_false",
        dest="search_backwards",
        help="Search from beginning of logs forward instead of backwards.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        help="Just show the last N log entries (ignores --context and failure detection).",
    )
    parser.add_argument(
        "--show-both-ends",
        action="store_true",
        help="Show both first error AND last error (useful for batch jobs with multiple failure points).",
    )

    args = parser.parse_args()

    job = describe_job(args.job_id)
    job_uid = job.get("uid")
    job_state = job.get("status", {}).get("state")
    if not job_uid:
        print("Could not determine job UID from gcloud response.", file=sys.stderr)
        sys.exit(2)

    logs = read_job_logs(job_uid, args.limit)
    if not logs:
        print("No logs found for that job.", file=sys.stderr)
        sys.exit(3)

    print(f"Job: {args.job_id} (state: {job_state}, uid: {job_uid})")
    print(f"Total log entries: {len(logs)}")

    # Handle --tail option (just show last N entries)
    if args.tail:
        print(f"Showing last {args.tail} log entries")
        print("=" * 80)
        tail_entries = logs[-args.tail:] if len(logs) > args.tail else logs
        pretty_print(tail_entries)
        return

    # Handle --show-both-ends option
    if args.show_both_ends:
        # Find first failure
        first_context, first_idx = find_failure_context(logs, args.context, search_backwards=False)
        # Find last failure
        last_context, last_idx = find_failure_context(logs, args.context, search_backwards=True)

        print("=" * 80)
        print("FIRST FAILURE POINT")
        print("=" * 80)
        print(f"Showing {len(first_context)} entries around index {first_idx}")
        print("-" * 80)
        pretty_print(first_context)

        if first_idx != last_idx:
            print("\n" + "=" * 80)
            print("LAST FAILURE POINT")
            print("=" * 80)
            print(f"Showing {len(last_context)} entries around index {last_idx}")
            print("-" * 80)
            pretty_print(last_context)
        else:
            print("\n(First and last failure points are the same)")
        return

    # Default: find single failure point
    context_entries, failure_idx = find_failure_context(logs, args.context, args.search_backwards)

    search_dir = "backwards (from end)" if args.search_backwards else "forwards (from start)"
    print(f"Search direction: {search_dir}")
    print(f"Showing {len(context_entries)} entries around index {failure_idx}")
    print("=" * 80)
    pretty_print(context_entries)


if __name__ == "__main__":
    main()
