#!/usr/bin/env python3
"""Interactive command-line viewer for Google Cloud Batch jobs with keyboard navigation."""

from __future__ import annotations

import curses
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

# Import functions from the original batch_jobs.py script
sys.path.insert(0, os.path.dirname(__file__))
from batch_jobs import (
    load_jobs,
    filter_jobs,
    job_to_row,
    parse_rfc3339,
    DEFAULT_PROJECT,
    DEFAULT_LOCATION,
)

# Failure keywords for log analysis
FAILURE_KEYWORDS = [
    ("job failed", 10),
    ("❌", 10),
    ("not found", 9),
    ("exception", 9),
    ("traceback", 9),
    ("failed", 7),
    ("error:", 6),
    ("error ", 5),
    ("warn", 2),
]

def read_job_logs(job_uid: str, project: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch logs for a job from Cloud Logging."""
    try:
        filter_expr = (
            f'(logName="projects/{project}/logs/batch_task_logs" '
            f'OR logName="projects/{project}/logs/batch_agent_logs") '
            f'AND labels.job_uid="{job_uid}"'
        )
        result = subprocess.run(
            [
                "gcloud",
                "logging",
                "read",
                filter_expr,
                "--limit",
                str(limit),
                "--format",
                "json",
                "--order",
                "desc",  # Descending = most recent first (faster for failures)
                "--freshness",
                "7d",  # Only look at logs from last 7 days
                "--project",
                project,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,  # Add 15 second timeout
        )
        output = result.stdout.strip()
        if not output:
            return []
        logs = json.loads(output)
        # Reverse to get chronological order (for context display)
        return list(reversed(logs))
    except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return []


def score_log_entry(entry: Dict[str, Any]) -> int:
    """Score a log entry based on severity and keywords."""
    severity = (entry.get("severity") or "").upper()
    text = (entry.get("textPayload", "") or "").lower()

    score = 0
    if severity in {"ERROR", "CRITICAL", "ALERT", "EMERGENCY"}:
        score += 8
    elif severity == "WARNING":
        score += 2

    for keyword, keyword_score in FAILURE_KEYWORDS:
        if keyword.lower() in text:
            score += keyword_score

    return score


def find_failure_in_logs(entries: List[Dict[str, Any]], context: int = 5) -> List[Dict[str, Any]]:
    """Find and return log entries around the most likely failure point."""
    if not entries:
        return []

    # Score entries and find highest scoring one (searching backwards for most recent)
    max_score = 0
    failure_idx = len(entries) - 1

    for idx in range(len(entries) - 1, -1, -1):
        score = score_log_entry(entries[idx])
        if score > max_score:
            max_score = score
            failure_idx = idx
            if score >= 10:  # High priority match
                break

    start = max(0, failure_idx - context)
    end = min(len(entries), failure_idx + context + 1)
    return entries[start:end]


def show_failure_logs(stdscr, job: Dict[str, Any], project: str):
    """Display failure logs for a job in a scrollable popup."""
    height, width = stdscr.getmaxyx()

    job_name = job.get("name", "").split("/")[-1]
    job_uid = job.get("uid", "")
    state = (job.get("status") or {}).get("state", "")

    # Create popup window
    popup_height = height - 4
    popup_width = min(120, width - 4)
    popup_y = 2
    popup_x = (width - popup_width) // 2

    popup = curses.newwin(popup_height, popup_width, popup_y, popup_x)
    popup.keypad(True)
    popup.box()

    # Header
    popup.addstr(1, 2, f"Job: {job_name[:popup_width-6]}", curses.A_BOLD)
    popup.addstr(2, 2, f"State: {state}")
    popup.addstr(3, 2, "-" * (popup_width - 4))

    if not job_uid:
        popup.addstr(5, 2, "No job UID available")
        popup.addstr(popup_height - 2, 2, "Press any key to return...", curses.A_DIM)
        popup.refresh()
        popup.getch()
        return

    popup.addstr(5, 2, "Fetching logs (this may take 5-10 seconds)...", curses.A_DIM)
    popup.refresh()

    # Fetch logs (reduced limit for faster response)
    logs = read_job_logs(job_uid, project, limit=100)

    if not logs:
        popup.addstr(5, 2, " " * 50)  # Clear "Fetching logs..."
        popup.addstr(5, 2, "No logs found for this job")
        popup.addstr(popup_height - 2, 2, "Press any key to return...", curses.A_DIM)
        popup.refresh()
        popup.getch()
        return

    # Find failure context
    context_logs = find_failure_in_logs(logs, context=10)

    # Format log lines
    log_lines = []
    for entry in context_logs:
        ts = entry.get("timestamp", "?")
        severity = entry.get("severity", "INFO")
        text = entry.get("textPayload", "")
        if not text:
            text = json.dumps(entry.get("jsonPayload", {}))

        # Add header line
        log_lines.append(f"[{ts}] {severity}")
        # Add text lines (split long lines)
        for line in text.split("\n"):
            if line.strip():
                # Clean line: replace non-printable chars
                line = "".join(c if c.isprintable() or c == '\t' else '?' for c in line)
                # Wrap long lines
                while len(line) > popup_width - 6:
                    log_lines.append(line[:popup_width-6])
                    line = line[popup_width-6:]
                if line:
                    log_lines.append(line)
        log_lines.append("-" * (popup_width - 4))

    # Scrollable view
    scroll_offset = 0
    max_display_lines = popup_height - 6

    while True:
        popup.clear()
        popup.box()
        popup.addstr(1, 2, f"Job: {job_name[:popup_width-6]}", curses.A_BOLD)
        popup.addstr(2, 2, f"Failure Logs ({len(context_logs)} entries, {len(logs)} total)")
        popup.addstr(3, 2, "-" * (popup_width - 4))

        # Display log lines
        display_lines = log_lines[scroll_offset:scroll_offset + max_display_lines]
        for i, line in enumerate(display_lines):
            y = 4 + i
            if y >= popup_height - 2:
                break
            # Truncate line if too long and ensure it fits
            display_line = line[:popup_width-4]
            try:
                # Safely add string, avoiding bottom-right corner issue
                if y < popup_height - 1:
                    popup.addstr(y, 2, display_line)
            except curses.error:
                # Skip lines that cause curses errors (encoding issues, etc.)
                pass

        # Footer
        footer = f"↑↓ Scroll | q: Close | Total lines: {len(log_lines)}"
        popup.addstr(popup_height - 2, 2, footer[:popup_width-4], curses.A_DIM)
        popup.refresh()

        # Handle input
        key = popup.getch()
        if key == ord("q") or key == ord("Q"):
            break
        elif key == curses.KEY_UP or key == ord("k"):
            if scroll_offset > 0:
                scroll_offset -= 1
        elif key == curses.KEY_DOWN or key == ord("j"):
            if scroll_offset < max(0, len(log_lines) - max_display_lines):
                scroll_offset += 1
        elif key == curses.KEY_PPAGE:  # Page Up
            scroll_offset = max(0, scroll_offset - max_display_lines)
        elif key == curses.KEY_NPAGE:  # Page Down
            scroll_offset = min(len(log_lines) - max_display_lines, scroll_offset + max_display_lines)


def get_vm_instance_name(job_name: str, project: str) -> str | None:
    """Find the VM instance name for a running batch job."""
    try:
        # Use first 15 chars (safe for truncated VM names) or search by job name prefix
        search_prefix = job_name[:15]
        result = subprocess.run(
            [
                "gcloud",
                "compute",
                "instances",
                "list",
                "--project",
                project,
                "--format=value(name,zone)",
                f"--filter=name~{search_prefix}",  # Match first 15 chars (safer for truncation)
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")
        if lines and lines[0]:
            parts = lines[0].split("\t")
            if len(parts) >= 2:
                return parts[0], parts[1]  # instance_name, zone
    except subprocess.CalledProcessError:
        pass
    return None, None


def get_cpu_utilization(instance_name: str, zone: str, project: str) -> str:
    """Fetch recent CPU utilization for a VM instance using gcloud CLI."""
    try:
        # Use gcloud to get CPU metrics (simpler than REST API)
        result = subprocess.run(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                instance_name,
                "--zone", zone,
                "--project", project,
                "--format=value(status)",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        status = result.stdout.strip()
        if status != "RUNNING":
            return f"Instance status: {status}"

        # Just show basic info - detailed CPU metrics require complex API queries
        return f"Instance running in {zone} (CPU metrics require monitoring API setup)"

    except subprocess.TimeoutExpired:
        return "Timeout checking instance status"
    except Exception as e:
        return f"Error: {str(e)[:60]}"


def show_cpu_info(stdscr, job: Dict[str, Any], project: str):
    """Display CPU utilization info for a job in a popup."""
    height, width = stdscr.getmaxyx()

    # Get job details
    job_name = job.get("name", "").split("/")[-1]
    state = (job.get("status") or {}).get("state", "")

    # Create popup window
    popup_height = 15
    popup_width = min(80, width - 4)
    popup_y = (height - popup_height) // 2
    popup_x = (width - popup_width) // 2

    popup = curses.newwin(popup_height, popup_width, popup_y, popup_x)
    popup.box()

    # Header
    popup.addstr(1, 2, f"Job: {job_name[:popup_width-6]}", curses.A_BOLD)
    popup.addstr(2, 2, f"State: {state}")
    popup.addstr(3, 2, "-" * (popup_width - 4))

    if "RUNNING" not in state.upper():
        popup.addstr(5, 2, "CPU metrics only available for RUNNING jobs")
        popup.addstr(6, 2, "(VMs are terminated when jobs complete)")
    else:
        popup.addstr(5, 2, "Searching for VM instance...")
        popup.refresh()

        # Find VM
        instance_name, zone = get_vm_instance_name(job_name, project)

        if not instance_name:
            popup.addstr(6, 2, "No VM instance found")
            popup.addstr(7, 2, "(VM may not be provisioned yet)")
        else:
            popup.addstr(6, 2, f"Instance: {instance_name[:popup_width-12]}")
            popup.addstr(7, 2, f"Zone: {zone}")
            popup.addstr(8, 2, "-" * (popup_width - 4))
            popup.addstr(9, 2, "Checking status...")
            popup.refresh()

            # Get CPU/status info
            cpu_info = get_cpu_utilization(instance_name, zone, project)
            popup.addstr(10, 2, cpu_info[:popup_width-4])

    popup.addstr(popup_height - 2, 2, "Press any key to return...", curses.A_DIM)
    popup.refresh()
    popup.getch()


def interactive_table(stdscr, project: str, location: str, limit: int):
    """Display interactive job table with keyboard navigation."""
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(0)  # Blocking input
    stdscr.timeout(-1)

    # Init colors
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # RUNNING
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)  # FAILED
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # SUCCEEDED
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # QUEUED
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected

    selected_idx = 0
    scroll_offset = 0
    jobs = []  # Cache jobs list
    need_refresh = True

    headers = [
        "Job name",
        "Status",
        "# Tasks",
        "Memory/task",
        "Core/task",
        "Machine type",
        "Created",
        "Runtime",
    ]

    while True:
        # Only load jobs when needed (initial load or manual refresh)
        if need_refresh:
            stdscr.clear()
            stdscr.addstr(0, 0, "Loading jobs...", curses.A_DIM)
            stdscr.refresh()

            try:
                jobs = load_jobs(project, location, limit)
                jobs = filter_jobs(jobs, None, None)
                jobs.sort(
                    key=lambda j: parse_rfc3339(j.get("createTime"))
                    or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )
                need_refresh = False
            except Exception as e:
                stdscr.clear()
                stdscr.addstr(0, 0, f"Error: {str(e)[:stdscr.getmaxyx()[1]-1]}")
                stdscr.addstr(2, 0, "Press 'r' to retry or 'q' to quit")
                stdscr.refresh()
                key = stdscr.getch()
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    need_refresh = True
                continue

        stdscr.clear()
        height, width = stdscr.getmaxyx()

        if not jobs:
            stdscr.addstr(0, 0, "No jobs found")
            stdscr.addstr(2, 0, "Press 'r' to refresh or 'q' to quit")
            stdscr.refresh()
            key = stdscr.getch()
            if key == ord("q"):
                break
            elif key == ord("r"):
                need_refresh = True
            continue

        # Render table
        max_display_rows = height - 5
        display_jobs = jobs[scroll_offset : scroll_offset + max_display_rows]

        # Header
        stdscr.addstr(0, 0, "Google Cloud Batch Jobs", curses.A_BOLD)
        stdscr.addstr(
            1,
            0,
            f"Project: {project} | Location: {location} | Total: {len(jobs)}",
            curses.A_DIM,
        )

        # Column headers (simplified for terminal width)
        header_y = 3
        stdscr.addstr(header_y, 0, "Job name".ljust(40), curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr(header_y, 41, "Status".ljust(12), curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr(header_y, 54, "Runtime".ljust(15), curses.A_BOLD | curses.A_UNDERLINE)

        # Job rows
        for i, job in enumerate(display_jobs):
            row_y = header_y + 1 + i
            if row_y >= height - 2:
                break

            actual_idx = scroll_offset + i
            is_selected = actual_idx == selected_idx

            job_name = job.get("name", "").split("/")[-1]
            state = (job.get("status") or {}).get("state", "STATE_UNSPECIFIED")
            state_display = state.replace("STATE_", "").replace("_", " ").title()

            # Get runtime from job_to_row
            row_data = job_to_row(job)
            runtime = row_data[-1] if row_data else "-"

            # Color based on state
            color = curses.color_pair(0)
            if "RUNNING" in state.upper():
                color = curses.color_pair(1)
            elif "FAILED" in state.upper():
                color = curses.color_pair(2)
            elif "SUCCEEDED" in state.upper():
                color = curses.color_pair(3)
            elif "QUEUED" in state.upper() or "SCHEDULED" in state.upper():
                color = curses.color_pair(4)

            attrs = color
            if is_selected:
                attrs = curses.color_pair(5) | curses.A_BOLD

            # Render row
            stdscr.addstr(row_y, 0, job_name[:40].ljust(40), attrs)
            stdscr.addstr(row_y, 41, state_display[:12].ljust(12), attrs)
            stdscr.addstr(row_y, 54, runtime[:15].ljust(15), attrs)

        # Footer
        footer_y = height - 1
        footer = "Navigation: j/k or ↑↓ | Enter: Show Details | r: Refresh | q: Quit"
        stdscr.addstr(footer_y, 0, footer[:width-1], curses.A_DIM)

        stdscr.refresh()

        # Handle input
        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("r") or key == ord("R"):
            need_refresh = True  # Trigger reload on next iteration
        elif key == ord("j") or key == curses.KEY_DOWN:
            if selected_idx < len(jobs) - 1:
                selected_idx += 1
                if selected_idx >= scroll_offset + max_display_rows:
                    scroll_offset += 1
        elif key == ord("k") or key == curses.KEY_UP:
            if selected_idx > 0:
                selected_idx -= 1
                if selected_idx < scroll_offset:
                    scroll_offset -= 1
        elif key == ord("\n") or key == curses.KEY_ENTER or key == 10:
            # Show details for selected job (CPU for running, logs for failed)
            if 0 <= selected_idx < len(jobs):
                selected_job = jobs[selected_idx]
                state = (selected_job.get("status") or {}).get("state", "")

                if "FAILED" in state.upper():
                    show_failure_logs(stdscr, selected_job, project)
                elif "RUNNING" in state.upper():
                    show_cpu_info(stdscr, selected_job, project)
                else:
                    # For succeeded/other states, show basic info
                    show_cpu_info(stdscr, selected_job, project)


def main():
    """Main entry point for interactive mode."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Cloud Batch job viewer with keyboard navigation"
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="GCP project ID")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Batch region")
    parser.add_argument(
        "--limit", type=int, default=200, help="Maximum number of jobs to fetch"
    )
    args = parser.parse_args()

    try:
        curses.wrapper(interactive_table, args.project, args.location, args.limit)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
