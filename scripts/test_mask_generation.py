#!/usr/bin/env python3
"""Minimal helper script to exercise `generate_mask_frame`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fea_pyvista import generate_mask_frame
from environment import PROJECTS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-off hot/cold mask generation for a project."
    )
    parser.add_argument(
        "project_id",
        help="Project folder to use (e.g. 001 or a UUID).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECTS_ROOT,
        help="Base projects directory (defaults to environment.PROJECTS_ROOT).",
    )
    parser.add_argument("--tg", type=float, default=105.0, help="Glass transition °C.")
    parser.add_argument(
        "--dhigh",
        type=float,
        default=15.0,
        help="Hot threshold delta above Tg (°C).",
    )
    parser.add_argument(
        "--dlow",
        type=float,
        default=45.0,
        help="Cold threshold delta below Tg (°C).",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=-1.0,
        help="Time in seconds to visualize (-1 uses the latest timestep).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projects_root = args.root.expanduser().resolve()
    project_path = projects_root / args.project_id

    if not project_path.exists():
        print(f"Project folder not found: {project_path}", file=sys.stderr)
        return 1

    print(f"Using project folder: {project_path}")
    output_path = generate_mask_frame(
        args.project_id,
        str(projects_root),
        tg=args.tg,
        dHigh=args.dhigh,
        dLow=args.dlow,
        time_s=args.time,
    )

    if output_path is None:
        print("Mask generation finished but returned no file (no valid cells?).")
    else:
        output_path = Path(output_path)
        print(f"Mask artifacts written under: {output_path.parent}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
