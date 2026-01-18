#!/usr/bin/env python3
"""
Cloud Run Job processor - runs simulation pipeline stages
Triggered by Cloud Run Service, processes data from Cloud Storage
"""
import json
import os
import sys
from pathlib import Path

# Import processing functions from app.py
from app import (
    run_rust_code,
    run_visualization,
    run_mask_generation,
    update_progress_data,
    get_project_path,
)


def link_material_files(project_path: Path, material_type: str) -> None:
    """Copy material CSVs into rust_code_input (GCSFuse doesn't support symlinks)."""
    import shutil

    rust_input_path = project_path / "rust_code_input"
    rust_input_path.mkdir(parents=True, exist_ok=True)

    # Material files are already copied by Cloud Run UI during upload - skip copying
    # (Copying here with shutil.copy2 on GCSFuse creates 0-byte files)
    print(f"  Material files already uploaded by UI, skipping copy")

    # Print first line of each material file (works for both custom and pre-loaded)
    print(f"  Material files:")
    for filename in ("conductivity.csv", "sp_heat_cap_data.csv"):
        filepath = rust_input_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                print(f"    {filename}: {first_line}")

def main():
    """Main job processor entry point"""

    # Get job parameters from environment variables
    # These are set by the Cloud Run Service when triggering the job
    project_id = os.environ.get('PROJECT_ID')
    stage = os.environ.get('STAGE')
    timestep = float(os.environ.get('TIMESTEP', '1.0'))
    tg = float(os.environ.get('TG', '105.0'))
    dHigh = float(os.environ.get('DHIGH', '15.0'))
    dLow = float(os.environ.get('DLOW', '45.0'))
    time_s = float(os.environ.get('TIME_S', '-1.0'))
    material_type = os.environ.get('MATERIAL_TYPE', 'abs').strip().lower()

    if not project_id or not stage:
        print("ERROR: Missing required environment variables PROJECT_ID or STAGE")
        sys.exit(1)

    print(f"=" * 80)
    print(f"Cloud Run Job Processor Starting")
    print(f"Project ID: {project_id}")
    print(f"Stage: {stage}")
    print(f"Material Type: {material_type.upper()}")
    print(f"Timestep: {timestep}")
    print(f"Mask params: Tg={tg}, dHigh={dHigh}, dLow={dLow}, time_s={time_s}")
    print(f"=" * 80)

    # Verify project exists
    project_path = get_project_path(project_id)
    if not project_path.exists():
        print(f"ERROR: Project folder not found: {project_path}")
        sys.exit(1)

    # Ensure material files are in place (shared directory is mounted)
    link_material_files(project_path, material_type)

    # Update progress to show job is running
    update_progress_data(project_id, {
        "current": 0,
        "total": 0,
        "percentage": 0,
        "message": f"Job started - running {stage}",
        "job_status": "RUNNING",
        "stage": stage,
    })

    # Run the requested stage
    try:
        if stage == "rust_simulation":
            print(f"\n[{project_id}] Running Rust simulation...")
            success, message, duration = run_rust_code(project_id)

        elif stage == "visualization":
            print(f"\n[{project_id}] Running PyVista visualization (timestep={timestep}s)...")
            success, message, duration = run_visualization(
                project_id, timestep=timestep
            )

        elif stage == "mask_generation":
            print(f"\n[{project_id}] Running mask generation (Tg={tg}°C, ΔHigh={dHigh}°C, ΔLow={dLow}°C, time={time_s}s)...")
            success, message, duration = run_mask_generation(
                project_id, tg=tg, dHigh=dHigh, dLow=dLow, time_s=time_s
            )

        elif stage == "full_pipeline":
            print(f"\n[{project_id}] Running FULL pipeline...")

            # Run all stages sequentially
            stages = [
                ("rust_simulation", "Rust Simulation", run_rust_code, {}),
                ("visualization", "Visualization", run_visualization,
                 {"timestep": timestep}),
            ]

            for stage_key, stage_name, stage_func, kwargs in stages:
                print(f"\n{'=' * 80}")
                print(f"Starting: {stage_name}")
                print(f"{'=' * 80}")

                if kwargs:
                    success, message, duration = stage_func(project_id, **kwargs)
                else:
                    success, message, duration = stage_func(project_id)

                if not success:
                    print(f"ERROR: {stage_name} failed: {message}")
                    sys.exit(1)

                print(f"SUCCESS: {stage_name} completed in {duration:.2f}s")

            success = True
            message = "Full pipeline completed successfully"
            duration = 0  # Total duration tracked separately
        else:
            print(f"ERROR: Unknown stage: {stage}")
            sys.exit(1)

        # Report results
        print(f"\n{'=' * 80}")
        if success:
            print(f"✅ JOB COMPLETED SUCCESSFULLY")
            print(f"Stage: {stage}")
            print(f"Message: {message}")
            print(f"Duration: {duration:.2f}s")
            print(f"{'=' * 80}")

            # Update progress to show completion
            update_progress_data(project_id, {
                "current": 100,
                "total": 100,
                "percentage": 100,
                "message": f"Job completed successfully - {message}",
                "job_status": "SUCCEEDED",
                "stage": stage,
            })
            sys.exit(0)
        else:
            print(f"❌ JOB FAILED")
            print(f"Stage: {stage}")
            print(f"Error: {message}")
            print(f"Duration: {duration:.2f}s")
            print(f"{'=' * 80}")

            # Update progress to show failure
            update_progress_data(project_id, {
                "current": 0,
                "total": 0,
                "percentage": 0,
                "message": f"Job failed - {message}",
                "job_status": "FAILED",
                "stage": stage,
            })
            sys.exit(1)

    except Exception as e:
        import traceback
        print(f"\n{'=' * 80}")
        print(f"❌ EXCEPTION IN JOB PROCESSOR")
        print(f"{'=' * 80}")
        traceback.print_exc()

        # Update progress to show failure
        update_progress_data(project_id, {
            "current": 0,
            "total": 0,
            "percentage": 0,
            "message": f"Job crashed - {str(e)}",
            "job_status": "FAILED",
            "stage": stage,
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
