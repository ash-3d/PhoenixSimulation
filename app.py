import errno
import gc
import json
import os
import random
import shutil
import string
import subprocess
import time
import traceback
import zipfile
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, session, redirect, url_for
from authlib.integrations.flask_client import OAuth

# Cloud Batch client (for triggering processing jobs)
from google.api_core import exceptions as google_exceptions
from google.auth import compute_engine
from google.auth.transport import requests as google_requests
from google.cloud import batch_v1, storage
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

from environment import (
    ALLOWED_EXTENSIONS,
    BATCH_CONFIG,
    BATCH_GCS_BUCKET,
    BATCH_MAX_DURATION,
    BATCH_REGION,
    ENV,
    ENV_TARGETS,
    IS_CLOUD,
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    PIPELINE_FOLDERS,
    PROJECTS_ROOT,
    SECRET_KEY,
    get_project_path,
)
from fea_pyvista import generate_frames, generate_mask_frame

VERSION = "0.0.1"

CLOUD_BATCH_ENABLED = True
app = Flask(__name__)

# Trust X-Forwarded-Proto headers (needed for correct HTTPS redirects on Cloud Run)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure Flask session and OAuth
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_SECURE"] = IS_CLOUD  # HTTPS only in production
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Initialize OAuth
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=OAUTH_CLIENT_ID,
    client_secret=OAUTH_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Default processing parameters (used across UI, backend, and job processor)
DEFAULT_TIMESTEP = 1.0
DEFAULT_TG = 105.0
DEFAULT_DHIGH = 15.0
DEFAULT_DLOW = 45.0
DEFAULT_TIME_S = -1.0
DEFAULT_MATERIAL_TYPE = "abs"

# Configuration
# Use /app/projects in container, ./projects locally.
UPLOAD_FOLDER = PROJECTS_ROOT
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)  # Flask expects string
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size


def ensure_fd_headroom(min_soft_limit=8192):
    """Raise RLIMIT_NOFILE soft limit when possible to avoid EMFILE errors."""
    try:
        import resource
    except ImportError:
        return False

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = max(soft, min_soft_limit)
        hard_cap = hard

        if hard_cap != resource.RLIM_INFINITY:
            desired = min(desired, hard_cap)

        if desired <= soft:
            return False

        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard_cap))
        print(
            f"[startup] Increased RLIMIT_NOFILE from {soft} to {desired} "
            f"(hard limit: {'unlimited' if hard_cap == resource.RLIM_INFINITY else hard_cap})"
        )
        return True
    except (ValueError, OSError):
        return False


# Give the process plenty of file descriptors up front. Cloud Run defaults to 1024.
ensure_fd_headroom(int(os.environ.get("MIN_FD_LIMIT", "8192")))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_project_id():
    """Generate a random 4-character alphanumeric project ID"""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=4))


def save_timing_data(project_id, stage, duration_seconds):
    """Save timing data for a pipeline stage"""
    project_path = get_project_path(project_id)
    timing_file = project_path / "timing.json"

    timing_data = {}
    if timing_file.exists():
        try:
            with open(timing_file, "r") as f:
                timing_data = json.load(f)
        except Exception:
            pass

    timing_data[stage] = {
        "duration_seconds": round(duration_seconds, 2),
        "duration_formatted": format_duration(duration_seconds),
        "timestamp": datetime.now().isoformat(),
    }

    try:
        with open(timing_file, "w") as f:
            json.dump(timing_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save timing data: {e}")


def get_timing_data(project_id):
    """Get timing data for a project"""
    project_path = get_project_path(project_id)
    timing_file = project_path / "timing.json"

    if not timing_file.exists():
        return {}

    try:
        with open(timing_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_gcp_project_id():
    """Get GCP project ID from environment or metadata server"""
    gcp_project_id = os.environ.get("GCP_PROJECT") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT"
    )
    if gcp_project_id:
        return gcp_project_id

    # Try to get from metadata server
    try:
        import requests

        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
            timeout=1,
        )
        return response.text
    except Exception:
        return None


def save_uploaded_files(request_files, target_dir, file_mapping):
    """Save uploaded files from request to target directory

    Args:
        request_files: request.files object
        target_dir: Path object for target directory
        file_mapping: dict mapping request file keys to target filenames

    Returns:
        list of saved filenames

    Raises:
        ValueError: if invalid file format
    """
    saved = []
    for file_key, target_filename in file_mapping.items():
        if file_key in request_files and request_files[file_key].filename:
            file = request_files[file_key]
            if file and allowed_file(file.filename):
                file.save(target_dir / target_filename)
                saved.append(target_filename)
            else:
                raise ValueError(
                    f"Invalid file format for {file_key.replace('_', ' ').title()}"
                )
    return saved


def login_required(f):
    """Decorator to require Google authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip auth check if OAuth is not configured (local dev)
        if not OAUTH_CLIENT_ID or not OAUTH_CLIENT_SECRET:
            return f(*args, **kwargs)

        if "user" not in session:
            # For API endpoints, return JSON error
            if request.path.startswith("/project/") or request.path.startswith("/upload"):
                return jsonify({"error": "Authentication required"}), 401
            # For HTML pages, redirect to login
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def validate_project_id(check_exists=False):
    """Decorator to validate and sanitize project_id parameter

    Args:
        check_exists: If True, verify project directory exists

    Returns:
        Decorated function with sanitized project_id
    """

    def decorator(f):
        @wraps(f)
        def wrapper(project_id, *args, **kwargs):
            # Sanitize project_id
            safe_project_id = secure_filename(project_id)

            # Check if project exists
            if check_exists:
                project_path = get_project_path(safe_project_id)
                if not project_path.exists():
                    return jsonify(
                        {"success": False, "error": "Project not found"}
                    ), 404

            # Call original function with sanitized project_id
            return f(safe_project_id, *args, **kwargs)

        return wrapper

    return decorator


def save_error_data(project_id, stage, error_message, duration_seconds=0):
    """Save error data for a failed pipeline stage

    Args:
        project_id: The project ID
        stage: The stage that failed
        error_message: The error message to save
        duration_seconds: How long the stage ran before failing
    """
    project_path = get_project_path(project_id)
    error_file = project_path / "error.json"

    error_data = {
        "stage": stage,
        "error": error_message,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration_seconds, 2),
    }

    try:
        with open(error_file, "w") as f:
            json.dump(error_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save error data: {e}")


def get_error_data(project_id):
    """Get error data for a project

    Args:
        project_id: The project ID

    Returns:
        dict: Error data with keys: stage, error, timestamp, duration_seconds
        None if no error file exists
    """
    project_path = get_project_path(project_id)
    error_file = project_path / "error.json"

    if not error_file.exists():
        return None

    try:
        with open(error_file, "r") as f:
            return json.load(f)
    except Exception:
        return None


def clear_error_data(project_id):
    """Clear error data for a project when starting a new stage"""
    project_path = get_project_path(project_id)
    error_file = project_path / "error.json"

    if error_file.exists():
        try:
            error_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clear error data: {e}")


def save_job_data(project_id, stage, job_id, job_location):
    """Save Cloud Batch job information for status tracking

    Args:
        project_id: The project ID
        stage: The pipeline stage being executed
        job_id: The Cloud Batch job ID
        job_location: The full job resource location
    """
    project_path = get_project_path(project_id)
    job_file = project_path / "job_status.json"

    job_data = {
        "job_id": job_id,
        "job_location": job_location,
        "stage": stage,
        "status": "SUBMITTED",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save job data: {e}")


def get_job_data(project_id):
    """Get Cloud Batch job information for a project

    Args:
        project_id: The project ID

    Returns:
        dict: Job data with keys: job_id, job_location, stage, status, timestamp
        None if no job file exists
    """
    project_path = get_project_path(project_id)
    job_file = project_path / "job_status.json"

    if not job_file.exists():
        return None

    try:
        with open(job_file, "r") as f:
            return json.load(f)
    except Exception:
        return None


def clear_job_data(project_id):
    """Clear job tracking data when a job completes or fails"""
    project_path = get_project_path(project_id)
    job_file = project_path / "job_status.json"

    if job_file.exists():
        try:
            job_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clear job data: {e}")


def update_progress_data(project_id, progress_data):
    """Update progress data for a project

    Args:
        project_id: The project ID
        progress_data: dict with progress information
    """
    if IS_CLOUD:
        storage_client = None
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(BATCH_GCS_BUCKET)
            blob = bucket.blob(f"{project_id}/progress.json")
            blob.upload_from_string(json.dumps(progress_data))
        except Exception as e:
            print(f"Warning: Failed to update progress data: {e}")
        finally:
            # Explicitly close storage client to release file descriptors
            if storage_client:
                try:
                    storage_client.close()
                except Exception:
                    pass
    else:
        # Local development mode
        project_path = get_project_path(project_id)
        progress_file = project_path / "progress.json"
        try:
            with open(progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to update progress data: {e}")


def get_progress_data(project_id):
    """Get progress data for a project from cloud storage or local filesystem

    Args:
        project_id: The project ID

    Returns:
        dict: Progress data with keys: current, total, percentage, message, job_status, stage
    """
    default_progress = {
        "current": 0,
        "total": 0,
        "percentage": 0,
        "message": "Not started",
        "job_status": "NOT_STARTED",
        "stage": None,
    }

    if IS_CLOUD:
        storage_client = None
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(BATCH_GCS_BUCKET)
            blob = bucket.blob(f"{project_id}/progress.json")

            if not blob.exists():
                return default_progress

            result = json.loads(blob.download_as_string())
            return result
        except Exception as e:
            raise Exception(f"GCS Read Error: {str(e)}")
        finally:
            # Explicitly close storage client to release file descriptors
            if storage_client:
                try:
                    storage_client.close()
                except Exception:
                    pass
    else:
        # Local development mode
        project_path = get_project_path(project_id)
        progress_file = project_path / "progress.json"

        if not project_path.exists():
            raise FileNotFoundError("Project not found")

        if not progress_file.exists():
            return default_progress

        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except Exception as e:
            raise Exception(str(e))


def get_project_config(project_id):
    """Get project configuration (filenames, material type)"""
    project_path = get_project_path(project_id)
    config_file = project_path / "project_config.json"
    
    if not config_file.exists():
        return {}
        
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read project config: {e}")
        return {}


def update_project_config(project_id, new_data):
    """Update project configuration with new data"""
    current_config = get_project_config(project_id)
    current_config.update(new_data)
    
    project_path = get_project_path(project_id)
    config_file = project_path / "project_config.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(current_config, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save project config: {e}")


def execute_pipeline_stage(
    project_id,
    stage,
    timestep=None,
    mask_params=None,
    material_type=None,
):
    """Execute a pipeline stage and return appropriate response

    Args:
        project_id: The project ID
        stage: The stage to execute
        timestep: Optional timestep parameter (defaults to 1.0)
        mask_params: Optional dict with mask parameters (tg, dHigh, dLow, time_s)
        material_type: Optional material type (defaults to 'abs')

    Returns:
        tuple: (response_dict, status_code)
    """
    if timestep is None or mask_params is None or material_type is None:
        try:
            data = request.get_json() or {}
            if timestep is None:
                timestep = float(data.get("timestep", DEFAULT_TIMESTEP))
            if mask_params is None:
                mask_params = {
                    "tg": float(data.get("tg", DEFAULT_TG)),
                    "dHigh": float(data.get("dHigh", DEFAULT_DHIGH)),
                    "dLow": float(data.get("dLow", DEFAULT_DLOW)),
                    "time_s": float(data.get("time_s", DEFAULT_TIME_S)),
                }
            if material_type is None:
                material_type = data.get("material_type", DEFAULT_MATERIAL_TYPE)
        except Exception:
            pass

    # If material_type is still None, try to read from project_config.json
    if material_type is None:
        project_config = get_project_config(project_id)
        material_type = project_config.get("material_type", DEFAULT_MATERIAL_TYPE)

    # Set defaults for timestep and mask_params if still missing
    if timestep is None: timestep = DEFAULT_TIMESTEP
    if mask_params is None:
         mask_params = {
            "tg": DEFAULT_TG,
            "dHigh": DEFAULT_DHIGH,
            "dLow": DEFAULT_DLOW,
            "time_s": DEFAULT_TIME_S,
        }
    
    print(f"[{project_id}] Starting pipeline stage: {stage}")
    print(f"[{project_id}] Material type: {material_type}")

    try:
        success, message, duration = trigger_cloud_batch_job(
            project_id,
            stage,
            timestep=timestep,
            mask_params=mask_params,
            material_type=material_type,
        )

        if success:
            print(f"[{project_id}] ✅ Stage {stage} completed successfully")
            return {
                "success": True,
                "message": message,
                "duration": format_duration(duration),
            }, 200
        else:
            print(f"[{project_id}] ❌ Stage {stage} failed: {message}")
            return {
                "success": False,
                "error": message,
                "duration": format_duration(duration),
            }, 500

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"[{project_id}] ❌ EXCEPTION in stage {stage}:")
        print(traceback_str)
        return {"success": False, "error": f"Unexpected error: {error_msg}"}, 500


def trigger_cloud_batch_job(
    project_id,
    stage,
    timestep=DEFAULT_TIMESTEP,
    mask_params=None,
    material_type=DEFAULT_MATERIAL_TYPE,
):
    """
    Trigger a Cloud Batch job to process a stage
    Uses stage-specific machine configurations (CPU for Rust, GPU for visualization)
    Returns: (success, message/execution_name, duration)
    """
    if mask_params is None:
        mask_params = {
            "tg": DEFAULT_TG,
            "dHigh": DEFAULT_DHIGH,
            "dLow": DEFAULT_DLOW,
            "time_s": DEFAULT_TIME_S,
        }

    if not CLOUD_BATCH_ENABLED:
        # Fall back to local processing (for development)
        print(f"[{project_id}] Cloud Batch disabled - running locally")
        if stage == "rust_simulation":
            return run_rust_code(project_id)
        elif stage == "visualization":
            return run_visualization(
                project_id,
                timestep=timestep,
            )
        elif stage == "mask_generation":
            return run_mask_generation(
                project_id,
                tg=mask_params["tg"],
                dHigh=mask_params["dHigh"],
                dLow=mask_params["dLow"],
                time_s=mask_params["time_s"],
            )

    try:
        # Get stage-specific configuration
        if stage not in BATCH_CONFIG:
            return (
                False,
                f"Invalid stage: {stage}. Valid stages: {list(BATCH_CONFIG.keys())}",
                0,
            )

        stage_config = BATCH_CONFIG[stage]
        machine_type = stage_config.machine_type
        cpu_count = stage_config.cpu_count
        memory_gb = stage_config.memory_gb
        gpu_type = stage_config.gpu_type
        gpu_count = stage_config.gpu_count

        # Get GCP project ID
        gcp_project_id = get_gcp_project_id()
        if not gcp_project_id:
            return False, "Could not determine GCP project ID", 0

        region = BATCH_REGION

        print(f"[{project_id}] Triggering Cloud Batch Job")
        print(f"  GCP Project: {gcp_project_id}")
        print(f"  Region: {region}")
        print(f"  Stage: {stage}")
        print(f"  Timestep: {timestep}")
        gpu_info = f", {gpu_count}x {gpu_type}" if gpu_count > 0 else ""
        print(
            f"  Machine: {machine_type} ({cpu_count} vCPUs, {memory_gb}GB RAM{gpu_info})"
        )
        print(f"  Description: {stage_config.description}")

        # Create Cloud Batch client (will be closed in finally block)
        client = None
        client = batch_v1.BatchServiceClient()

        # Create job configuration
        job = batch_v1.Job()

        # Generate unique job ID
        safe_project_id = project_id.lower().replace("_", "-")
        safe_stage = stage.lower().replace("_", "-")
        job_id = (
            f"{ENV_TARGETS.batch_job_prefix}-{safe_project_id}-"
            f"{safe_stage}-{int(time.time())}"
        )

        # Configure task
        runnable = batch_v1.Runnable()
        runnable.container = batch_v1.Runnable.Container()
        runnable.container.image_uri = f"gcr.io/{gcp_project_id}/citb4:latest"
        runnable.container.entrypoint = "/bin/bash"
        runnable.container.commands = ["-c", "python job_processor.py"]
        # Bind mount GCS volume into container
        runnable.container.volumes = ["/mnt/disks/gcs:/app/projects:rw"]

        # Set environment variables
        runnable.environment = batch_v1.Environment()
        env_vars = {
            "MODE": "processor",
            "PROJECT_ID": project_id,
            "STAGE": stage,
            "MATERIAL_TYPE": material_type,
            "TIMESTEP": str(timestep),
            "TG": str(mask_params["tg"]),
            "DHIGH": str(mask_params["dHigh"]),
            "DLOW": str(mask_params["dLow"]),
            "TIME_S": str(mask_params["time_s"]),
            "GCP_PROJECT": gcp_project_id,
            "ENV": ENV,
        }

        # Enable GPU rendering for GPU-enabled stages
        # Check both explicit GPU count and machine types with bundled GPUs (a2-ultragpu, a3-highgpu, g2-standard families)
        has_gpu = (
            gpu_count > 0
            or machine_type.startswith("a2-ultragpu")
            or machine_type.startswith("a3-highgpu")
            or machine_type.startswith("g2-standard")
        )
        if has_gpu:
            env_vars["USE_GPU"] = "1"
            env_vars["NVIDIA_VISIBLE_DEVICES"] = "all"
            env_vars["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility,graphics"
            gpu_source = (
                f"bundled with {machine_type}"
                if (
                    machine_type.startswith("a2-ultragpu")
                    or machine_type.startswith("a3-highgpu")
                    or machine_type.startswith("g2-standard")
                )
                else "separate accelerator"
            )
            print(f"  GPU Rendering: ENABLED ({gpu_source})")
        else:
            print("  GPU Rendering: DISABLED (CPU-only)")

        runnable.environment.variables = env_vars

        # Configure task spec
        task = batch_v1.TaskSpec()
        task.runnables = [runnable]

        # CPU/Memory configuration
        task.compute_resource = batch_v1.ComputeResource()
        task.compute_resource.cpu_milli = cpu_count * 1000  # vCPUs to milli-CPUs
        task.compute_resource.memory_mib = memory_gb * 1024  # GB to MiB

        # Max run duration
        task.max_run_duration = BATCH_MAX_DURATION

        # Mount Cloud Storage bucket for persistent data
        # Use /mnt/disks/gcs (writable on Container-Optimized OS)
        gcs_volume = batch_v1.Volume()
        gcs_volume.gcs = batch_v1.GCS()
        gcs_volume.gcs.remote_path = BATCH_GCS_BUCKET  # No trailing slash!
        gcs_volume.mount_path = "/mnt/disks/gcs"
        task.volumes = [gcs_volume]

        # Group tasks
        group = batch_v1.TaskGroup()
        group.task_count = 1
        group.task_spec = task

        job.task_groups = [group]

        # Configure allocation policy (PROD uses standard, DEV stays on spot for cost)
        policy = batch_v1.AllocationPolicy.InstancePolicy()
        policy.machine_type = machine_type
        if ENV == "PROD":
            policy_model = batch_v1.AllocationPolicy.ProvisioningModel.STANDARD
        else:
            policy_model = batch_v1.AllocationPolicy.ProvisioningModel.SPOT
        policy.provisioning_model = policy_model
        print(
            "  Provisioning Model: "
            + ("STANDARD (non-preemptible)" if ENV == "PROD" else "SPOT (preemptible)")
        )

        # Add GPU accelerators if required (not needed for machine types with bundled GPUs)
        if gpu_count > 0:
            accelerator = batch_v1.AllocationPolicy.Accelerator()
            accelerator.type_ = gpu_type
            accelerator.count = gpu_count
            policy.accelerators = [accelerator]
            print(f"  GPU Config: {gpu_count}x {gpu_type}")

        instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
        instances.policy = policy

        job.allocation_policy = batch_v1.AllocationPolicy()
        job.allocation_policy.instances = [instances]

        # Configure logs
        job.logs_policy = batch_v1.LogsPolicy()
        job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

        # Labels for tracking
        job.labels = {
            "project-id": safe_project_id,
            "stage": safe_stage,
        }

        # Create the job
        create_request = batch_v1.CreateJobRequest(
            parent=f"projects/{gcp_project_id}/locations/{region}",
            job_id=job_id,
            job=job,
        )

        print(f"[{project_id}] Submitting Cloud Batch job: {job_id}")
        operation = client.create_job(create_request)

        job_name = operation.name

        print(f"[{project_id}] Cloud Batch Job submitted successfully")
        print(f"  Job: {job_name}")
        print(f"  View logs: gcloud batch jobs describe {job_id} --location {region}")

        # Save job information for status tracking
        job_location = f"projects/{gcp_project_id}/locations/{region}/jobs/{job_id}"
        save_job_data(project_id, stage, job_id, job_location)

        # Initialize progress data to show job is submitted
        update_progress_data(
            project_id,
            {
                "current": 0,
                "total": 0,
                "percentage": 0,
                "message": f"Cloud Batch job submitted (ID: {job_id})",
                "job_status": "SUBMITTED",
                "stage": stage,
                "job_id": job_id,
            },
        )

        return True, f"Batch job started: {job_id}", 0

    except google_exceptions.NotFound as e:
        error_msg = f"Cloud Batch service or resources not found: {str(e)}"
        print(f"[{project_id}] ERROR: {error_msg}")
        return False, error_msg, 0
    except Exception as e:
        traceback.print_exc()
        error_msg = f"Failed to trigger Cloud Batch job: {str(e)}"
        print(f"[{project_id}] ERROR: {error_msg}")
        return False, error_msg, 0
    finally:
        # Explicitly close client to release file descriptors
        if client:
            try:
                client.transport.close()
            except Exception:
                pass


def get_des_version():
    """Get the current commit hash of the DES thermal simulation submodule"""
    try:
        script_dir = Path(__file__).parent.resolve()

        # Try to read from version file (used in Docker container)
        version_file = script_dir / "DES_VERSION.txt"
        if version_file.exists():
            return version_file.read_text().strip()

        # Fall back to git command (for local development)
        des_dir = script_dir / "DES_thermal_simulation"
        if not des_dir.exists():
            return "not found"

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(des_dir),
            text=True,
            capture_output=True,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown"
    except Exception:
        return "error"


def get_project_folders():
    """Get list of project folders"""
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    return [f.name for f in UPLOAD_FOLDER.iterdir() if f.is_dir()]


def get_project_status(project_id):
    """Get the status of each pipeline stage for a project"""
    project_path = get_project_path(project_id)
    status = {}

    timing_data = get_timing_data(project_id)

    # Check for error and job status
    error_data = get_error_data(project_id)
    job_data = get_job_data(project_id)

    # Map folders to their corresponding stage names
    folder_to_stage = {
        "frame_generator_input": "rust_simulation",
        "frames_filtered_active_only": "visualization",
    }

    for folder, description in PIPELINE_FOLDERS.items():
        folder_path = project_path / folder
        stage_name = folder_to_stage.get(folder, folder)

        # Determine stage state
        if folder_path.exists() and len(list(folder_path.iterdir())) > 0:
            # Stage completed successfully
            files = [f.name for f in folder_path.iterdir()]
            status[folder] = {
                "state": "succeeded",
                "exists": True,
                "files": files,
                "count": len(files),
                "description": description,
            }
        elif error_data and error_data.get("stage") == stage_name:
            # Stage failed
            status[folder] = {
                "state": "failed",
                "exists": False,
                "files": [],
                "count": 0,
                "description": description,
            }
        elif job_data and job_data.get("stage") == stage_name:
            # Stage is currently running
            status[folder] = {
                "state": "running",
                "exists": False,
                "files": [],
                "count": 0,
                "description": description,
            }
        else:
            # Stage not run yet
            status[folder] = {
                "state": "not_run",
                "exists": False,
                "files": [],
                "count": 0,
                "description": description,
            }

    status["timing"] = timing_data

    # Check for animation file at project root
    animation_file = project_path / "fea_animation.mp4"
    if animation_file.exists():
        status["animation"] = {
            "state": "succeeded",
            "exists": True,
            "path": "fea_animation.mp4",
        }
    elif error_data and error_data.get("stage") == "visualization":
        status["animation"] = {"state": "failed", "exists": False}
    elif job_data and job_data.get("stage") == "visualization":
        status["animation"] = {"state": "running", "exists": False}
    else:
        status["animation"] = {"state": "not_run", "exists": False}

    # Check for mask files at project root
    mask_png = project_path / "hot_cold_mask.png"
    mask_3mf = project_path / "hot_cold_mask.3mf"
    has_masks = mask_png.exists() or mask_3mf.exists()

    if has_masks:
        mask_files = []
        if mask_png.exists():
            mask_files.append("hot_cold_mask.png")
        if mask_3mf.exists():
            mask_files.append("hot_cold_mask.3mf")
        status["masks"] = {
            "state": "succeeded",
            "exists": True,
            "files": mask_files,
            "count": len(mask_files),
        }
    elif error_data and error_data.get("stage") == "mask_generation":
        status["masks"] = {"state": "failed", "exists": False, "files": [], "count": 0}
    elif job_data and job_data.get("stage") == "mask_generation":
        status["masks"] = {"state": "running", "exists": False, "files": [], "count": 0}
    else:
        status["masks"] = {"state": "not_run", "exists": False, "files": [], "count": 0}

    # Get animation file size from progress.json
    try:
        progress_data = get_progress_data(project_id)
        status["animation_file_size_formatted"] = progress_data.get(
            "animation_file_size_formatted"
        )
    except Exception:
        status["animation_file_size_formatted"] = None

    return status


def run_rust_code(project_id):
    """Run Rust thermal simulation code (uses standard filenames: Input_file.txt, wall.gcode)"""
    start_time = time.time()
    try:
        project_path = get_project_path(project_id)
        rust_input_path = project_path / "rust_code_input"
        frame_input_path = project_path / "frame_generator_input"

        required_files = [
            "Input_file.txt",
            "wall.gcode",
            "conductivity.csv",
            "sp_heat_cap_data.csv",
        ]

        # Check for all required files
        for file in required_files:
            if not (rust_input_path / file).exists():
                return False, f"Missing input file: {file}", 0

        frame_input_path.mkdir(parents=True, exist_ok=True)

        script_dir = Path(__file__).parent.resolve()
        rust_code_dir = script_dir / "DES_thermal_simulation"

        rust_output_dir = rust_code_dir / "outputfiles"
        rust_output_dir.mkdir(parents=True, exist_ok=True)

        rust_input_dir = rust_code_dir / "inputfiles"
        rust_input_dir.mkdir(parents=True, exist_ok=True)

        # Copy all required files (already have standard names)
        for file in required_files:
            shutil.copy2(rust_input_path / file, rust_input_dir / file)

        print(f"[{project_id}] Starting Rust simulation...")
        if Path("/app").exists():
            # In container - use pre-built release binary
            rust_binary = (
                rust_code_dir / "target" / "release" / "DES_thermal_simulation"
            )
            result = subprocess.run(
                [str(rust_binary)],
                cwd=str(rust_code_dir),
                text=True,
                capture_output=True,
            )
        else:
            # Local development - use cargo run
            result = subprocess.run(
                ["cargo", "run", "--release"],
                cwd=str(rust_code_dir),
                text=True,
                capture_output=True,
            )

        # Print output to console (captured by Cloud Run logs)
        print(
            f"[{project_id}] Rust simulation completed with exit code {result.returncode}"
        )
        if result.stdout:
            print(f"[{project_id}] STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"[{project_id}] STDERR:\n{result.stderr}")

        if result.returncode != 0:
            duration = time.time() - start_time
            # Create detailed error message with full stderr/stdout
            error_details = (
                f"Rust execution failed (exit code {result.returncode}).\n\n"
            )
            if result.stderr:
                error_details += f"STDERR:\n{result.stderr}\n\n"
            if result.stdout:
                error_details += f"STDOUT:\n{result.stdout}"

            # Save full error to error.json
            save_error_data(project_id, "rust_simulation", error_details, duration)

            # Return shortened preview for API response
            error_preview = (result.stderr or result.stdout or "No error output")[:300]
            return (
                False,
                f"Rust execution failed (exit code {result.returncode}). Error: {error_preview}",
                duration,
            )

        rust_output_dir = rust_code_dir / "outputfiles"
        output_files = [
            "nodefile.csv",
            "elementfile.csv",
            "node_temps.csv",
            "activation_times.csv",
            "elem_temps.csv",
        ]

        missing_files = []
        for file in output_files:
            src_file = rust_output_dir / file
            dst_file = frame_input_path / file
            if src_file.exists():
                shutil.move(str(src_file), str(dst_file))
            else:
                missing_files.append(file)

        if missing_files:
            duration = time.time() - start_time
            error_msg = (
                f"Rust completed but missing output files: {', '.join(missing_files)}"
            )
            save_error_data(project_id, "rust_simulation", error_msg, duration)
            return (
                False,
                error_msg,
                duration,
            )

        duration = time.time() - start_time
        save_timing_data(project_id, "rust_simulation", duration)
        clear_error_data(project_id)  # Clear any previous errors on success
        print(
            f"[{project_id}] ✓ Rust simulation completed successfully in {duration:.2f}s"
        )
        return True, "Rust code execution completed successfully", duration

    except Exception as e:
        traceback.print_exc()
        duration = time.time() - start_time
        error_msg = f"Error running Rust code: {str(e)}\n\n{traceback.format_exc()}"
        save_error_data(project_id, "rust_simulation", error_msg, duration)
        return False, f"Error running Rust code: {str(e)}", duration


def run_visualization(
    project_id,
    timestep=DEFAULT_TIMESTEP,
):
    """Run the visualization stage (frames + animation output)"""
    start_time = time.time()
    try:
        project_path = get_project_path(project_id)
        frame_input_path = project_path / "frame_generator_input"
        frames_output_path = project_path / "frames_filtered_active_only"

        # Step 1: Validate input files
        required_files = [
            "nodefile.csv",
            "elementfile.csv",
            "node_temps.csv",
            "activation_times.csv",
        ]
        for file in required_files:
            if not (frame_input_path / file).exists():
                error_msg = f"Missing FEA file: {file}"
                save_error_data(project_id, "visualization", error_msg, 0)
                return False, error_msg, 0

        frames_output_path.mkdir(parents=True, exist_ok=True)

        # Step 2: Generate frames
        print(f"[{project_id}] Generating frames (timestep={timestep}s)...")
        generate_frames(project_id, timestep=timestep)

        frames = list(frames_output_path.iterdir())
        if not frames:
            duration = time.time() - start_time
            error_msg = "No frames were generated"
            save_error_data(project_id, "visualization", error_msg, duration)
            return False, error_msg, duration

        print(f"[{project_id}] Generated {len(frames)} frames successfully")

        # Step 3: Create animation from frames
        print(f"[{project_id}] Creating animation from {len(frames)} frames...")
        subprocess.run(
            f'echo "{project_id}" | python make_animation.py',
            shell=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        video_path = project_path / "fea_animation.mp4"
        if not video_path.exists():
            duration = time.time() - start_time
            error_msg = (
                f"Generated {len(frames)} frames but animation file was not created"
            )
            save_error_data(project_id, "visualization", error_msg, duration)
            return (
                False,
                error_msg,
                duration,
            )

        # Success! Save timing for the merged stage
        duration = time.time() - start_time
        save_timing_data(project_id, "visualization", duration)
        clear_error_data(project_id)  # Clear any previous errors on success
        return (
            True,
            f"Visualization completed: {len(frames)} frames + animation (timestep: {timestep}s)",
            duration,
        )

    except Exception as e:
        traceback.print_exc()
        duration = time.time() - start_time
        error_msg = (
            f"Error during visualization stage: {str(e)}\n\n{traceback.format_exc()}"
        )
        save_error_data(project_id, "visualization", error_msg, duration)
        return False, f"Error during visualization stage: {str(e)}", duration


def run_mask_generation(
    project_id,
    tg=DEFAULT_TG,
    dHigh=DEFAULT_DHIGH,
    dLow=DEFAULT_DLOW,
    time_s=DEFAULT_TIME_S,
):
    """Run the mask generation stage (hot/cold mask visualization)"""
    start_time = time.time()
    try:
        project_path = get_project_path(project_id)
        frame_input_path = project_path / "frame_generator_input"

        # Step 1: Validate input files
        required_files = [
            "nodefile.csv",
            "elementfile.csv",
            "node_temps.csv",
            "activation_times.csv",
        ]
        for file in required_files:
            if not (frame_input_path / file).exists():
                error_msg = f"Missing FEA file: {file}"
                save_error_data(project_id, "mask_generation", error_msg, 0)
                return False, error_msg, 0

        # Step 2: Generate hot/cold mask visualization
        print(
            f"[{project_id}] Generating hot/cold mask (Tg={tg}°C, ΔHigh={dHigh}°C, ΔLow={dLow}°C, time={time_s}s)..."
        )
        mask_path = generate_mask_frame(
            project_id, tg=tg, dHigh=dHigh, dLow=dLow, time_s=time_s
        )

        if not mask_path:
            duration = time.time() - start_time
            error_msg = "Mask generation returned None"
            save_error_data(project_id, "mask_generation", error_msg, duration)
            return False, error_msg, duration

        print(
            f"[{project_id}] Mask visualization generated successfully: {mask_path}"
        )

        # Success!
        duration = time.time() - start_time
        save_timing_data(project_id, "mask_generation", duration)
        clear_error_data(project_id)  # Clear any previous errors on success
        return (
            True,
            f"Mask generation completed (Tg={tg}°C, ΔHigh={dHigh}°C, ΔLow={dLow}°C)",
            duration,
        )

    except Exception as e:
        traceback.print_exc()
        duration = time.time() - start_time
        error_msg = (
            f"Error during mask generation stage: {str(e)}\n\n{traceback.format_exc()}"
        )
        save_error_data(project_id, "mask_generation", error_msg, duration)
        return False, f"Error during mask generation stage: {str(e)}", duration


@app.route("/")
@login_required
def index():
    projects = get_project_folders()
    random_id = generate_project_id()
    des_version = get_des_version()
    user_email = session.get("user", {}).get("email", "Unknown")
    return render_template(
        "index.html",
        projects=projects,
        random_project_id=random_id,
        des_version=des_version,
        version=VERSION,
        is_dev=(ENV == "DEV"),
        environment=ENV,
        user_email=user_email,
    )


@app.route("/login")
def login():
    """Initiate Google OAuth login"""
    # Skip OAuth if not configured (local dev without credentials)
    if not OAUTH_CLIENT_ID or not OAUTH_CLIENT_SECRET:
        return jsonify({"error": "OAuth not configured. Set OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET environment variables."}), 500

    # Force HTTPS for Cloud Run (Cloud Run always serves over HTTPS)
    redirect_uri = url_for("authorize", _external=True, _scheme='https')
    return google.authorize_redirect(redirect_uri)


@app.route("/authorize")
def authorize():
    """OAuth callback route"""
    try:
        token = google.authorize_access_token()
        user_info = token.get("userinfo")

        if user_info:
            session["user"] = {
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "picture": user_info.get("picture"),
            }
            print(f"[AUTH] User logged in: {user_info.get('email')}")

        return redirect(url_for("index"))
    except Exception as e:
        print(f"[AUTH] Login failed: {e}")
        return jsonify({"error": f"Authentication failed: {str(e)}"}), 401


@app.route("/logout")
def logout():
    """Logout and clear session"""
    user_email = session.get("user", {}).get("email", "Unknown")
    session.pop("user", None)
    print(f"[AUTH] User logged out: {user_email}")
    return redirect(url_for("login"))


@app.route("/get-upload-url", methods=["POST"])
@login_required
def get_upload_url():
    """Generate signed URL for direct GCS upload (bypasses Cloud Run 32MB limit)"""
    try:
        data = request.get_json()
        project_id = data.get("project_id", "").strip()
        filename = data.get("filename", "").strip()
        file_type = data.get("file_type", "").strip()

        if not project_id or not filename or not file_type:
            return jsonify(
                {"error": "project_id, filename, and file_type are required"}
            ), 400

        project_id = secure_filename(project_id)
        safe_filename = secure_filename(filename)

        if not allowed_file(safe_filename):
            return jsonify({"error": f"File type not allowed: {safe_filename}"}), 400

        # Map file types to standard filenames
        # All files are renamed to standard names expected by Rust simulation
        file_mapping = {
            "config": "Input_file.txt",
            "gcode": "wall.gcode",
            "conductivity": "conductivity.csv",
            "heat_capacity": "sp_heat_cap_data.csv",
        }

        # Use standard name if known file type, otherwise use original filename
        target_filename = file_mapping.get(file_type, safe_filename)

        # Generate signed URL for direct GCS upload to final location
        if IS_CLOUD:
            storage_client = None
            try:
                # Use IAM-based signing (works with Cloud Run default credentials)
                credentials = compute_engine.Credentials()
                credentials.refresh(google_requests.Request())

                storage_client = storage.Client(credentials=credentials)
                bucket = storage_client.bucket(BATCH_GCS_BUCKET)
                blob = bucket.blob(f"{project_id}/rust_code_input/{target_filename}")

                # Get service account email from metadata server
                service_account_email = credentials.service_account_email

                # URL expires in 1 hour - use IAM signing
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=3600,
                    method="PUT",
                    content_type="application/octet-stream",
                    service_account_email=service_account_email,
                    access_token=credentials.token,
                )

                return jsonify(
                    {
                        "upload_url": signed_url,
                        "blob_path": f"{project_id}/rust_code_input/{target_filename}",
                        "file_type": file_type,
                        "target_filename": target_filename,
                    }
                )
            finally:
                if storage_client:
                    try:
                        storage_client.close()
                    except Exception:
                        pass
        else:
            # Local mode - return indication to use regular upload
            return jsonify({"upload_url": None, "use_regular_upload": True})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate upload URL: {str(e)}"}), 500


@app.route("/finalize-upload", methods=["POST"])
@login_required
def finalize_upload():
    """Finalize upload - handle material property symlinks for pre-loaded materials"""
    try:
        data = request.get_json()
        project_id = data.get("project_id", "").strip()
        material_type = data.get("material_type", "custom").strip().lower()
        uploaded_files = data.get("uploaded_files", {})  # {file_type: filename}

        if not project_id:
            return jsonify({"error": "Project ID is required"}), 400

        project_id = secure_filename(project_id)
        project_path = get_project_path(project_id)
        rust_input_path = project_path / "rust_code_input"
        rust_input_path.mkdir(parents=True, exist_ok=True)

        processed_files = []

        # Handle material property files based on material type
        if material_type in ["abs", "pc", "peek"]:
            # Use pre-loaded material data - copy files (GCSFuse doesn't support symlinks)
            script_dir = Path(__file__).parent.resolve()
            material_variants_dir = (
                script_dir / "material_variants" / material_type.upper()
            )

            material_files = {
                "conductivity.csv": material_variants_dir / "conductivity.csv",
                "sp_heat_cap_data.csv": material_variants_dir / "sp_heat_cap_data.csv",
            }

            for target_name, source_path in material_files.items():
                target_path = rust_input_path / target_name
                if not source_path.exists():
                    print(f"Warning: Material file not found: {source_path}")
                    continue
                # Copy file instead of symlink (GCSFuse compatible)
                shutil.copy2(source_path, target_path)
                processed_files.append(
                    f"{target_name} (pre-loaded {material_type.upper()})"
                )

        # Add uploaded files to the list (files are already renamed to standard names)
        for file_type, filename in uploaded_files.items():
            processed_files.append(filename)

        # Save only material_type to project_config.json
        # Filenames are now always standard (Input_file.txt, wall.gcode)
        update_project_config(project_id, {"material_type": material_type})

        if processed_files:
            return jsonify(
                {
                    "message": f"Successfully processed {len(processed_files)} file(s) for project {project_id}",
                    "files": processed_files,
                    "project_id": project_id,
                    "material_type": material_type,
                }
            )
        else:
            return jsonify({"error": "No files were processed"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to finalize upload: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
@login_required
def upload_files():
    try:
        project_id = request.form.get("project_id", "").strip()
        material_type = request.form.get("material_type", "custom").strip().lower()

        if not project_id:
            return jsonify({"error": "Project ID is required"}), 400

        project_id = secure_filename(project_id)
        project_path = get_project_path(project_id)
        rust_input_path = project_path / "rust_code_input"

        # Only create the rust_code_input folder - other folders will be created by their respective stages
        rust_input_path.mkdir(parents=True, exist_ok=True)

        uploaded_files = []

        # Handle material property files based on material type
        if material_type in ["abs", "pc", "peek"]:
            # Use pre-loaded material data - create symbolic links
            script_dir = Path(__file__).parent.resolve()
            material_variants_dir = (
                script_dir / "material_variants" / material_type.upper()
            )

            # Create symbolic links for conductivity and heat capacity files
            material_files = {
                "conductivity.csv": material_variants_dir / "conductivity.csv",
                "sp_heat_cap_data.csv": material_variants_dir / "sp_heat_cap_data.csv",
            }

            for target_name, source_path in material_files.items():
                target_path = rust_input_path / target_name
                if target_path.exists() or target_path.is_symlink():
                    target_path.unlink()
                target_path.symlink_to(source_path)
                uploaded_files.append(
                    f"{target_name} (pre-loaded {material_type.upper()})"
                )
        else:
            # Custom material - expect uploaded files
            try:
                material_files = save_uploaded_files(
                    request.files,
                    rust_input_path,
                    {
                        "conductivity_file": "conductivity.csv",
                        "heat_capacity_file": "sp_heat_cap_data.csv",
                    },
                )
                uploaded_files.extend(material_files)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

        # Handle remaining required files (config and gcode)
        # Always save with standard names
        try:
            for file_key, standard_name in [("config_file", "Input_file.txt"), ("gcode_file", "wall.gcode")]:
                if file_key in request.files:
                    f = request.files[file_key]
                    if f and f.filename and allowed_file(f.filename):
                        # Save with standard name, not original filename
                        f.save(rust_input_path / standard_name)
                        uploaded_files.append(standard_name)

        except Exception as e:
             return jsonify({"error": f"Error saving files: {str(e)}"}), 400

        # Save only material_type to project_config.json
        # Filenames are now always standard (Input_file.txt, wall.gcode)
        update_project_config(project_id, {"material_type": material_type})

        if uploaded_files:
            return jsonify(
                {
                    "message": f"Successfully uploaded {len(uploaded_files)} file(s) to project {project_id} with material type: {material_type.upper()}",
                    "files": uploaded_files,
                    "project_id": project_id,
                    "material_type": material_type,
                }
            )
        else:
            return jsonify({"error": "Failed to upload files"}), 400

    except Exception as e:
        # Catch any unhandled exceptions and return JSON error instead of HTML
        traceback.print_exc()
        error_message = str(e)
        print(f"[ERROR] Upload failed: {error_message}")
        return jsonify({"error": f"Upload failed: {error_message}"}), 500


@app.route("/download/<project_id>/<stage>")
@validate_project_id(check_exists=True)
def download_stage(project_id, stage):
    """Download files from a specific pipeline stage"""
    if stage not in PIPELINE_FOLDERS:
        return jsonify({"error": "Invalid pipeline stage"}), 400

    stage_path = get_project_path(project_id) / stage

    if not stage_path.exists():
        return jsonify({"error": f"No files found for stage {stage}"}), 404

    zip_path = Path("/tmp") / f"{project_id}_{stage}.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file_path in stage_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(stage_path)
                zipf.write(file_path, arcname)

    return send_file(
        str(zip_path), as_attachment=True, download_name=f"{project_id}_{stage}.zip"
    )


@app.route("/download/<project_id>/animation")
@validate_project_id(check_exists=True)
def download_animation(project_id):
    """Download animation MP4 file"""
    video_path = get_project_path(project_id) / "fea_animation.mp4"

    if not video_path.exists():
        return jsonify({"error": "Animation file not found"}), 404

    return send_file(
        str(video_path),
        as_attachment=True,
        download_name=f"{project_id}_animation.mp4",
        mimetype="video/mp4",
    )


@app.route("/download/<project_id>/mask")
@validate_project_id(check_exists=True)
def download_mask(project_id):
    """Download hot/cold mask visualization image"""
    mask_path = get_project_path(project_id) / "hot_cold_mask.png"

    if not mask_path.exists():
        return jsonify({"error": "Mask file not found"}), 404

    return send_file(
        str(mask_path),
        as_attachment=True,
        download_name=f"{project_id}_hot_cold_mask.png",
        mimetype="image/png",
    )


@app.route("/download/<project_id>/mask3mf")
@validate_project_id(check_exists=True)
def download_mask_3mf(project_id):
    """Download hot/cold mask 3D model (3MF format)"""
    mask_3mf_path = get_project_path(project_id) / "hot_cold_mask.3mf"

    if not mask_3mf_path.exists():
        # Try STL fallback if 3MF doesn't exist
        mask_stl_path = get_project_path(project_id) / "hot_cold_mask.stl"
        if mask_stl_path.exists():
            return send_file(
                str(mask_stl_path),
                as_attachment=True,
                download_name=f"{project_id}_hot_cold_mask.stl",
                mimetype="model/stl",
            )
        return jsonify({"error": "Mask 3D file not found"}), 404

    return send_file(
        str(mask_3mf_path),
        as_attachment=True,
        download_name=f"{project_id}_hot_cold_mask.3mf",
        mimetype="model/3mf",
    )


@app.route("/download/<project_id>/input_config")
@validate_project_id(check_exists=True)
def download_input_config(project_id):
    """Download input configuration files (Input_file.txt and gcode)"""
    rust_input_path = get_project_path(project_id) / "rust_code_input"

    if not rust_input_path.exists():
        return jsonify({"error": "Input files not found"}), 404

    # Find the gcode file (might have different names)
    gcode_file = None
    for file in rust_input_path.iterdir():
        if file.suffix.lower() == ".gcode":
            gcode_file = file
            break

    config_file = rust_input_path / "Input_file.txt"

    if not config_file.exists() and not gcode_file:
        return jsonify({"error": "No configuration files found"}), 404

    # Create ZIP with just these two files
    zip_path = Path("/tmp") / f"{project_id}_input_config.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        if config_file.exists():
            zipf.write(config_file, "Input_file.txt")
        if gcode_file:
            zipf.write(gcode_file, gcode_file.name)

    return send_file(
        str(zip_path),
        as_attachment=True,
        download_name=f"{project_id}_input_config.zip",
    )


@app.route("/projects")
def list_projects():
    projects = []
    for project_id in get_project_folders():
        status = get_project_status(project_id)
        projects.append({"id": project_id, "status": status})

    return jsonify(projects)


@app.route("/project/<project_id>/run/<stage>", methods=["POST"])
@login_required
@validate_project_id(check_exists=True)
def run_pipeline_stage(project_id, stage):
    """Run a specific pipeline stage"""
    valid_stages = ["rust_simulation", "visualization", "mask_generation"]
    if stage not in valid_stages:
        print(f"[ERROR] Invalid stage requested: {stage}")
        return jsonify({"success": False, "error": f"Invalid stage: {stage}"}), 400

    response_data, status_code = execute_pipeline_stage(project_id, stage)
    return jsonify(response_data), status_code


@app.route("/project/<project_id>/run_all", methods=["POST"])
@login_required
@validate_project_id(check_exists=True)
def run_full_pipeline(project_id):
    """Run the entire pipeline from start to finish"""
    response_data, status_code = execute_pipeline_stage(project_id, "full_pipeline")

    # Add execution field for backward compatibility
    if response_data.get("success") and "message" in response_data:
        response_data["execution"] = response_data["message"]

    return jsonify(response_data), status_code


@app.route("/project/<project_id>/progress")
@validate_project_id()
def get_progress(project_id):
    """Get the current progress of frame generation for a project"""
    try:
        progress_data = get_progress_data(project_id)
        return jsonify(progress_data)
    except FileNotFoundError:
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/<project_id>/error")
@validate_project_id()
def get_error(project_id):
    """Get error data for a project"""
    try:
        error_data = get_error_data(project_id)
        if error_data is None:
            return jsonify({"has_error": False})
        return jsonify({"has_error": True, **error_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/<project_id>/job_status")
@validate_project_id()
def get_batch_job_status(project_id):
    """Get the status of the current Cloud Batch job for a project"""
    client = None
    try:
        job_data = get_job_data(project_id)
        if not job_data:
            return jsonify({"status": "NO_JOB", "message": "No job found"})

        # If not using Cloud Batch, return completed
        if not CLOUD_BATCH_ENABLED:
            return jsonify({"status": "SUCCEEDED", "message": "Local processing"})

        # Query Cloud Batch API for job status
        client = batch_v1.BatchServiceClient()
        job = client.get_job(name=job_data["job_location"])

        # Map Cloud Batch status to our status
        status_map = {
            batch_v1.JobStatus.State.STATE_UNSPECIFIED: "UNKNOWN",
            batch_v1.JobStatus.State.QUEUED: "QUEUED",
            batch_v1.JobStatus.State.SCHEDULED: "RUNNING",
            batch_v1.JobStatus.State.RUNNING: "RUNNING",
            batch_v1.JobStatus.State.SUCCEEDED: "SUCCEEDED",
            batch_v1.JobStatus.State.FAILED: "FAILED",
            batch_v1.JobStatus.State.DELETION_IN_PROGRESS: "FAILED",
        }

        job_status = status_map.get(job.status.state, "UNKNOWN")

        # Clear job data if completed or failed
        if job_status in ["SUCCEEDED", "FAILED"]:
            clear_job_data(project_id)

        return jsonify(
            {
                "status": job_status,
                "job_id": job_data["job_id"],
                "stage": job_data["stage"],
                "message": f"Job {job_status.lower()}",
            }
        )

    except Exception as e:
        print(f"Error checking job status: {e}")
        return jsonify({"status": "ERROR", "error": str(e)}), 500
    finally:
        # Explicitly close client to release file descriptors
        if client:
            try:
                client.transport.close()
            except Exception:
                pass


@app.route("/download/<project_id>/error")
@validate_project_id(check_exists=True)
def download_error(project_id):
    """Download error.json file"""
    error_file = get_project_path(project_id) / "error.json"

    if not error_file.exists():
        return jsonify({"error": "No error file found"}), 404

    return send_file(
        str(error_file),
        as_attachment=True,
        download_name=f"{project_id}_error.json",
        mimetype="application/json",
    )


@app.route("/project/<project_id>/delete", methods=["DELETE"])
@login_required
@validate_project_id(check_exists=True)
def delete_project(project_id):
    """Delete an entire project folder"""
    project_path = get_project_path(project_id)

    try:
        shutil.rmtree(project_path)

        return jsonify(
            {"success": True, "message": f"Project {project_id} deleted successfully"}
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/project/<project_id>/rename", methods=["POST"])
@login_required
@validate_project_id(check_exists=True)
def rename_project(project_id):
    """Rename a project folder"""
    project_path = get_project_path(project_id)

    try:
        data = request.get_json()
        new_name = data.get("new_name", "").strip()

        if not new_name:
            return jsonify(
                {"success": False, "error": "New project name is required"}
            ), 400

        # Sanitize the new name
        safe_new_name = secure_filename(new_name)

        if not safe_new_name:
            return jsonify({"success": False, "error": "Invalid project name"}), 400

        # Check if a project with the new name already exists
        new_project_path = get_project_path(safe_new_name)

        if new_project_path.exists():
            return jsonify(
                {"success": False, "error": "A project with this name already exists"}
            ), 409

        # Rename the project folder (retry with aggressive cleanup if we hit the file descriptor ceiling)
        try:
            project_path.rename(new_project_path)
        except OSError as rename_error:
            if rename_error.errno != errno.EMFILE:
                raise

            # Aggressive cleanup: force garbage collection multiple times and wait
            print(
                f"[{project_id}] Hit file descriptor limit during rename, cleaning up..."
            )
            gc.collect()
            gc.collect()
            time.sleep(0.1)  # Give OS time to close file descriptors
            ensure_fd_headroom()

            # Retry rename operation
            try:
                project_path.rename(new_project_path)
            except OSError as second_error:
                if second_error.errno == errno.EMFILE:
                    # Still failing - log more diagnostic info
                    import resource

                    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                    raise OSError(
                        errno.EMFILE,
                        f"Too many open files even after cleanup. FD limits: soft={soft}, hard={hard}. "
                        "Try reducing concurrent requests or increasing system limits.",
                    )
                raise

        return jsonify(
            {
                "success": True,
                "message": f"Project renamed from {project_id} to {safe_new_name}",
                "new_name": safe_new_name,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    # Use PORT environment variable for Cloud Run, fallback to 5000 for local development
    port = int(os.environ.get("PORT", 5000))

    if port == 5000:
        # Local development mode
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        # Production mode
        from waitress import serve

        print(f"Starting production server on port {port}...")
        # Reduced thread count from 4 to 2 to minimize file descriptor usage
        serve(app, host="0.0.0.0", port=port, threads=2)
