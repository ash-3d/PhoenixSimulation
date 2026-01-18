* Don't read CSV files as a whole as an agent (code can, your context window is limited), just read like 10 lines
* After each edit identify things to simplify and reduce code
* Data shouldn't leave europe

## Project File Structure

```
CITB4_python/
├── fea_pyvista.py         # Main FEA processing script with PyVista visualization
├── app.py                  # Flask web application for file upload/download interface
├── job_processor.py        # Cloud Batch processor script (runs pipeline stages)
├── make_animation.py       # Creates MP4 animations from generated frame sequences
├── start.sh               # Container startup script
├── cloudbuild.yaml        # CI/CD configuration for GitHub auto-deploy
├── requirements.txt       # Python dependencies (Python 3.11+, tested with 3.13)
├── environment.py         # Centralized runtime/env + Cloud Batch configuration
├── env_targets.py         # DEV/PROD resource targets (buckets, services, prefixes)
├── .python-version        # Python version specification for uv
├── .envrc                 # direnv configuration for auto-activation
├── README.md              # Project documentation
├── USAGE.md               # Usage instructions
├── DEPLOYMENT.md          # Complete deployment guide
├── CLAUDE.md              # Development notes and commands (this file)
├── templates/
│   └── index.html         # Flask web interface template
├── projects/              # Working directory for processing projects
│   └── 001/               # Example project directory
│       ├── rust_code_input/           # Input files for Rust thermal simulation
│       │   ├── conductivity.csv       # Material thermal conductivity data
│       │   ├── sp_heat_cap_data.csv   # Material specific heat capacity data
│       │   ├── Input_file.txt         # Simulation configuration
│       │   └── wall.gcode             # G-code toolpath file
│       ├── frame_generator_input/     # FEA data for visualization (output of Rust simulation)
│       │   ├── activation_times.csv   # Element activation timing data
│       │   ├── elementfile.csv        # Element connectivity definitions
│       │   ├── node_temps.csv         # Node temperature data (huge)
│       │   └── nodefile.csv           # Node coordinate definitions
│       ├── frames_filtered_active_only/  # Generated visualization frames (PNG sequence)
│       ├── fea_animation.mp4          # Final animation video (at project root)
│       ├── hot_cold_mask.png          # Hot/cold mask visualization (at project root)
│       └── hot_cold_mask.3mf          # Hot/cold mask 3D model (at project root)
├── fetch_batch_failure_logs.py  # Debug failed batch jobs (searches backwards from end, use --tail or --show-both-ends)
├── material_variants/     # ABS/PC/PEEK conductivity & heat capacity CSV bundles
```

### Data Flow (Pipeline Stages)

1. **Rust Simulation** (`rust_simulation` stage):
   - Input: `rust_code_input/` (gcode, material CSVs, config)
   - Output: `frame_generator_input/` (nodefile, elementfile, node_temps, activation_times)

2. **Visualization + Animation** (`visualization` stage):
   - Input: `frame_generator_input/`
   - Processing: `fea_pyvista.py` generates frame sequence
   - Output: `frames_filtered_active_only/` (PNG frames) + `fea_animation.mp4` (at project root)

3. **Mask Generation** (`mask_generation` stage - optional, separate):
   - Input: `frame_generator_input/`
   - Output: `hot_cold_mask.png` + `hot_cold_mask.3mf` (at project root)
   - Can be re-run with different parameters (Tg, dHigh, dLow, time_s) without re-rendering 

node_temps.csv is important. it contains node temperatures which we average into element temperatures. When temperatures don't change during a time step , it's not specified in the file. Also not all nodes are present, theres' a mismatch 
Encountered bug:

it produces temperature change for remaining timesteps but stopped printing new voxels                                                                                                                         
citb4-projects-dev/200xboat_new/frame_generator_input
eg:   
  elementfile.csv:       2,034,830 lines
  activation_times.csv:  2,034,830 lines 
  elem_temps.csv:        2,034,830 lines 
  nodefile.csv:          3,382,607 lines
  node_temps.csv:        3,382,607 lines 


### Progress Tracking
- Progress tracked via `progress.json` file in each project directory (format: `{"current": 0, "total": 78, "percentage": 0, "message": "...", "job_status": "RUNNING", "stage": "..."}`)
- Job status flow: `SUBMITTED` (when batch job created) → `RUNNING` (job_processor starts) → `SUCCEEDED`/`FAILED` (job_processor exits). UI polls progress.json every 2s and only shows green on SUCCEEDED.

## Deployment Architecture

**CITB4 uses a hybrid cloud architecture:**
- **Cloud Run Service** (`citb4`): Lightweight Flask UI for uploads/downloads/triggering
- **Cloud Batch** (dynamic jobs): Heavy computational processing with stage-specific machines:
  - **Rust Simulation**: `n2-highmem-128` (128 vCPUs, 512GB RAM) - CPU-intensive thermal simulation
  - **Visualization + Animation**: `n2-highmem-128` (128 vCPUs, 512GB RAM) - CPU-only PyVista rendering (GPU disabled for now)
  - **Mask Generation**: `n2-highmem-32` (32 vCPUs, 128GB RAM) - Single-frame mask rendering (optional stage)
- **Cloud Storage** (`citb4-projects` bucket): Persistent data storage

See **DEPLOYMENT.md** for complete deployment guide.

### Build and Run Locally
```bash
podman build -t citb4 .
podman run -p 5000:5000 -e MODE=ui citb4
podman run -e MODE=processor -e PROJECT_ID=TEST -e STAGE=rust_simulation citb4
```

### Deploy to Google Cloud (Automated)
Deployment is automated via GitHub. push to main branch triggers auto-deploy via `cloudbuild.yaml`.

```bash
# One-time setup: Create Cloud Storage bucket and enable APIs
gsutil mb -l europe-west4 gs://citb4-projects
gcloud services enable batch.googleapis.com

# Verify bucket exists and permissions
gsutil ls gs://citb4-projects
gsutil iam get gs://citb4-projects

# If needed, grant Compute Engine service account access
# (Replace PROJECT_NUMBER with your project number from gcloud projects list)
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Push to GitHub to trigger auto-deploy
git push origin main

# Manual deployment (if needed)
gcloud builds submit --config cloudbuild.yaml .
```

This deploys:
1. Cloud Run Service: `citb4` (Flask UI)
2. Enables Cloud Batch API (processing jobs created dynamically with 128 vCPUs)
3. Mounts Cloud Storage bucket for persistent data

**Important**: GCS bucket path must NOT have trailing slash (`citb4-projects` not `citb4-projects/`) or Cloud Batch will fail with `CODE_VOLUME_INVALID_ARGUMENT`.

### Development/Manual Processing
```bash
# Run interactive shell
podman run -it -p 5000:5000 citb4 /bin/bash

# Run manual processing inside container
podman exec -it <container_name> python job_processor.py

# Run Flask UI inside container
podman exec -it <container_name> python app.py
```

## Web Interface
- **Local**: http://localhost:5000
- **Cloud Run**: Get URL via `gcloud run services describe citb4 --region europe-west1 --format='value(status.url)'`
- Upload input files with project ID
  - **Large file uploads (>32MB)**: Uses direct GCS upload with signed URLs to bypass Cloud Run's 32MB HTTP/1 limit
  - **Flow**: Frontend → `/get-upload-url` → Direct GCS upload to `{project_id}/rust_code_input/` → `/finalize-upload` (handles material symlinks)
  - No intermediate temp storage - files uploaded directly to final location
- Click play buttons to trigger Cloud Batch jobs (not run in UI anymore!)
- Download processed output files as ZIP
- Health check endpoint: `/health`

### Path Configuration
- **app.py** automatically detects environment and uses appropriate project directory:
  - Cloud Run: `/app/projects` (mounted from Cloud Storage bucket `citb4-projects`)
  - Local: `./projects`
- This prevents permission errors when running locally while maintaining cloud compatibility
- **Environment-specific project paths**: PROD uses `prod-projects` (GCS: `gs://prod-projects/`), DEV uses `citb4-projects-dev` (GCS: `gs://citb4-projects-dev/`)
- Environment determined by git branch: `release` → PROD, all others → DEV (or set via `ENV` environment variable)

### Cloud Configuration
- **Cloud Batch Jobs** (Processor):
  - Logs viewable via `gcloud batch jobs describe <job-id> --location europe-west4`
  - View all jobs: `gcloud batch jobs list --location europe-west4`
- **Cloud Run Logs**:
  - View recent logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=citb4-dev" --limit 100 --format=json --freshness=1h`
  - View errors only: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=citb4-dev AND severity>=ERROR" --limit 50 --freshness=1h`
  - Real-time logs: `gcloud alpha logging tail "resource.labels.service_name=citb4-dev"`

## Environment Setup

### Using uv 
```bash
# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

**DES_thermal_simulation submodule :**
```bash
git submodule update --remote DES_thermal_simulation
```

## notes :
* **Quotas** (europe-west4 + global):
  - `CPUS_ALL_REGIONS`: 128 ✅ approved (current jobs only use CPUs)
  - `GPUS_ALL_REGIONS`: 2 ✅ approved (kept for future GPU runs, currently unused)
* **Rendering mode**: Visualization stage now runs software rendering on n2-highmem-128 (GPU disabled); flip `USE_GPU=1` + Batch config if GPU acceleration returns
* **Batch provisioning**: PROD jobs run on standard (non-preemptible) instances, DEV keeps SPOT nodes for cost savings
* gcode file is related to path length etc so it affects time taken for simulation
* wall.gcode size scales linearly with time taken . estimate time

* Docs about fes simulation input / DES rust simulation output (eg: elem_temps.csv) (with illustrative images) in: ./DES_thermal_simulation/docs/
  included Original research paper as well
* case sensitivity issue on Linux
  The DES submodule has `Interpolator.rs` (capital I) but Rust code imports it as `mod interpolator;` (lowercase). we use Linux so create a symlink:
`ln -sf Interpolator.rs interpolator.rs`
* using direnv for local development

### History

* `K_SERVICE=1` This environment variable is automatically set by Google Cloud Run and triggers cloud-specific configuration:
history:
* Had to downgrade on local system steamos to 3.12 python from 3.13 for rendering to work due to osmessa issue
* Conda mayavi installation  took a lot of time and we switched to pyvista
* tried cloud run jobs and  cloud run service before settling for cloud batch run

### TODO
  * Calculate length of toolpath first and choose spot if estimate is less than 1 hr

### OAuth Dev Setup Notes:
  * OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET for dev are set as Cloud Run environment variables.
  * The authorized redirect URI for dev is `https://citb4-dev-<YOUR_PROJECT_NUMBER>.europe-west1.run.app/authorize`.
  * Replace `<YOUR_PROJECT_NUMBER>` with your GCP project number (found via `gcloud projects describe PROJECT_ID --format='value(projectNumber)'`).
