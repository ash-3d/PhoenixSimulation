This work is based on [research work](https://doi.org/10.3390/ma13214985) done at University of Maine by Sunil Bhandari and Roberto Lopez-Anido in `https://github.com/linus131/DES_thermal_model`.

This project is a wrapper around it using GCP with a UI and visualization of the printing process, optimized for speed and graphics rendering.
Most of the project was made using AI-assisted coding.

Further development ideas:
* Thermal structural coupling analysis can be done using FEA simulation tools (e.g., Calculix) to understand the warping behaviour.
* Save run costs by using spot/on-demand based on toolpath length (longer runs = risk of interruptions in SPOT).

## Project File Structure

```
PhoenixSimulation/
├── app.py                  # Flask web application for file upload/download interface
├── fea_pyvista.py          # Main FEA processing script with PyVista visualization
├── job_processor.py        # Cloud Batch processor script (runs pipeline stages)
├── make_animation.py       # Creates MP4 animations from generated frame sequences
├── environment.py          # Centralized runtime/env + Cloud Batch configuration
├── env_targets.py          # DEV/PROD resource targets (buckets, services, prefixes)
├── start.sh                # Container startup script
├── Dockerfile              # Container build configuration
├── cloudbuild.yaml         # CI/CD configuration for GitHub auto-deploy
├── requirements.txt        # Python dependencies (Python 3.11+)
├── README.md               # Project documentation (CLAUDE.md & AGENTS.md symlink here)
├── DEPLOYMENT.md           # Complete deployment guide
├── SETUP.md                # Initial setup instructions
├── .gitignore              # Git ignore patterns
├── templates/
│   └── index.html          # Flask web interface template
├── scripts/                # Utility scripts
│   ├── batch_jobs.py              # Cloud Batch job management
│   ├── batch_jobs_interactive.py  # Interactive batch job tools
│   ├── builds.py                  # Build utilities
│   ├── fetch_batch_failure_logs.py # Debug failed batch jobs
│   └── test_mask_generation.py    # Mask generation tests
├── DES_thermal_simulation/ # Rust thermal simulation (MIT license)
│   ├── src/                       # Thanks to https://github.com/linus131/DES_thermal_model
│   ├── Cargo.toml                 # Rust dependencies
│   └── inputfiles/                # Example input files
├── DES_docs/               # Research documentation
│   ├── des_outputs.txt            # Output format documentation
│   ├── paper_materials-13-04985.pdf  # Original research paper
│   └── *.jpeg                     # Diagrams (flowcharts, element visualization)
├── material_variants/      # Material property bundles (conductivity & heat capacity CSVs)
│   ├── ABS/                       
│   ├── PC/                       
│   └── PEEK/                    
├── projects/               # Working directory for processing projects
│   └── <project_id>/              # Each project directory contains:
│       ├── rust_code_input/           # Input files for Rust thermal simulation
│       │   ├── conductivity.csv       # Material thermal conductivity data
│       │   ├── sp_heat_cap_data.csv   # Material specific heat capacity data
│       │   ├── Input_file.txt         # Simulation configuration
│       │   └── wall.gcode             # G-code toolpath file
│       ├── frame_generator_input/     # FEA data (output of Rust simulation)
│       │   ├── activation_times.csv   # Element activation timing data
│       │   ├── elementfile.csv        # Element connectivity definitions
│       │   ├── node_temps.csv         # Node temperature data (large)
│       │   └── nodefile.csv           # Node coordinate definitions
│       ├── frames_filtered_active_only/  # Generated visualization frames (PNG)
│       ├── fea_animation.mp4          # Final animation video
│       ├── hot_cold_mask.png          # Hot/cold mask visualization
│       ├── hot_cold_mask.3mf          # Hot/cold mask 3D model
│       └── progress.json              # Job progress tracking
├── docs/                   # Additional documentation
└── scratch/                # Temporary working files
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

#### Notes on node_temps.csv
`node_temps.csv` is important. It contains node temperatures which we average into element temperatures. When temperatures don't change during a time step, it's not specified in the file. Also, not all nodes are present; there's a mismatch.

**Known bug**: It produces temperature changes for remaining timesteps but stops printing new voxels.

Example file sizes from `citb4-projects-dev/200xboat_new/frame_generator_input`:
```
elementfile.csv:       2,034,830 lines
activation_times.csv:  2,034,830 lines
elem_temps.csv:        2,034,830 lines
nodefile.csv:          3,382,607 lines
node_temps.csv:        3,382,607 lines
```

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
Deployment is automated via GitHub. Push to main branch triggers auto-deploy via `cloudbuild.yaml`.

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

## notes :
* **Tested on available CPU/GPUs** (europe-west4 + global):
  - `CPUS_ALL_REGIONS`: 128. 
  - `GPUS_ALL_REGIONS`: 2. 
* **Batch provisioning**: PROD jobs run on standard (non-preemptible) instances, DEV keeps SPOT nodes for cost savings
* gcode file is related to path length etc so it affects time taken for simulation
* wall.gcode size scales linearly with time taken . estimate time

* Docs about FEA simulation input / DES rust simulation output (eg: `elem_temps.csv`) in: `./DES_docs/`
  Includes original research paper and diagrams
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


### OAuth Dev Setup Notes:
  * OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET for dev are set as Cloud Run environment variables.
  * The authorized redirect URI for dev is `https://citb4-dev-<YOUR_PROJECT_NUMBER>.europe-west1.run.app/authorize`.
  * Replace `<YOUR_PROJECT_NUMBER>` with your GCP project number (found via `gcloud projects describe PROJECT_ID --format='value(projectNumber)'`).
