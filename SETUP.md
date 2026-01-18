# Setup Guide Deployment

This guide covers the initial setup required to deploy CITB4 to your own Google Cloud Platform project.

## Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and authenticated
- Python 3.13 
- `uv` package manager (or standard `pip`)

## Required Environment Variables

### OAuth Configuration (for user authentication)

Create OAuth 2.0 credentials in Google Cloud Console:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials
2. Create OAuth 2.0 Client ID (Web application type)
3. Add authorized redirect URIs:
   - **DEV**: `https://citb4-dev-<PROJECT_NUMBER>.<REGION>.run.app/authorize`
   - **PROD**: `https://citb4-<PROJECT_NUMBER>.<REGION>.run.app/authorize`
4. Download the client secret JSON

Set these as Cloud Run environment variables:
```bash
OAUTH_CLIENT_ID=<your-client-id>.apps.googleusercontent.com
OAUTH_CLIENT_SECRET=<your-client-secret>
SECRET_KEY=<random-secret-key-for-flask-sessions>
```

To generate a secure SECRET_KEY:
```bash
python -c "import os; print(os.urandom(24).hex())"
```

### GCP Project Configuration

```bash
# Your GCP project ID
export GCP_PROJECT=your-project-id

# Environment (DEV or PROD)
export ENV=DEV  # or PROD

# Optional: override default region (defaults to europe-west4)
export BATCH_REGION=europe-west4
```

## Initial GCP Setup

### 1. Enable Required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  batch.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com
```

### 2. Create Cloud Storage Buckets


```bash
# Development bucket
gsutil mb -l europe-west4 gs://citb4-projects-dev

# Production bucket (when ready)
gsutil mb -l europe-west4 gs://prod-projects
```

Verify bucket locations:
```bash
gsutil ls -L -b gs://citb4-projects-dev | grep location
gsutil ls -L -b gs://prod-projects | grep location
```

### 3. Configure IAM Permissions

Grant Cloud Batch service account access to storage:

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT --format='value(projectNumber)')

# Grant storage admin role to Compute Engine service account (used by Cloud Batch)
gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Grant Cloud Run service account storage access
gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/iam.serviceAccountTokenCreator"
```

### 4. Configure OAuth Credentials in Cloud Run

**Option A: Using gcloud CLI**

```bash
# For DEV environment
gcloud run services update citb4-dev \
  --region=europe-west1 \
  --update-env-vars=OAUTH_CLIENT_ID=<your-client-id>,OAUTH_CLIENT_SECRET=<your-client-secret>,SECRET_KEY=<your-secret-key>

# For PROD environment
gcloud run services update citb4 \
  --region=europe-west1 \
  --update-env-vars=OAUTH_CLIENT_ID=<your-client-id>,OAUTH_CLIENT_SECRET=<your-client-secret>,SECRET_KEY=<your-secret-key>
```

**Option B: Using Google Cloud Console**

1. Navigate to Cloud Run → Select service (citb4-dev or citb4)
2. Click "Edit & Deploy New Revision"
3. Go to "Variables & Secrets" tab
4. Add environment variables:
   - `OAUTH_CLIENT_ID`
   - `OAUTH_CLIENT_SECRET`
   - `SECRET_KEY`

### 5. Request Quota Increases (if needed)

CITB4 uses high-CPU machines for processing. Check and request quota increases:

```bash
# Check current quotas
gcloud compute project-info describe --project=$GCP_PROJECT

# Request increases via: https://console.cloud.google.com/iam-admin/quotas
```

**Required quotas for europe-west4:**
- `CPUS_ALL_REGIONS`: 128+ (for n2-highmem-128 instances)
- `N2_CPUS`: 128+ (region-specific)
- `GPUS_ALL_REGIONS`: 2 (optional, for future GPU support)

## Deployment

### Automated Deployment (GitHub)

The repository is configured for automatic deployment via Cloud Build:

1. **Fork or clone the repository**
2. **Configure GitHub → Google Cloud Build integration**:
   - Go to Cloud Build → Triggers
   - Connect your GitHub repository
   - Create trigger for `main` branch (DEV) and `release` branch (PROD)

3. **Push to trigger deployment**:
   ```bash
   git push origin main  # Deploys to DEV (citb4-dev)
   git push origin release  # Deploys to PROD (citb4)
   ```

### Manual Deployment

```bash
# Build and deploy manually
gcloud builds submit --config cloudbuild.yaml .
```

## Local Development Setup

### 1. Install Python Dependencies

Using `uv` (recommended):
```bash
# Install uv if not already installed
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Using standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize Rust Submodule

```bash
git submodule update --init --recursive
git submodule update --remote DES_thermal_simulation

# Linux case-sensitivity fix for Rust imports
cd DES_thermal_simulation/src
ln -sf Interpolator.rs interpolator.rs
cd ../..
```

### 3. Configure Local Environment

Using `direnv` (optional, recommended):
```bash
# .envrc is already configured
direnv allow
```

Or manually activate:
```bash
source .venv/bin/activate
```

### 4. Run Locally

```bash
# Run Flask UI
python app.py

# Or using podman/docker
podman build -t citb4 .
podman run -p 5000:5000 -e MODE=ui citb4
```

Access at: http://localhost:5000

## Post-Deployment Verification

### 1. Check Cloud Run Service

```bash
# Get service URL
gcloud run services describe citb4-dev --region=europe-west1 --format='value(status.url)'

# Test health endpoint
curl https://citb4-dev-<PROJECT_NUMBER>.europe-west1.run.app/health
```

### 2. Verify OAuth Configuration

Visit the Cloud Run URL and click "Login with Google". You should be redirected to Google sign-in.

### 3. Test File Upload

1. Upload a test project with required input files:
   - `rust_code_input/conductivity.csv`
   - `rust_code_input/sp_heat_cap_data.csv`
   - `rust_code_input/Input_file.txt`
   - `rust_code_input/wall.gcode`

2. Click "Run Simulation" and verify Cloud Batch job is created

3. Check job status:
   ```bash
   gcloud batch jobs list --location=europe-west4
   ```

### 4. Monitor Logs

```bash
# Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=citb4-dev" --limit 50 --format=json

# Cloud Batch logs
gcloud logging read "resource.type=cloud_batch_job" --limit 50 --format=json
```

## Troubleshooting

### OAuth Errors

**Error**: "redirect_uri_mismatch"
- **Fix**: Add the exact Cloud Run URL to OAuth authorized redirect URIs in Google Cloud Console

### Cloud Batch Failures

**Error**: "CODE_VOLUME_INVALID_ARGUMENT"
- **Fix**: Ensure GCS bucket path has NO trailing slash in `cloudbuild.yaml`

**Error**: "PERMISSION_DENIED"
- **Fix**: Grant service account permissions (see IAM setup above)

### Large File Uploads

Files >32MB require direct GCS upload (handled automatically by the app). Ensure:
- Service account has `storage.objectAdmin` role
- OAuth is properly configured for signed URL generation

## Data storage 

All data remains in Europe:
- Cloud Storage: `europe-west4` (Netherlands)
- Cloud Batch: `europe-west4`
- Cloud Run: `europe-west1` (Belgium) (This is not consistent because some machines weren't available in some regions)

## Support

For issues specific to CITB4 deployment:
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed architecture documentation
- Review [CLAUDE.md](CLAUDE.md) for development notes
- Check Cloud Run and Cloud Batch logs for error details

For GCP-specific issues:
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Batch Documentation](https://cloud.google.com/batch/docs)
- [Google Cloud Support](https://cloud.google.com/support)
