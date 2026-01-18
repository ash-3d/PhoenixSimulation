# Cloud Run Jobs Deployment Guide

This guide walks you through deploying CITB4 with the **hybrid Cloud Run architecture**:
- **Cloud Run Service** (`citb4-ui`): Lightweight Flask UI for uploads/downloads
- **Cloud Run Job** (`citb4-processor`): Heavy computational processing (up to 24 hours)

## Architecture Overview

```
User → Cloud Run Service (UI)
         ↓ uploads files to Cloud Storage bucket
         ↓ triggers...
      Cloud Run Job (runs simulation)
         ↓ saves results to Cloud Storage
         ↓ completes
User ← Cloud Run Service (downloads results)
```


---

## Prerequisites

1. Google Cloud Project with billing enabled
2. `gcloud` CLI installed and authenticated
3. GitHub repository connected to Cloud Build (or manual deployment)

---

## Step 1: Create Cloud Storage Bucket

This bucket stores all project data persistently across deployments:

```bash
# Create bucket in same region as Cloud Run services
gsutil mb -l europe-west1 gs://citb4-projects

# Verify bucket was created
gsutil ls gs://citb4-projects
```

---

## Step 2: Enable Required APIs

```bash
# Enable Cloud Run, Cloud Build, and Container Registry APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Step 5: Verify Deployment

### Check Service Status

```bash
# Check Cloud Run Service
gcloud run services describe citb4-ui --region europe-west1

# Check Cloud Run Job
gcloud run jobs describe citb4-processor --region europe-west1

# Get Service URL
gcloud run services describe citb4-ui \
  --region europe-west1 \
  --format='value(status.url)'
```

### Test the UI

```bash
# Get the URL
SERVICE_URL=$(gcloud run services describe citb4-ui \
  --region europe-west1 \
  --format='value(status.url)')

echo "Access your app at: $SERVICE_URL"

# Open in browser
xdg-open $SERVICE_URL  # Linux
```

---

##  GitHub Auto-Deploy 

Monitor the build:

```bash
# Watch Cloud Build logs
gcloud builds list --limit 1

# Get detailed logs
gcloud builds log $(gcloud builds list --limit 1 --format='value(id)')
```

---

## Monitoring and Troubleshooting

### View Service Logs (UI)

```bash
# Real-time logs
gcloud run services logs tail citb4-ui --region europe-west1

# Recent logs
gcloud run services logs read citb4-ui --region europe-west1 --limit 100
```

### View Job Logs (Processing)

```bash
# List recent job executions
gcloud run jobs executions list --job citb4-processor --region europe-west1

# Get logs for specific execution
gcloud run jobs executions logs read EXECUTION_NAME --region europe-west1

# Or tail latest execution
gcloud run jobs executions logs tail $(gcloud run jobs executions list \
  --job citb4-processor --region europe-west1 --limit 1 --format='value(name)') \
  --region europe-west1
```

### Check Cloud Storage

```bash
# List projects in bucket
gsutil ls gs://citb4-projects/

# Check specific project
gsutil ls -r gs://citb4-projects/PROJECT_ID/
```


## Cost Optimization

### Current Configuration

- **UI Service**: Scales to 0 when idle (minimal cost)
- **Processor Job**: Only runs when triggered (pay per execution)
- **Storage**: ~$0.02/GB/month for Cloud Storage

### To Further Reduce Costs

```bash
# Keep UI service scaled to zero when not in use
gcloud run services update citb4-ui \
  --region europe-west1 \
  --min-instances 0

# Reduce job resources if possible
gcloud run jobs update citb4-processor \
  --region europe-west1 \
  --memory 16Gi \
  --cpu 4
```


## Cleanup

To delete everything:

```bash
# Delete Cloud Run Service
gcloud run services delete citb4-ui --region europe-west1

# Delete Cloud Run Job
gcloud run jobs delete citb4-processor --region europe-west1

# Delete container images
gcloud container images delete gcr.io/$PROJECT_ID/citb4:latest

# Delete Cloud Storage bucket (WARNING: deletes all project data!)
gsutil rm -r gs://citb4-projects/

# Delete Cloud Build trigger
gcloud builds triggers delete citb4-auto-deploy
```
----
deploying changes: 
#  authenticate:
gcloud auth login
gcloud auth configure-docker  # For GCR

# Or for Artifact Registry:
gcloud auth configure-docker REGION-docker.pkg.dev


# For GCR
podman push gcr.io/YOUR_PROJECT_ID/citb4:latest




---

## Updating the Application

With auto-deploy configured, updates are automatic:

1. Make code changes locally
2. Commit and push to GitHub
3. Cloud Build automatically deploys new version
4. Both Service and Job are updated

For manual updates:

```bash
# Submit build manually
gcloud builds submit --config cloudbuild.yaml .
```
gcloud services enable artifactregistry.googleapis.com --project=YOUR_PROJECT_ID
gcloud artifacts repositories create citb-repo \
    --repository-format=docker \
    --location=europe-west3 \
    --description="CITB4 Docker Repo" \
    --project=YOUR_PROJECT_ID
podman tag gcr.io/YOUR_PROJECT_ID/citb4:latest \
    europe-west3-docker.pkg.dev/YOUR_PROJECT_ID/citb-repo/citb4:latest
-----
for pushing new containers

* (Cloud Run Jobs were limited to 8 vCPUs max), we switched to batch
*- Note: Switched to g2-standard-48 with L4 GPUs (better availability than H100/A100)
    179 -* **GPU acceleration**: Visualization stage uses 4x NVIDIA L4 on g2-standard-48 (standard/on-demand)
