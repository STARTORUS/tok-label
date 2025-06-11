# File Server Deployment Guide

## Overview
This file server provides REST APIs for file management and Postgres data export capabilities using FastAPI. Key features:
- Static file hosting via `/files` endpoint
- File listing API
- PostgreSQL data export to CSV format
- Docker container deployment

## Prerequisites
- Docker installed
- Access to GitLab container registry
- Python 3.8+ (for local development)

## Installation
1. Build Docker image:
```bash
docker build -t gitlab.startorus.org:5050/scientific-computing/file-server:latest -f Dockerfile .
```

2. Run container:
```bash
docker run -d \
  -p 8000:8000 \
  -v ./data:/data \
  -e DATA_DIR=/data \
  gitlab.startorus.org:5050/scientific-computing/file-server:latest
```

## Updating the Service
1. After making code changes, rebuild the image:
```bash
docker build -t gitlab.startorus.org:5050/scientific-computing/file-server:latest .
```

2. Push to registry:
```bash
docker push gitlab.startorus.org:5050/scientific-computing/file-server:latest
```

3. Redeploy your containers

## Kubernetes Deployment
1. Apply persistent volume (from deploy/pv.yaml):
```bash
kubectl apply -f deploy/pv.yaml
```

2. Deploy application (from deploy/file-server.yaml):
```bash
kubectl apply -f deploy/file-server.yaml
```

## API Documentation
### List Files
```bash
GET /list?dir={directory_path}
```

Example request:
```bash
curl "http://localhost:8000/list?dir=test"
```

### Export Data
```bash
POST /export/
{
  "project_name": "string",
  "shots": [int],
  "name_table_columns": {
    "key": ["table_name", ["col1", "col2"]]
  },
  "t_min": 0.0,
  "t_max": 10.0,
  "resolution": 0.1
}
```

Example request:
```bash
curl -X POST "http://localhost:8000/export/" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "demo_project",
    "shots": [240830026, 240830027],
    "name_table_columns": {
      "ammeter": ["ammeter", ["CS1", "PFP1"]],
      "flux_loop": ["flux_loop", [1, 2, 8]]
    },
    "t_min": 0.0,
    "t_max": 1.0,
    "resolution": 0.001
  }'
```

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| DATA_DIR | /data   | Persistent storage mount point |
| PYTHONUNBUFFERED | 1 | Disable Python output buffering |

## Testing
Run integration tests:
```bash
python test.py
```

## CI/CD Pipeline
The `.gitlab-ci.yml` file contains pre-configured pipeline stages for:
- Automated testing
- Container building
- Deployment to Kubernetes clusters
