# Docker Configuration Documentation

## Overview

This document describes the Docker configuration for the LaLiga Match Prediction System. The system uses Docker and Docker Compose to containerize and orchestrate multiple services.

## Services

### 1. Backend (FastAPI)
```yaml
backend:
  build: 
    context: ./src/backend
    dockerfile: dockerfile
  ports:
    - "8000:8000"
  environment:
    - MLFLOW_TRACKING_URI=http://mlflow:5000
  depends_on:
    - mlflow
```

### 2. Frontend (Streamlit)
```yaml
frontend:
  build:
    context: ./src/frontend
    dockerfile: dockerfile
  ports:
    - "8501:8501"
  environment:
    - BACKEND_API_URL=http://backend:8000
  depends_on:
    - backend
```

### 3. MLflow
```yaml
mlflow:
  image: mlflow
  ports:
    - "5000:5000"
  environment:
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  depends_on:
    - minio
```

### 4. MinIO (Object Storage)
```yaml
minio:
  image: minio/minio
  ports:
    - "9000:9000"
    - "9001:9001"
  environment:
    - MINIO_ROOT_USER=minioadmin
    - MINIO_ROOT_PASSWORD=minioadmin
```

### 5. Prometheus & Grafana
```yaml
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
  depends_on:
    - prometheus
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory:
```bash
# API Configuration
API_PORT=8000
API_HOST=0.0.0.0

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Prometheus Configuration
PROMETHEUS_PORT=9090

# Grafana Configuration
GRAFANA_PORT=3000
```

## Deployment

### Local Development
```bash
# Build and start all services
docker-compose up -d

# Build specific service
docker-compose build <service_name>

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Production Deployment
```bash
# Build with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale backend=3
```

## Volume Management

### Data Persistence
```yaml
volumes:
  mlflow_data:
    driver: local
  minio_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
```

## Networking

### Service Communication
- Internal network for service-to-service communication
- Exposed ports for external access
- Reverse proxy configuration for production

## Security Considerations

1. Container Security
   - Use official base images
   - Regular security updates
   - Minimal container privileges

2. Network Security
   - Internal network isolation
   - Port exposure control
   - TLS/SSL configuration

3. Secret Management
   - Environment variables
   - Docker secrets
   - Vault integration

## Monitoring

### Health Checks
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Resource Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '0.50'
      memory: 512M
    reservations:
      cpus: '0.25'
      memory: 256M
```

## Troubleshooting

### Common Issues
1. Container Startup Failures
   - Check logs: `docker-compose logs <service>`
   - Verify environment variables
   - Check resource availability

2. Network Problems
   - Verify service discovery
   - Check port mappings
   - Inspect network configuration

3. Volume Issues
   - Check permissions
   - Verify mount points
   - Inspect volume data

## Best Practices

1. Development
   - Use development-specific configurations
   - Enable debug logging
   - Mount source code for live updates

2. Production
   - Use production-optimized builds
   - Implement proper logging
   - Configure resource limits
   - Enable health checks

3. Maintenance
   - Regular image updates
   - Log rotation
   - Backup strategies
   - Monitoring setup 