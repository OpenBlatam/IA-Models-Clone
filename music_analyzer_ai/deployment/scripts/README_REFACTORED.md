# Refactored Deployment Scripts - Documentation

## 📋 Overview

All deployment scripts have been refactored to follow DevOps best practices with modular design, reusable libraries, and improved error handling.

## 🏗️ Architecture

```
scripts/
├── lib/                    # Reusable libraries
│   ├── common.sh          # Common utilities
│   ├── docker.sh          # Docker functions
│   ├── kubernetes.sh      # Kubernetes functions
│   └── cloud.sh           # Cloud provider functions
├── deploy.sh              # Main deployment (refactored)
├── monitor.sh             # Monitoring (refactored)
├── performance-test.sh    # Performance testing
├── chaos-engineering.sh  # Chaos testing
├── cost-optimization.sh   # Cost analysis
├── compliance-audit.sh   # Security audit
└── disaster-recovery.sh   # DR procedures
```

## 📚 Library Functions

### Common Library (`lib/common.sh`)

**Logging:**
- `log_info()` - Info messages
- `log_success()` - Success messages
- `log_error()` - Error messages
- `log_warn()` - Warning messages
- `log_debug()` - Debug messages (DEBUG=true)

**Validation:**
- `validate_command()` - Check if command exists
- `validate_required_vars()` - Validate environment variables
- `validate_file_exists()` - Check file existence
- `validate_directory_exists()` - Check directory existence

**Utilities:**
- `retry_with_backoff()` - Retry with exponential backoff
- `wait_for_condition()` - Wait for condition to be met
- `health_check()` - HTTP health check
- `get_script_dir()` - Get script directory
- `get_project_root()` - Get project root
- `create_temp_file()` - Create temporary file
- `create_temp_dir()` - Create temporary directory
- `parse_args()` - Parse command line arguments
- `is_ci()` - Check if running in CI
- `get_environment()` - Get current environment
- `generate_id()` - Generate unique ID
- `format_bytes()` - Format bytes to human-readable
- `check_port()` - Check port availability

### Docker Library (`lib/docker.sh`)

**Image Management:**
- `docker_is_running()` - Check if Docker daemon is running
- `docker_check_image_exists()` - Check if image exists
- `docker_pull_image()` - Pull Docker image
- `docker_build_image()` - Build Docker image
- `docker_save_image()` - Save image to file
- `docker_load_image()` - Load image from file

**Container Management:**
- `docker_stop_container()` - Stop container
- `docker_start_container()` - Start container
- `docker_get_container_status()` - Get container status
- `docker_get_container_logs()` - Get container logs
- `docker_cleanup_images()` - Clean up old images

**Docker Compose:**
- `docker_compose_up()` - Start services
- `docker_compose_down()` - Stop services

### Kubernetes Library (`lib/kubernetes.sh`)

**Deployment:**
- `k8s_check_kubectl()` - Validate kubectl
- `k8s_check_namespace()` - Check/create namespace
- `k8s_wait_for_deployment()` - Wait for deployment
- `k8s_wait_for_pod()` - Wait for pod
- `k8s_scale_deployment()` - Scale deployment
- `k8s_rollout_restart()` - Restart deployment
- `k8s_rollout_status()` - Check rollout status
- `k8s_rollback_deployment()` - Rollback deployment

**Resource Management:**
- `k8s_get_pod_status()` - Get pod status
- `k8s_get_pod_logs()` - Get pod logs
- `k8s_apply_manifest()` - Apply manifest
- `k8s_delete_resource()` - Delete resource
- `k8s_get_resource_yaml()` - Get resource YAML
- `k8s_port_forward()` - Port forward

### Cloud Library (`lib/cloud.sh`)

**AWS:**
- `aws_check_credentials()` - Validate AWS credentials
- `aws_get_instance_id()` - Get instance ID
- `aws_get_instance_ip()` - Get instance IP
- `aws_upload_to_s3()` - Upload to S3
- `aws_download_from_s3()` - Download from S3

**Azure:**
- `azure_check_credentials()` - Validate Azure credentials
- `azure_get_resource_group()` - Get resource group
- `azure_upload_to_storage()` - Upload to storage

**GCP:**
- `gcp_check_credentials()` - Validate GCP credentials
- `gcp_get_instance_ip()` - Get instance IP

## 🔧 Usage Examples

### Using Common Library

```bash
#!/bin/bash
source "$(dirname "$0")/lib/common.sh"

init_common

validate_command docker docker.io
validate_required_vars ENVIRONMENT API_URL

health_check "http://localhost:8000/health" 10 5

retry_with_backoff 3 2 "curl -f http://localhost:8000/health"
```

### Using Docker Library

```bash
#!/bin/bash
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker.sh"

if ! docker_is_running; then
    log_error "Docker is not running"
    exit 1
fi

docker_build_image "." "Dockerfile" "my-app:latest"
docker_start_container "my-app" "my-app:latest" ".env" "8000:8000"
```

### Using Kubernetes Library

```bash
#!/bin/bash
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/kubernetes.sh"

k8s_check_kubectl
k8s_check_namespace "production"
k8s_wait_for_deployment "my-app" "production" 300
```

## ✅ Benefits of Refactoring

1. **Modularity**: Functions are reusable across scripts
2. **Maintainability**: Changes in one place affect all scripts
3. **Testability**: Functions can be tested independently
4. **Consistency**: Same patterns across all scripts
5. **Error Handling**: Centralized error handling
6. **Logging**: Consistent logging format
7. **Validation**: Reusable validation functions

## 📝 Best Practices Applied

- ✅ POSIX-compliant syntax
- ✅ Error handling with trap
- ✅ Input validation
- ✅ Modular functions
- ✅ Comprehensive logging
- ✅ Temporary file cleanup
- ✅ Retry logic with backoff
- ✅ Health checks
- ✅ Environment variable validation

## 🔄 Migration Guide

### Old Script Pattern

```bash
# Old way
if ! command -v docker &> /dev/null; then
    echo "Docker not found"
    exit 1
fi

if curl -f http://localhost:8000/health; then
    echo "Healthy"
fi
```

### New Script Pattern

```bash
# New way
source "$(dirname "$0")/lib/common.sh"
init_common

validate_command docker docker.io
health_check "http://localhost:8000/health"
```

## 🧪 Testing

Test individual functions:

```bash
# Test common library
source lib/common.sh
init_common
validate_command docker docker.io

# Test docker library
source lib/docker.sh
docker_is_running

# Test kubernetes library
source lib/kubernetes.sh
k8s_check_kubectl
```

---

**Version**: 2.0.0  
**Last Updated**: 2024  
**Status**: ✅ Refactored & Production Ready




