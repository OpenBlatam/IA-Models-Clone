#!/bin/bash

# =============================================================================
# Production Deployment Script for Next-Level HeyGen AI FastAPI
# Comprehensive deployment with security, backup, monitoring, and rollback
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
DEPLOYMENT_NAME="heygen-ai-$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_DIR="${PROJECT_ROOT}/config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Deployment configuration
DEPLOYMENT_CONFIG=(
    "ENVIRONMENT=production"
    "OPTIMIZATION_TIER=3"
    "PROFILING_LEVEL=1"
    "ENABLE_GPU_OPTIMIZATION=true"
    "ENABLE_REDIS=true"
    "ENABLE_REQUEST_BATCHING=true"
    "ENABLE_PERFORMANCE_PROFILING=true"
    "MAX_CONCURRENT_REQUESTS=1000"
    "DEFAULT_BATCH_SIZE=8"
    "WORKERS=4"
    "LOG_LEVEL=info"
)

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# =============================================================================
# Pre-deployment Checks
# =============================================================================

check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check required environment variables
    local required_vars=(
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "SECRET_KEY"
        "JWT_SECRET"
        "OPENROUTER_API_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Check disk space (minimum 10GB)
    local available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -lt 10 ]]; then
        error "Insufficient disk space. Available: ${available_space}GB, Required: 10GB"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    log "✅ Prerequisites check passed"
}

check_security() {
    log "Performing security checks..."
    
    # Check for sensitive files
    local sensitive_files=(
        ".env"
        "config/production.env"
        "*.pem"
        "*.key"
        "secrets/"
    )
    
    for pattern in "${sensitive_files[@]}"; do
        if find "${PROJECT_ROOT}" -name "$pattern" -type f 2>/dev/null | grep -q .; then
            warn "Sensitive files found matching pattern: $pattern"
        fi
    done
    
    # Check file permissions
    local critical_files=(
        "main_next_level.py"
        "Dockerfile.production"
        "docker-compose.production.yml"
    )
    
    for file in "${critical_files[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${file}" ]]; then
            local perms=$(stat -c "%a" "${PROJECT_ROOT}/${file}")
            if [[ $perms != "644" ]]; then
                warn "File ${file} has unusual permissions: $perms"
            fi
        fi
    done
    
    # Check for hardcoded secrets
    if grep -r "password\|secret\|key" "${PROJECT_ROOT}" --include="*.py" --include="*.yml" --include="*.yaml" | grep -v "YOUR_" | grep -v "example" | grep -v "placeholder"; then
        warn "Potential hardcoded secrets found in code"
    fi
    
    log "✅ Security checks completed"
}

# =============================================================================
# Backup and Rollback Management
# =============================================================================

create_backup() {
    log "Creating deployment backup..."
    
    local backup_path="${BACKUP_DIR}/${DEPLOYMENT_NAME}"
    mkdir -p "${backup_path}"
    
    # Backup current deployment
    if docker-compose -f "${PROJECT_ROOT}/docker-compose.production.yml" ps -q | grep -q .; then
        log "Backing up current deployment state..."
        
        # Export current images
        docker save heygen-ai:latest -o "${backup_path}/heygen-ai-image.tar"
        
        # Backup volumes
        docker run --rm -v heygen_postgres_data:/data -v "${backup_path}":/backup alpine tar czf /backup/postgres-data.tar.gz -C /data .
        docker run --rm -v heygen_redis_data:/data -v "${backup_path}":/backup alpine tar czf /backup/redis-data.tar.gz -C /data .
        
        # Backup configuration
        cp -r "${PROJECT_ROOT}/config" "${backup_path}/"
        cp "${PROJECT_ROOT}/docker-compose.production.yml" "${backup_path}/"
        
        # Create backup manifest
        cat > "${backup_path}/backup-manifest.json" << EOF
{
    "backup_name": "${DEPLOYMENT_NAME}",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployment_state": "pre-deployment",
    "images": ["heygen-ai:latest"],
    "volumes": ["postgres_data", "redis_data"],
    "config_files": ["config/", "docker-compose.production.yml"]
}
EOF
        
        log "✅ Backup created: ${backup_path}"
    else
        log "No existing deployment to backup"
    fi
}

rollback_deployment() {
    local backup_name="${1:-}"
    
    if [[ -z "$backup_name" ]]; then
        error "Backup name is required for rollback"
        exit 1
    fi
    
    local backup_path="${BACKUP_DIR}/${backup_name}"
    if [[ ! -d "$backup_path" ]]; then
        error "Backup directory not found: ${backup_path}"
        exit 1
    fi
    
    log "Rolling back to backup: ${backup_name}"
    
    # Stop current deployment
    docker-compose -f "${PROJECT_ROOT}/docker-compose.production.yml" down
    
    # Restore images
    if [[ -f "${backup_path}/heygen-ai-image.tar" ]]; then
        docker load -i "${backup_path}/heygen-ai-image.tar"
    fi
    
    # Restore volumes
    if [[ -f "${backup_path}/postgres-data.tar.gz" ]]; then
        docker run --rm -v heygen_postgres_data:/data -v "${backup_path}":/backup alpine tar xzf /backup/postgres-data.tar.gz -C /data
    fi
    
    if [[ -f "${backup_path}/redis-data.tar.gz" ]]; then
        docker run --rm -v heygen_redis_data:/data -v "${backup_path}":/backup alpine tar xzf /backup/redis-data.tar.gz -C /data
    fi
    
    # Restore configuration
    if [[ -d "${backup_path}/config" ]]; then
        cp -r "${backup_path}/config" "${PROJECT_ROOT}/"
    fi
    
    # Restart deployment
    docker-compose -f "${PROJECT_ROOT}/docker-compose.production.yml" up -d
    
    log "✅ Rollback completed successfully"
}

# =============================================================================
# Deployment Process
# =============================================================================

build_images() {
    log "Building production Docker images..."
    
    cd "${PROJECT_ROOT}"
    
    # Build with production optimizations
    docker build \
        --target production \
        --build-arg OPTIMIZATION_TIER=3 \
        --build-arg ENABLE_GPU=true \
        --build-arg BUILD_ENV=production \
        -f Dockerfile.production \
        -t heygen-ai:latest \
        -t heygen-ai:${DEPLOYMENT_NAME} \
        .
    
    log "✅ Docker images built successfully"
}

deploy_services() {
    log "Deploying production services..."
    
    cd "${PROJECT_ROOT}"
    
    # Deploy with production configuration
    docker-compose -f docker-compose.production.yml up -d
    
    log "✅ Services deployed successfully"
}

wait_for_health() {
    log "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s "http://localhost:8000/health" > /dev/null 2>&1; then
            log "✅ Application is healthy"
            return 0
        fi
        
        attempt=$((attempt + 1))
        info "Waiting for health check... (${attempt}/${max_attempts})"
        sleep 10
    done
    
    error "Health check failed after ${max_attempts} attempts"
    return 1
}

run_smoke_tests() {
    log "Running smoke tests..."
    
    local tests=(
        "curl -f http://localhost:8000/health"
        "curl -f http://localhost:8000/metrics/performance"
        "curl -f http://localhost:9091/-/healthy"
        "curl -f http://localhost:3000/api/health"
    )
    
    local failed_tests=0
    
    for test in "${tests[@]}"; do
        if eval "$test" > /dev/null 2>&1; then
            log "✅ Smoke test passed: $test"
        else
            error "❌ Smoke test failed: $test"
            failed_tests=$((failed_tests + 1))
        fi
    done
    
    if [[ $failed_tests -gt 0 ]]; then
        error "Smoke tests failed: ${failed_tests} tests failed"
        return 1
    fi
    
    log "✅ All smoke tests passed"
}

# =============================================================================
# Post-deployment Verification
# =============================================================================

verify_deployment() {
    log "Verifying deployment..."
    
    # Check service status
    local services=(
        "heygen-ai-app"
        "postgres"
        "redis"
        "prometheus"
        "grafana"
        "alertmanager"
    )
    
    for service in "${services[@]}"; do
        if docker-compose -f "${PROJECT_ROOT}/docker-compose.production.yml" ps "$service" | grep -q "Up"; then
            log "✅ Service $service is running"
        else
            error "❌ Service $service is not running"
            return 1
        fi
    done
    
    # Check application metrics
    local metrics_response=$(curl -s http://localhost:8000/metrics/performance)
    if echo "$metrics_response" | jq -e '.optimization_tier' > /dev/null 2>&1; then
        log "✅ Application metrics endpoint is responding"
    else
        error "❌ Application metrics endpoint is not responding correctly"
        return 1
    fi
    
    # Check monitoring stack
    if curl -f -s http://localhost:9091/-/healthy > /dev/null 2>&1; then
        log "✅ Prometheus is healthy"
    else
        error "❌ Prometheus is not healthy"
        return 1
    fi
    
    if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
        log "✅ Grafana is healthy"
    else
        error "❌ Grafana is not healthy"
        return 1
    fi
    
    log "✅ Deployment verification completed successfully"
}

# =============================================================================
# Monitoring and Alerts Setup
# =============================================================================

setup_monitoring() {
    log "Setting up monitoring and alerts..."
    
    # Wait for Prometheus to be ready
    sleep 30
    
    # Check if Prometheus is collecting metrics
    local metrics_count=$(curl -s http://localhost:9091/api/v1/query?query=up | jq '.data.result | length')
    if [[ $metrics_count -gt 0 ]]; then
        log "✅ Prometheus is collecting ${metrics_count} metrics"
    else
        warn "⚠️ Prometheus is not collecting metrics yet"
    fi
    
    # Check AlertManager
    if curl -f -s http://localhost:9093/-/healthy > /dev/null 2>&1; then
        log "✅ AlertManager is healthy"
    else
        warn "⚠️ AlertManager is not healthy"
    fi
    
    log "✅ Monitoring setup completed"
}

# =============================================================================
# Cleanup and Maintenance
# =============================================================================

cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Keep only last 5 backups
    local backup_count=$(ls -1 "${BACKUP_DIR}" | wc -l)
    if [[ $backup_count -gt 5 ]]; then
        local backups_to_remove=$(ls -1t "${BACKUP_DIR}" | tail -n +6)
        for backup in $backups_to_remove; do
            rm -rf "${BACKUP_DIR}/${backup}"
            log "Removed old backup: ${backup}"
        done
    fi
    
    # Clean up old Docker images
    docker image prune -f --filter "until=24h"
    
    log "✅ Cleanup completed"
}

# =============================================================================
# Main Deployment Function
# =============================================================================

main() {
    log "🚀 Starting Next-Level HeyGen AI Production Deployment"
    log "=================================================="
    log "Deployment Name: ${DEPLOYMENT_NAME}"
    log "Project Root: ${PROJECT_ROOT}"
    log "Backup Directory: ${BACKUP_DIR}"
    log "=================================================="
    
    # Pre-deployment phase
    check_prerequisites
    check_security
    create_backup
    
    # Deployment phase
    build_images
    deploy_services
    
    # Post-deployment phase
    wait_for_health
    run_smoke_tests
    verify_deployment
    setup_monitoring
    
    # Cleanup
    cleanup_old_backups
    
    log "=================================================="
    log "🎉 Production Deployment Completed Successfully!"
    log "=================================================="
    log "Application URL: http://localhost:8000"
    log "API Documentation: http://localhost:8000/docs"
    log "Grafana Dashboard: http://localhost:3000"
    log "Prometheus Metrics: http://localhost:9091"
    log "AlertManager: http://localhost:9093"
    log "=================================================="
    log "Backup Name: ${DEPLOYMENT_NAME}"
    log "Rollback Command: $0 rollback ${DEPLOYMENT_NAME}"
    log "=================================================="
}

# =============================================================================
# Command Line Interface
# =============================================================================

case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        if [[ -z "${2:-}" ]]; then
            error "Backup name is required for rollback"
            echo "Usage: $0 rollback <backup-name>"
            exit 1
        fi
        rollback_deployment "$2"
        ;;
    "backup")
        create_backup
        ;;
    "health")
        wait_for_health
        ;;
    "test")
        run_smoke_tests
        ;;
    "verify")
        verify_deployment
        ;;
    "cleanup")
        cleanup_old_backups
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|backup|health|test|verify|cleanup}"
        echo ""
        echo "Commands:"
        echo "  deploy    - Deploy the application to production"
        echo "  rollback  - Rollback to a previous backup"
        echo "  backup    - Create a backup of current deployment"
        echo "  health    - Wait for health checks"
        echo "  test      - Run smoke tests"
        echo "  verify    - Verify deployment status"
        echo "  cleanup   - Clean up old backups and images"
        exit 1
        ;;
esac 