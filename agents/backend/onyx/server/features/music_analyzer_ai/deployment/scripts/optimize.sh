#!/bin/bash
# Optimization Script for Music Analyzer AI
# Analyzes and optimizes performance, resources, and costs

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/kubernetes.sh"

# Initialize
init_common

# Configuration
readonly OPTIMIZATION_TYPE="${1:-all}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-./optimization-results}"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Optimize Docker images
optimize_docker_images() {
    log_info "Optimizing Docker images..."
    
    local images=("music-analyzer-ai-backend" "music-analyzer-ai-frontend")
    
    for image in "${images[@]}"; do
        log_info "Analyzing image: ${image}"
        
        # Check image size
        local size=$(docker images "${image}:latest" --format "{{.Size}}" 2>/dev/null || echo "N/A")
        log_info "Current size: ${size}"
        
        # Recommendations
        log_info "Recommendations:"
        log_info "  - Use multi-stage builds"
        log_info "  - Remove unnecessary dependencies"
        log_info "  - Use .dockerignore"
        log_info "  - Consider distroless images"
    done
    
    log_success "Docker image optimization analysis completed"
}

# Optimize resource requests/limits
optimize_resources() {
    log_info "Optimizing resource requests and limits..."
    
    if ! k8s_check_kubectl; then
        log_warn "kubectl not available, skipping resource optimization"
        return 0
    fi
    
    local namespace="${NAMESPACE:-production}"
    
    # Get current resource usage
    kubectl top pods -n "${namespace}" --no-headers 2>/dev/null | while read pod cpu mem; do
        log_info "Pod: ${pod}"
        log_info "  CPU Usage: ${cpu}"
        log_info "  Memory Usage: ${mem}"
        
        # Get requests/limits
        local requests=$(kubectl get pod "${pod}" -n "${namespace}" \
            -o jsonpath='{.spec.containers[0].resources.requests}' 2>/dev/null || echo "{}")
        local limits=$(kubectl get pod "${pod}" -n "${namespace}" \
            -o jsonpath='{.spec.containers[0].resources.limits}' 2>/dev/null || echo "{}")
        
        log_info "  Requests: ${requests}"
        log_info "  Limits: ${limits}"
    done
    
    log_success "Resource optimization analysis completed"
}

# Optimize database queries
optimize_database() {
    log_info "Optimizing database..."
    
    log_info "Recommendations:"
    log_info "  - Add indexes on frequently queried columns"
    log_info "  - Use connection pooling"
    log_info "  - Implement query caching"
    log_info "  - Analyze slow queries"
    log_info "  - Consider read replicas for read-heavy workloads"
    
    log_success "Database optimization recommendations generated"
}

# Optimize API endpoints
optimize_api() {
    log_info "Optimizing API endpoints..."
    
    log_info "Recommendations:"
    log_info "  - Implement response caching"
    log_info "  - Use pagination for large datasets"
    log_info "  - Compress responses (gzip)"
    log_info "  - Implement rate limiting"
    log_info "  - Use async processing for long operations"
    log_info "  - Optimize serialization"
    
    log_success "API optimization recommendations generated"
}

# Optimize frontend
optimize_frontend() {
    log_info "Optimizing frontend..."
    
    log_info "Recommendations:"
    log_info "  - Code splitting"
    log_info "  - Lazy loading"
    log_info "  - Image optimization"
    log_info "  - Bundle size reduction"
    log_info "  - CDN for static assets"
    log_info "  - Service worker for caching"
    
    log_success "Frontend optimization recommendations generated"
}

# Generate optimization report
generate_optimization_report() {
    log_info "Generating optimization report..."
    
    local report_file="${OUTPUT_DIR}/optimization-report-${TIMESTAMP}.md"
    
    cat > "${report_file}" << EOF
# Optimization Report

**Date:** $(date)
**Type:** ${OPTIMIZATION_TYPE}

## Docker Images

- Use multi-stage builds
- Remove unnecessary dependencies
- Optimize layer caching

## Resources

- Right-size based on actual usage
- Use HPA for auto-scaling
- Consider spot instances

## Database

- Add indexes
- Use connection pooling
- Implement caching

## API

- Response caching
- Pagination
- Compression
- Rate limiting

## Frontend

- Code splitting
- Lazy loading
- Image optimization
- CDN

EOF
    
    log_success "Optimization report generated: ${report_file}"
}

# Main function
main() {
    case "${OPTIMIZATION_TYPE}" in
        docker)
            optimize_docker_images
            ;;
        resources)
            optimize_resources
            ;;
        database)
            optimize_database
            ;;
        api)
            optimize_api
            ;;
        frontend)
            optimize_frontend
            ;;
        all|*)
            optimize_docker_images
            optimize_resources
            optimize_database
            optimize_api
            optimize_frontend
            generate_optimization_report
            ;;
    esac
}

main "$@"




