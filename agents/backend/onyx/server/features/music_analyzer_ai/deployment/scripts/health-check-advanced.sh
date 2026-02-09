#!/bin/bash
# Advanced Health Check Script
# Comprehensive health checking with detailed reporting

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/kubernetes.sh"

# Initialize
init_common

# Configuration
readonly HEALTH_CHECK_TYPE="${1:-full}"
readonly HEALTH_URL="${HEALTH_URL:-http://localhost:8000}"
readonly NAMESPACE="${NAMESPACE:-production}"
readonly OUTPUT_FILE="${OUTPUT_FILE:-health-report-$(date +%Y%m%d_%H%M%S).json}"

# Health check results
declare -A HEALTH_RESULTS

# Check basic health endpoint
check_basic_health() {
    log_info "Checking basic health endpoint..."
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}/health" 2>/dev/null || echo "000")
    
    if [ "${response}" == "200" ]; then
        HEALTH_RESULTS["basic"]="healthy"
        log_success "Basic health check passed"
        return 0
    else
        HEALTH_RESULTS["basic"]="unhealthy"
        log_error "Basic health check failed (HTTP ${response})"
        return 1
    fi
}

# Check readiness endpoint
check_readiness() {
    log_info "Checking readiness endpoint..."
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}/health/ready" 2>/dev/null || echo "000")
    
    if [ "${response}" == "200" ]; then
        HEALTH_RESULTS["readiness"]="ready"
        log_success "Readiness check passed"
        return 0
    else
        HEALTH_RESULTS["readiness"]="not-ready"
        log_error "Readiness check failed (HTTP ${response})"
        return 1
    fi
}

# Check liveness endpoint
check_liveness() {
    log_info "Checking liveness endpoint..."
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}/health/live" 2>/dev/null || echo "000")
    
    if [ "${response}" == "200" ]; then
        HEALTH_RESULTS["liveness"]="alive"
        log_success "Liveness check passed"
        return 0
    else
        HEALTH_RESULTS["liveness"]="not-alive"
        log_error "Liveness check failed (HTTP ${response})"
        return 1
    fi
}

# Check detailed health
check_detailed_health() {
    log_info "Checking detailed health endpoint..."
    
    local health_data=$(curl -s "${HEALTH_URL}/health/detailed" 2>/dev/null || echo "{}")
    
    if [ -n "${health_data}" ] && [ "${health_data}" != "{}" ]; then
        HEALTH_RESULTS["detailed"]="available"
        log_success "Detailed health data retrieved"
        echo "${health_data}" | jq '.' 2>/dev/null || echo "${health_data}"
        return 0
    else
        HEALTH_RESULTS["detailed"]="unavailable"
        log_warn "Detailed health endpoint not available"
        return 1
    fi
}

# Check container health
check_container_health() {
    log_info "Checking container health..."
    
    local containers=("music-analyzer-ai-backend" "music-analyzer-ai-frontend")
    local all_healthy=true
    
    for container in "${containers[@]}"; do
        local status=$(docker_get_container_status "${container}")
        
        if [ "${status}" == "running" ]; then
            log_success "Container ${container} is running"
            HEALTH_RESULTS["container_${container}"]="running"
        else
            log_error "Container ${container} status: ${status}"
            HEALTH_RESULTS["container_${container}"]="${status}"
            all_healthy=false
        fi
    done
    
    if [ "${all_healthy}" == "true" ]; then
        return 0
    else
        return 1
    fi
}

# Check Kubernetes health
check_kubernetes_health() {
    if ! k8s_check_kubectl; then
        log_warn "kubectl not available, skipping Kubernetes health check"
        return 0
    fi
    
    log_info "Checking Kubernetes deployment health..."
    
    local deployment="music-analyzer-ai-backend"
    local status=$(k8s_get_pod_status "${deployment}-0" "${NAMESPACE}" 2>/dev/null || echo "not-found")
    
    if [ "${status}" == "Running" ]; then
        HEALTH_RESULTS["kubernetes"]="healthy"
        log_success "Kubernetes deployment is healthy"
        return 0
    else
        HEALTH_RESULTS["kubernetes"]="unhealthy"
        log_error "Kubernetes deployment status: ${status}"
        return 1
    fi
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local dependencies=("database" "redis" "external-apis")
    local all_healthy=true
    
    for dep in "${dependencies[@]}"; do
        # This would check actual dependency health
        HEALTH_RESULTS["dependency_${dep}"]="unknown"
    done
    
    log_success "Dependency check completed"
}

# Generate health report
generate_health_report() {
    log_info "Generating health report..."
    
    local report_data=$(cat <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "environment": "${ENVIRONMENT:-unknown}",
  "health_checks": {
$(for key in "${!HEALTH_RESULTS[@]}"; do
    echo "    \"${key}\": \"${HEALTH_RESULTS[$key]}\","
done | sed '$ s/,$//')
  },
  "overall_status": "$(determine_overall_status)"
}
EOF
)
    
    echo "${report_data}" | jq '.' > "${OUTPUT_FILE}" 2>/dev/null || echo "${report_data}" > "${OUTPUT_FILE}"
    
    log_success "Health report generated: ${OUTPUT_FILE}"
}

# Determine overall status
determine_overall_status() {
    local unhealthy_count=0
    
    for status in "${HEALTH_RESULTS[@]}"; do
        if [[ "${status}" =~ (unhealthy|not-ready|not-alive|failed|error) ]]; then
            unhealthy_count=$((unhealthy_count + 1))
        fi
    done
    
    if [ ${unhealthy_count} -eq 0 ]; then
        echo "healthy"
    elif [ ${unhealthy_count} -le 2 ]; then
        echo "degraded"
    else
        echo "unhealthy"
    fi
}

# Main function
main() {
    case "${HEALTH_CHECK_TYPE}" in
        basic)
            check_basic_health
            ;;
        readiness)
            check_readiness
            ;;
        liveness)
            check_liveness
            ;;
        detailed)
            check_detailed_health
            ;;
        container)
            check_container_health
            ;;
        kubernetes)
            check_kubernetes_health
            ;;
        dependencies)
            check_dependencies
            ;;
        full|*)
            check_basic_health
            check_readiness
            check_liveness
            check_detailed_health
            check_container_health
            check_kubernetes_health
            check_dependencies
            generate_health_report
            ;;
    esac
}

main "$@"




