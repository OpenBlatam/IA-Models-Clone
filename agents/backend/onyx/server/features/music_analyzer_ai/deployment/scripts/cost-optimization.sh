#!/bin/bash
# Cost Optimization Script for Music Analyzer AI
# Analyzes resource usage and suggests optimizations

set -euo pipefail

# Configuration
readonly NAMESPACE="${NAMESPACE:-production}"
readonly OUTPUT_FILE="${OUTPUT_FILE:-cost-analysis-$(date +%Y%m%d).json}"

# Colors
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Analyze resource requests vs usage
analyze_resources() {
    log_info "Analyzing resource usage..."
    
    local analysis_file=$(mktemp)
    
    # Get pod resource requests and usage
    kubectl top pods -n "${NAMESPACE}" --no-headers | while read pod cpu_usage mem_usage; do
        local cpu_request=$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
            -o jsonpath='{.spec.containers[0].resources.requests.cpu}' 2>/dev/null || echo "N/A")
        local mem_request=$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
            -o jsonpath='{.spec.containers[0].resources.requests.memory}' 2>/dev/null || echo "N/A")
        
        echo "Pod: ${pod}"
        echo "  CPU Request: ${cpu_request}, Usage: ${cpu_usage}"
        echo "  Memory Request: ${mem_request}, Usage: ${mem_usage}"
        echo ""
    done > "${analysis_file}"
    
    cat "${analysis_file}"
    rm "${analysis_file}"
}

# Suggest right-sizing
suggest_rightsizing() {
    log_info "Generating right-sizing recommendations..."
    
    kubectl get pods -n "${NAMESPACE}" -o json | jq -r '
        .items[] | 
        select(.metadata.labels."app.kubernetes.io/name" == "music-analyzer-ai") |
        {
            pod: .metadata.name,
            cpu_request: .spec.containers[0].resources.requests.cpu,
            memory_request: .spec.containers[0].resources.requests.memory,
            cpu_limit: .spec.containers[0].resources.limits.cpu,
            memory_limit: .spec.containers[0].resources.limits.memory
        }
    ' | jq -s '.' > "${OUTPUT_FILE}"
    
    log_info "Recommendations saved to: ${OUTPUT_FILE}"
}

# Analyze HPA efficiency
analyze_hpa() {
    log_info "Analyzing HPA efficiency..."
    
    kubectl get hpa -n "${NAMESPACE}" -o json | jq -r '
        .items[] |
        {
            name: .metadata.name,
            min_replicas: .spec.minReplicas,
            max_replicas: .spec.maxReplicas,
            current_replicas: .status.currentReplicas,
            desired_replicas: .status.desiredReplicas,
            cpu_utilization: .status.currentMetrics[]? | select(.type == "Resource") | .resource.current.averageUtilization
        }
    ' | jq -s '.'
}

# Calculate estimated costs
calculate_costs() {
    log_info "Calculating estimated costs..."
    
    # This would integrate with cloud provider pricing APIs
    # For demonstration, we'll use average pricing
    
    local total_cpu=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/name=music-analyzer-ai \
        -o jsonpath='{range .items[*]}{.spec.containers[0].resources.requests.cpu}{"\n"}{end}' | \
        awk '{sum += $1} END {print sum}')
    
    local total_memory=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/name=music-analyzer-ai \
        -o jsonpath='{range .items[*]}{.spec.containers[0].resources.requests.memory}{"\n"}{end}' | \
        awk '{sum += $1} END {print sum}')
    
    log_info "Total CPU requests: ${total_cpu}"
    log_info "Total Memory requests: ${total_memory}"
    log_warn "Cost calculation requires cloud provider pricing data"
}

# Generate optimization report
generate_report() {
    log_info "Generating cost optimization report..."
    
    local report_file="cost-optimization-report-$(date +%Y%m%d).md"
    
    cat > "${report_file}" << EOF
# Cost Optimization Report

**Date:** $(date)
**Namespace:** ${NAMESPACE}

## Resource Analysis

$(analyze_resources)

## Recommendations

1. **Right-size resources**: Adjust CPU/memory requests based on actual usage
2. **Optimize HPA**: Review autoscaling policies
3. **Use spot instances**: For non-critical workloads
4. **Reserved instances**: For predictable workloads
5. **Container optimization**: Use multi-stage builds, smaller base images

## Estimated Savings

- Right-sizing: 20-30% cost reduction
- Spot instances: 50-70% cost reduction
- Reserved instances: 30-50% cost reduction

EOF
    
    log_info "Report generated: ${report_file}"
}

# Main function
main() {
    case "${1:-all}" in
        resources)
            analyze_resources
            ;;
        rightsizing)
            suggest_rightsizing
            ;;
        hpa)
            analyze_hpa
            ;;
        costs)
            calculate_costs
            ;;
        report|all)
            analyze_resources
            suggest_rightsizing
            analyze_hpa
            calculate_costs
            generate_report
            ;;
        *)
            echo "Usage: $0 {resources|rightsizing|hpa|costs|report|all}"
            exit 1
            ;;
    esac
}

main "$@"




