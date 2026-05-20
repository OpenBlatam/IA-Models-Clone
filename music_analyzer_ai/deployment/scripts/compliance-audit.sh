#!/bin/bash
# Compliance and Security Audit Script
# Checks for security best practices and compliance

set -euo pipefail

# Configuration
readonly NAMESPACE="${NAMESPACE:-production}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-./compliance-reports}"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check security contexts
check_security_contexts() {
    log_info "Checking security contexts..."
    
    local issues=0
    
    kubectl get pods -n "${NAMESPACE}" -o json | jq -r '
        .items[] |
        select(.spec.containers[0].securityContext.runAsNonRoot != true) |
        .metadata.name
    ' | while read pod; do
        log_error "Pod ${pod} does not run as non-root"
        issues=$((issues + 1))
    done
    
    if [ ${issues} -eq 0 ]; then
        log_success "All pods run as non-root"
    fi
}

# Check resource limits
check_resource_limits() {
    log_info "Checking resource limits..."
    
    local issues=0
    
    kubectl get pods -n "${NAMESPACE}" -o json | jq -r '
        .items[] |
        select(.spec.containers[0].resources.limits == null) |
        .metadata.name
    ' | while read pod; do
        log_error "Pod ${pod} does not have resource limits"
        issues=$((issues + 1))
    done
    
    if [ ${issues} -eq 0 ]; then
        log_success "All pods have resource limits"
    fi
}

# Check network policies
check_network_policies() {
    log_info "Checking network policies..."
    
    local np_count=$(kubectl get networkpolicies -n "${NAMESPACE}" --no-headers 2>/dev/null | wc -l)
    
    if [ ${np_count} -eq 0 ]; then
        log_warn "No network policies found in namespace ${NAMESPACE}"
    else
        log_success "Found ${np_count} network policies"
    fi
}

# Check image security
check_image_security() {
    log_info "Checking image security..."
    
    kubectl get pods -n "${NAMESPACE}" -o json | jq -r '
        .items[] |
        {
            pod: .metadata.name,
            image: .spec.containers[0].image,
            imagePullPolicy: .spec.containers[0].imagePullPolicy
        } |
        "\(.pod)|\(.image)|\(.imagePullPolicy)"
    ' | while IFS='|' read pod image policy; do
        if [ "${policy}" != "Always" ] && [[ "${image}" != *":latest"* ]]; then
            log_warn "Pod ${pod} uses image ${image} with policy ${policy}"
        fi
    done
}

# Check secrets management
check_secrets() {
    log_info "Checking secrets management..."
    
    # Check for hardcoded secrets (basic check)
    kubectl get pods -n "${NAMESPACE}" -o json | jq -r '
        .items[] |
        .spec.containers[0].env[]? |
        select(.value != null) |
        select(.value | test("password|secret|key|token"; "i")) |
        "\(.name): potential hardcoded secret"
    ' | while read line; do
        log_warn "${line}"
    done
}

# Check RBAC
check_rbac() {
    log_info "Checking RBAC..."
    
    # Check for overly permissive roles
    kubectl get roles,rolebindings -n "${NAMESPACE}" -o json | jq -r '
        .items[] |
        select(.rules[]?.verbs[]? == "*") |
        .metadata.name
    ' | while read role; do
        log_warn "Role ${role} has wildcard permissions"
    done
}

# Generate compliance report
generate_report() {
    log_info "Generating compliance report..."
    
    mkdir -p "${OUTPUT_DIR}"
    local report_file="${OUTPUT_DIR}/compliance-report-${TIMESTAMP}.md"
    
    cat > "${report_file}" << EOF
# Compliance and Security Audit Report

**Date:** $(date)
**Namespace:** ${NAMESPACE}

## Security Contexts

$(check_security_contexts)

## Resource Limits

$(check_resource_limits)

## Network Policies

$(check_network_policies)

## Image Security

$(check_image_security)

## Secrets Management

$(check_secrets)

## RBAC

$(check_rbac)

## Recommendations

1. Ensure all pods run as non-root
2. Set resource limits for all containers
3. Implement network policies
4. Use image pull policies appropriately
5. Use secrets management (Vault, Sealed Secrets)
6. Follow principle of least privilege for RBAC

EOF
    
    log_success "Report generated: ${report_file}"
}

# Main function
main() {
    case "${1:-all}" in
        security-contexts)
            check_security_contexts
            ;;
        resource-limits)
            check_resource_limits
            ;;
        network-policies)
            check_network_policies
            ;;
        images)
            check_image_security
            ;;
        secrets)
            check_secrets
            ;;
        rbac)
            check_rbac
            ;;
        report|all)
            check_security_contexts
            check_resource_limits
            check_network_policies
            check_image_security
            check_secrets
            check_rbac
            generate_report
            ;;
        *)
            echo "Usage: $0 {security-contexts|resource-limits|network-policies|images|secrets|rbac|report|all}"
            exit 1
            ;;
    esac
}

main "$@"




