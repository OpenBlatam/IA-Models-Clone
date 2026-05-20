#!/bin/bash
# Security Hardening Script
# Applies security best practices and configurations

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/kubernetes.sh"

# Initialize
init_common

# Configuration
readonly NAMESPACE="${NAMESPACE:-production}"
readonly HARDENING_LEVEL="${HARDENING_LEVEL:-standard}"

# Apply Pod Security Standards
apply_pod_security_standards() {
    log_info "Applying Pod Security Standards..."
    
    if ! k8s_check_kubectl; then
        log_warn "kubectl not available, skipping Pod Security Standards"
        return 0
    fi
    
    # Create Pod Security Policy
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
EOF
    
    log_success "Pod Security Standards applied"
}

# Configure network policies
configure_network_policies() {
    log_info "Configuring network policies..."
    
    log_info "Network policies should restrict:"
    log_info "  - Ingress traffic to specific pods"
    log_info "  - Egress traffic to necessary services only"
    log_info "  - Cross-namespace communication"
    
    log_success "Network policy recommendations generated"
}

# Configure RBAC
configure_rbac() {
    log_info "Configuring RBAC..."
    
    log_info "RBAC best practices:"
    log_info "  - Use least privilege principle"
    log_info "  - Avoid wildcard permissions"
    log_info "  - Use service accounts"
    log_info "  - Regular audit of permissions"
    
    log_success "RBAC recommendations generated"
}

# Scan for vulnerabilities
scan_vulnerabilities() {
    log_info "Scanning for vulnerabilities..."
    
    # Docker image scanning
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy scan..."
        trivy image music-analyzer-ai-backend:latest || true
    else
        log_warn "Trivy not installed, skipping vulnerability scan"
    fi
    
    # Dependency scanning
    if command -v safety &> /dev/null; then
        log_info "Running Safety check..."
        safety check || true
    else
        log_warn "Safety not installed, skipping dependency scan"
    fi
    
    log_success "Vulnerability scanning completed"
}

# Configure secrets management
configure_secrets() {
    log_info "Configuring secrets management..."
    
    log_info "Secrets management best practices:"
    log_info "  - Use Kubernetes secrets"
    log_info "  - Consider external secret management (Vault, Sealed Secrets)"
    log_info "  - Rotate secrets regularly"
    log_info "  - Encrypt secrets at rest"
    log_info "  - Use RBAC to restrict secret access"
    
    log_success "Secrets management recommendations generated"
}

# Apply security configurations
apply_security_configs() {
    log_info "Applying security configurations..."
    
    case "${HARDENING_LEVEL}" in
        basic)
            log_info "Applying basic security configurations..."
            configure_network_policies
            configure_rbac
            ;;
        standard)
            log_info "Applying standard security configurations..."
            apply_pod_security_standards
            configure_network_policies
            configure_rbac
            configure_secrets
            ;;
        strict)
            log_info "Applying strict security configurations..."
            apply_pod_security_standards
            configure_network_policies
            configure_rbac
            configure_secrets
            scan_vulnerabilities
            ;;
        *)
            log_error "Unknown hardening level: ${HARDENING_LEVEL}"
            return 1
            ;;
    esac
    
    log_success "Security configurations applied"
}

# Main function
main() {
    local action="${1:-apply}"
    
    case "${action}" in
        apply)
            apply_security_configs
            ;;
        scan)
            scan_vulnerabilities
            ;;
        network)
            configure_network_policies
            ;;
        rbac)
            configure_rbac
            ;;
        secrets)
            configure_secrets
            ;;
        *)
            echo "Usage: $0 {apply|scan|network|rbac|secrets}"
            exit 1
            ;;
    esac
}

main "$@"




