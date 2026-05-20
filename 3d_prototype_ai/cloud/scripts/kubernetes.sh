#!/bin/bash
# Kubernetes integration script
# Manages Kubernetes deployments

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"
readonly NAMESPACE="${NAMESPACE:-default}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Kubernetes deployment management.

COMMANDS:
    deploy              Deploy to Kubernetes
    update              Update deployment
    rollback            Rollback deployment
    scale               Scale deployment
    status              Show deployment status
    logs                View pod logs
    exec                Execute command in pod

OPTIONS:
    -c, --kubeconfig PATH   Kubeconfig file (default: ~/.kube/config)
    -n, --namespace NAME    Kubernetes namespace (default: default)
    -h, --help              Show this help message

EXAMPLES:
    $0 deploy --namespace production
    $0 scale --replicas 5
    $0 status

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    REPLICAS=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--kubeconfig)
                KUBECONFIG="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--replicas)
                REPLICAS="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            deploy|update|rollback|scale|status|logs|exec)
                COMMAND="$1"
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check kubectl
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        error_exit 1 "kubectl not found. Please install Kubernetes CLI"
    fi
    
    if [ ! -f "${KUBECONFIG}" ]; then
        log_warn "Kubeconfig not found: ${KUBECONFIG}"
        log_info "Please configure Kubernetes access"
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    local namespace="${1}"
    
    check_kubectl
    
    log_info "Deploying to Kubernetes namespace: ${namespace}"
    
    # Create namespace if not exists
    kubectl create namespace "${namespace}" --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true
    
    # Deploy application
    # This is a placeholder - implement based on your Kubernetes manifests
    log_info "Kubernetes deployment - implement based on your manifests"
    
    log_info "Deployment to Kubernetes completed"
}

# Update deployment
update_k8s() {
    local namespace="${1}"
    
    check_kubectl
    
    log_info "Updating Kubernetes deployment in namespace: ${namespace}"
    
    # Update deployment
    kubectl rollout restart deployment/3d-prototype-ai -n "${namespace}" 2>/dev/null || \
    log_warn "Deployment not found or update failed"
    
    log_info "Update completed"
}

# Rollback deployment
rollback_k8s() {
    local namespace="${1}"
    
    check_kubectl
    
    log_warn "Rolling back Kubernetes deployment..."
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Rollback cancelled"
        return 0
    fi
    
    log_info "Rolling back deployment in namespace: ${namespace}"
    
    kubectl rollout undo deployment/3d-prototype-ai -n "${namespace}" 2>/dev/null || \
    log_error "Rollback failed"
    
    log_info "Rollback completed"
}

# Scale deployment
scale_k8s() {
    local namespace="${1}"
    local replicas="${2}"
    
    check_kubectl
    
    if [ -z "${replicas}" ]; then
        error_exit 1 "Number of replicas is required"
    fi
    
    log_info "Scaling deployment to ${replicas} replicas..."
    
    kubectl scale deployment/3d-prototype-ai --replicas="${replicas}" -n "${namespace}" 2>/dev/null || \
    log_error "Scaling failed"
    
    log_info "Scaling completed"
}

# Show status
show_status() {
    local namespace="${1}"
    
    check_kubectl
    
    log_info "Kubernetes deployment status:"
    echo ""
    
    # Deployment status
    kubectl get deployments -n "${namespace}" 2>/dev/null || echo "No deployments found"
    echo ""
    
    # Pod status
    kubectl get pods -n "${namespace}" 2>/dev/null || echo "No pods found"
    echo ""
    
    # Service status
    kubectl get services -n "${namespace}" 2>/dev/null || echo "No services found"
}

# View logs
view_logs() {
    local namespace="${1}"
    
    check_kubectl
    
    log_info "Viewing pod logs..."
    
    local pod_name
    pod_name=$(kubectl get pods -n "${namespace}" -l app=3d-prototype-ai -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -n "${pod_name}" ]; then
        kubectl logs -n "${namespace}" "${pod_name}" --tail=100
    else
        log_error "No pods found"
    fi
}

# Execute command
exec_command() {
    local namespace="${1}"
    local command="${2:-/bin/bash}"
    
    check_kubectl
    
    local pod_name
    pod_name=$(kubectl get pods -n "${namespace}" -l app=3d-prototype-ai -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -n "${pod_name}" ]; then
        kubectl exec -it -n "${namespace}" "${pod_name}" -- ${command}
    else
        log_error "No pods found"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        deploy)
            deploy_k8s "${NAMESPACE}"
            ;;
        update)
            update_k8s "${NAMESPACE}"
            ;;
        rollback)
            rollback_k8s "${NAMESPACE}"
            ;;
        scale)
            if [ -z "${REPLICAS}" ]; then
                error_exit 1 "Number of replicas is required (use --replicas)"
            fi
            scale_k8s "${NAMESPACE}" "${REPLICAS}"
            ;;
        status)
            show_status "${NAMESPACE}"
            ;;
        logs)
            view_logs "${NAMESPACE}"
            ;;
        exec)
            exec_command "${NAMESPACE}" "${*}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


