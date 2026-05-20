#!/bin/bash
# Kubernetes Utility Library
# Reusable Kubernetes functions for deployment scripts

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Kubernetes functions
k8s_check_kubectl() {
    validate_command kubectl kubectl
}

k8s_check_namespace() {
    local namespace="$1"
    
    if ! kubectl get namespace "${namespace}" &> /dev/null; then
        log_info "Creating namespace: ${namespace}"
        kubectl create namespace "${namespace}"
    fi
}

k8s_wait_for_deployment() {
    local deployment_name="$1"
    local namespace="${2:-default}"
    local timeout="${3:-300}"
    
    log_info "Waiting for deployment ${deployment_name} in namespace ${namespace}"
    
    if kubectl wait --for=condition=available \
        deployment/"${deployment_name}" \
        -n "${namespace}" \
        --timeout="${timeout}s"; then
        log_success "Deployment ${deployment_name} is ready"
        return 0
    else
        log_error "Deployment ${deployment_name} failed to become ready"
        return 1
    fi
}

k8s_wait_for_pod() {
    local pod_selector="$1"
    local namespace="${2:-default}"
    local timeout="${3:-300}"
    
    log_info "Waiting for pods matching: ${pod_selector}"
    
    if kubectl wait --for=condition=Ready \
        pod -l "${pod_selector}" \
        -n "${namespace}" \
        --timeout="${timeout}s"; then
        log_success "Pods are ready"
        return 0
    else
        log_error "Pods failed to become ready"
        return 1
    fi
}

k8s_get_pod_status() {
    local pod_name="$1"
    local namespace="${2:-default}"
    
    kubectl get pod "${pod_name}" -n "${namespace}" \
        -o jsonpath='{.status.phase}' 2>/dev/null || echo "not-found"
}

k8s_get_pod_logs() {
    local pod_name="$1"
    local namespace="${2:-default}"
    local lines="${3:-50}"
    
    kubectl logs "${pod_name}" -n "${namespace}" --tail="${lines}" 2>&1
}

k8s_scale_deployment() {
    local deployment_name="$1"
    local replicas="$2"
    local namespace="${3:-default}"
    
    log_info "Scaling deployment ${deployment_name} to ${replicas} replicas"
    
    if kubectl scale deployment "${deployment_name}" \
        --replicas="${replicas}" \
        -n "${namespace}"; then
        log_success "Deployment scaled successfully"
        return 0
    else
        log_error "Failed to scale deployment"
        return 1
    fi
}

k8s_rollout_restart() {
    local deployment_name="$1"
    local namespace="${2:-default}"
    
    log_info "Restarting deployment ${deployment_name}"
    
    if kubectl rollout restart deployment/"${deployment_name}" -n "${namespace}"; then
        log_success "Deployment restarted successfully"
        return 0
    else
        log_error "Failed to restart deployment"
        return 1
    fi
}

k8s_rollout_status() {
    local deployment_name="$1"
    local namespace="${2:-default}"
    
    kubectl rollout status deployment/"${deployment_name}" -n "${namespace}"
}

k8s_rollback_deployment() {
    local deployment_name="$1"
    local namespace="${2:-default}"
    
    log_warn "Rolling back deployment ${deployment_name}"
    
    if kubectl rollout undo deployment/"${deployment_name}" -n "${namespace}"; then
        log_success "Deployment rolled back successfully"
        return 0
    else
        log_error "Failed to rollback deployment"
        return 1
    fi
}

k8s_apply_manifest() {
    local manifest_file="$1"
    
    log_info "Applying Kubernetes manifest: ${manifest_file}"
    
    if kubectl apply -f "${manifest_file}"; then
        log_success "Manifest applied successfully"
        return 0
    else
        log_error "Failed to apply manifest"
        return 1
    fi
}

k8s_delete_resource() {
    local resource_type="$1"
    local resource_name="$2"
    local namespace="${3:-default}"
    
    log_info "Deleting ${resource_type}/${resource_name} in namespace ${namespace}"
    
    if kubectl delete "${resource_type}" "${resource_name}" -n "${namespace}"; then
        log_success "Resource deleted successfully"
        return 0
    else
        log_error "Failed to delete resource"
        return 1
    fi
}

k8s_get_resource_yaml() {
    local resource_type="$1"
    local resource_name="$2"
    local namespace="${3:-default}"
    
    kubectl get "${resource_type}" "${resource_name}" -n "${namespace}" -o yaml
}

k8s_port_forward() {
    local resource_type="$1"
    local resource_name="$2"
    local local_port="$3"
    local remote_port="$4"
    local namespace="${5:-default}"
    
    log_info "Port forwarding ${resource_type}/${resource_name}:${remote_port} -> localhost:${local_port}"
    
    kubectl port-forward "${resource_type}/${resource_name}" \
        "${local_port}:${remote_port}" \
        -n "${namespace}" &
    
    local pid=$!
    echo "${pid}"
}

# Export functions
export -f k8s_check_kubectl k8s_check_namespace
export -f k8s_wait_for_deployment k8s_wait_for_pod
export -f k8s_get_pod_status k8s_get_pod_logs
export -f k8s_scale_deployment k8s_rollout_restart k8s_rollout_status
export -f k8s_rollback_deployment k8s_apply_manifest k8s_delete_resource
export -f k8s_get_resource_yaml k8s_port_forward




