#!/bin/bash
# Configuration Validation Script
# Validates all configuration files and environment variables

set -euo pipefail

# Source libraries
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Initialize
init_common

# Configuration
readonly CONFIG_TYPE="${1:-all}"
readonly PROJECT_ROOT=$(get_project_root)
readonly BACKEND_PATH="${PROJECT_ROOT}/agents/backend/onyx/server/features/music_analyzer_ai"

# Validate environment file
validate_env_file() {
    local env_file="${1:-.env}"
    
    log_info "Validating environment file: ${env_file}"
    
    if [ ! -f "${env_file}" ]; then
        log_error "Environment file not found: ${env_file}"
        return 1
    fi
    
    # Check for required variables
    local required_vars=(
        "ENVIRONMENT"
        "API_HOST"
        "API_PORT"
        "SECRET_KEY"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "${env_file}"; then
            missing_vars+=("${var}")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required variables: ${missing_vars[*]}"
        return 1
    fi
    
    # Check for empty values
    if grep -q "=$\|^[^#].*=$" "${env_file}"; then
        log_warn "Found empty values in environment file"
    fi
    
    log_success "Environment file validation passed"
    return 0
}

# Validate Docker Compose file
validate_docker_compose() {
    local compose_file="${1:-docker-compose.yml}"
    
    log_info "Validating Docker Compose file: ${compose_file}"
    
    if [ ! -f "${compose_file}" ]; then
        log_error "Docker Compose file not found: ${compose_file}"
        return 1
    fi
    
    # Validate YAML syntax
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "${compose_file}" config > /dev/null 2>&1 || {
            log_error "Docker Compose file has syntax errors"
            return 1
        }
    else
        log_warn "docker-compose not available, skipping syntax validation"
    fi
    
    log_success "Docker Compose file validation passed"
    return 0
}

# Validate Kubernetes manifests
validate_kubernetes_manifests() {
    local manifests_dir="${1:-deployment/kubernetes}"
    
    log_info "Validating Kubernetes manifests..."
    
    if [ ! -d "${manifests_dir}" ]; then
        log_warn "Kubernetes manifests directory not found: ${manifests_dir}"
        return 0
    fi
    
    if ! k8s_check_kubectl; then
        log_warn "kubectl not available, skipping Kubernetes validation"
        return 0
    fi
    
    # Validate YAML files
    find "${manifests_dir}" -name "*.yaml" -o -name "*.yml" | while read manifest; do
        if ! kubectl apply --dry-run=client -f "${manifest}" &> /dev/null; then
            log_error "Invalid Kubernetes manifest: ${manifest}"
            return 1
        fi
    done
    
    log_success "Kubernetes manifests validation passed"
    return 0
}

# Validate Ansible playbooks
validate_ansible_playbooks() {
    local ansible_dir="${1:-deployment/ansible}"
    
    log_info "Validating Ansible playbooks..."
    
    if [ ! -d "${ansible_dir}" ]; then
        log_warn "Ansible directory not found: ${ansible_dir}"
        return 0
    fi
    
    if ! command -v ansible-playbook &> /dev/null; then
        log_warn "ansible-playbook not available, skipping Ansible validation"
        return 0
    fi
    
    # Validate syntax
    if command -v ansible-lint &> /dev/null; then
        ansible-lint "${ansible_dir}/playbook.yml" || {
            log_warn "Ansible linting found issues"
        }
    fi
    
    # Check syntax
    ansible-playbook --syntax-check "${ansible_dir}/playbook.yml" || {
        log_error "Ansible playbook has syntax errors"
        return 1
    }
    
    log_success "Ansible playbooks validation passed"
    return 0
}

# Validate Terraform configuration
validate_terraform() {
    local terraform_dir="${1:-deployment/terraform}"
    
    log_info "Validating Terraform configuration..."
    
    if [ ! -d "${terraform_dir}" ]; then
        log_warn "Terraform directory not found: ${terraform_dir}"
        return 0
    fi
    
    if ! command -v terraform &> /dev/null; then
        log_warn "terraform not available, skipping Terraform validation"
        return 0
    fi
    
    cd "${terraform_dir}"
    
    # Initialize and validate
    terraform init -backend=false > /dev/null 2>&1
    terraform validate || {
        log_error "Terraform configuration has errors"
        return 1
    }
    
    log_success "Terraform configuration validation passed"
    return 0
}

# Validate all configurations
validate_all() {
    log_info "Validating all configurations..."
    
    local errors=0
    
    validate_env_file ".env" || errors=$((errors + 1))
    validate_env_file ".env.production" || true
    validate_docker_compose "docker-compose.yml" || errors=$((errors + 1))
    validate_docker_compose "deployment/docker-compose.yml" || true
    validate_kubernetes_manifests "deployment/kubernetes" || errors=$((errors + 1))
    validate_ansible_playbooks "deployment/ansible" || errors=$((errors + 1))
    validate_terraform "deployment/terraform" || errors=$((errors + 1))
    
    if [ ${errors} -eq 0 ]; then
        log_success "All configurations validated successfully"
        return 0
    else
        log_error "Configuration validation failed with ${errors} error(s)"
        return 1
    fi
}

# Main function
main() {
    cd "${BACKEND_PATH}/deployment" || {
        log_error "Deployment directory not found"
        exit 1
    }
    
    case "${CONFIG_TYPE}" in
        env)
            validate_env_file "${2:-.env}"
            ;;
        docker)
            validate_docker_compose "${2:-docker-compose.yml}"
            ;;
        kubernetes)
            validate_kubernetes_manifests "${2:-kubernetes}"
            ;;
        ansible)
            validate_ansible_playbooks "${2:-ansible}"
            ;;
        terraform)
            validate_terraform "${2:-terraform}"
            ;;
        all|*)
            validate_all
            ;;
    esac
}

main "$@"




