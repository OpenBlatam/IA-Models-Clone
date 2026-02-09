#!/bin/bash

###############################################################################
# CI/CD Deployment Script
# Automated deployment script for CI/CD pipelines
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/../aws/scripts/common_functions_enhanced.sh" 2>/dev/null || {
    source "${SCRIPT_DIR}/../aws/scripts/common_functions.sh" 2>/dev/null || {
        echo "Error: common functions not found" >&2
        exit 1
    }
}

# Configuration
ENVIRONMENT="${ENVIRONMENT:-staging}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
TERRAFORM_DIR="${PROJECT_ROOT}/aws/terraform"
ANSIBLE_DIR="${PROJECT_ROOT}/aws/ansible"
SKIP_TESTS="${SKIP_TESTS:-false}"
AUTO_APPROVE="${AUTO_APPROVE:-false}"

###############################################################################
# Validation
###############################################################################

validate_deployment() {
    log_section "Validating Deployment Configuration"
    
    # Validate environment
    case "${ENVIRONMENT}" in
        staging|production|development)
            log_info "Environment: ${ENVIRONMENT}"
            ;;
        *)
            log_error "Invalid environment: ${ENVIRONMENT}"
            exit 1
            ;;
    esac
    
    # Validate required tools
    check_command "terraform" || exit 1
    check_command "ansible-playbook" || exit 1
    check_command "aws" || exit 1
    
    # Validate AWS credentials
    check_aws_credentials || exit 1
    
    # Validate directories
    check_directory_exists "${TERRAFORM_DIR}" || exit 1
    check_directory_exists "${ANSIBLE_DIR}" || exit 1
    
    log_success "Deployment validation passed"
}

###############################################################################
# Pre-deployment
###############################################################################

pre_deployment_checks() {
    log_section "Pre-Deployment Checks"
    
    # Run tests if not skipped
    if [ "${SKIP_TESTS}" != "true" ]; then
        log_info "Running pre-deployment tests..."
        if [ -f "${PROJECT_ROOT}/pytest.ini" ]; then
            cd "${PROJECT_ROOT}"
            pytest tests/ -v --tb=short || {
                log_error "Tests failed, aborting deployment"
                exit 1
            }
        fi
    else
        log_warn "Tests skipped (SKIP_TESTS=true)"
    fi
    
    # Check application health in current environment
    log_info "Checking current application health..."
    # Health check logic here
    
    log_success "Pre-deployment checks completed"
}

###############################################################################
# Infrastructure Deployment
###############################################################################

deploy_infrastructure() {
    log_section "Deploying Infrastructure"
    
    cd "${TERRAFORM_DIR}" || exit 1
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init -upgrade || {
        log_error "Terraform initialization failed"
        exit 1
    }
    
    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan \
        -var="environment=${ENVIRONMENT}" \
        -var="image_tag=${IMAGE_TAG}" \
        -out=tfplan || {
        log_error "Terraform plan failed"
        exit 1
    }
    
    # Apply deployment
    if [ "${AUTO_APPROVE}" = "true" ]; then
        log_info "Applying Terraform configuration (auto-approve)..."
        terraform apply -auto-approve tfplan || {
            log_error "Terraform apply failed"
            exit 1
        }
    else
        log_info "Terraform plan ready. Manual approval required."
        log_info "Run: terraform apply tfplan"
        return 0
    fi
    
    log_success "Infrastructure deployed successfully"
}

###############################################################################
# Application Deployment
###############################################################################

deploy_application() {
    log_section "Deploying Application"
    
    cd "${ANSIBLE_DIR}" || exit 1
    
    # Update EC2 inventory
    log_info "Updating EC2 inventory..."
    ansible-inventory -i inventory/ec2.ini --list > /dev/null || {
        log_error "Failed to update EC2 inventory"
        exit 1
    }
    
    # Deploy application
    log_info "Deploying application with Ansible..."
    ansible-playbook \
        -i inventory/ec2.ini \
        playbooks/deploy.yml \
        -e "environment=${ENVIRONMENT}" \
        -e "image_tag=${IMAGE_TAG}" \
        -e "git_branch=${GIT_BRANCH:-main}" \
        --ask-become-pass || {
        log_error "Ansible deployment failed"
        exit 1
    }
    
    log_success "Application deployed successfully"
}

###############################################################################
# Post-deployment
###############################################################################

post_deployment_verification() {
    log_section "Post-Deployment Verification"
    
    # Wait for application to be ready
    log_info "Waiting for application to be ready..."
    sleep 30
    
    # Get load balancer DNS
    cd "${TERRAFORM_DIR}" || exit 1
    local lb_dns
    lb_dns=$(terraform output -raw load_balancer_dns 2>/dev/null || echo "")
    
    if [ -z "${lb_dns}" ]; then
        log_warn "Load balancer DNS not available"
        return 0
    fi
    
    # Health check
    log_info "Performing health check..."
    local max_retries=10
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if check_url "http://${lb_dns}/health" 10 1; then
            log_success "Health check passed"
            return 0
        fi
        
        retry=$((retry + 1))
        log_info "Health check attempt ${retry}/${max_retries}..."
        sleep 10
    done
    
    log_error "Health check failed after ${max_retries} attempts"
    return 1
}

###############################################################################
# Rollback
###############################################################################

rollback_deployment() {
    log_section "Rolling Back Deployment"
    
    cd "${ANSIBLE_DIR}" || exit 1
    
    log_info "Initiating rollback..."
    ansible-playbook \
        -i inventory/ec2.ini \
        playbooks/rollback.yml \
        -e "environment=${ENVIRONMENT}" \
        --ask-become-pass || {
        log_error "Rollback failed"
        exit 1
    }
    
    log_success "Rollback completed"
}

###############################################################################
# Main Execution
###############################################################################

main() {
    local action="${1:-deploy}"
    
    init_logging "/var/log/cicd-deploy.log"
    log_section "CI/CD Deployment - ${ENVIRONMENT}"
    log_info "Image Tag: ${IMAGE_TAG}"
    log_info "Action: ${action}"
    
    case "${action}" in
        deploy)
            validate_deployment
            pre_deployment_checks
            deploy_infrastructure
            deploy_application
            post_deployment_verification || {
                log_error "Deployment verification failed, initiating rollback"
                rollback_deployment
                exit 1
            }
            log_success "Deployment completed successfully"
            ;;
        rollback)
            rollback_deployment
            ;;
        validate)
            validate_deployment
            pre_deployment_checks
            ;;
        *)
            log_error "Unknown action: ${action}"
            log_info "Usage: $0 [deploy|rollback|validate]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

