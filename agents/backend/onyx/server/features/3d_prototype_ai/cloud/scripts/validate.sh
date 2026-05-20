#!/bin/bash
# Validation script for deployment configuration
# Validates all configuration files, credentials, and prerequisites

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Validation results
VALIDATION_ERRORS=0

# Validate Terraform configuration
validate_terraform() {
    log_info "Validating Terraform configuration..."
    
    local terraform_dir="${CLOUD_DIR}/terraform"
    
    if [ ! -d "${terraform_dir}" ]; then
        log_warn "Terraform directory not found, skipping..."
        return 0
    fi
    
    cd "${terraform_dir}"
    
    # Check for required files
    local required_files=("main.tf" "variables.tf" "outputs.tf")
    for file in "${required_files[@]}"; do
        if [ ! -f "${file}" ]; then
            log_error "Missing Terraform file: ${file}"
            ((VALIDATION_ERRORS++))
        fi
    done
    
    # Validate Terraform syntax if terraform is available
    if command -v terraform &> /dev/null; then
        log_info "Running terraform fmt -check..."
        if ! terraform fmt -check > /dev/null 2>&1; then
            log_warn "Terraform files are not properly formatted"
            log_info "Run 'terraform fmt' to fix formatting"
        fi
        
        log_info "Running terraform validate..."
        if terraform init -backend=false > /dev/null 2>&1; then
            if ! terraform validate > /dev/null 2>&1; then
                log_error "Terraform validation failed"
                terraform validate
                ((VALIDATION_ERRORS++))
            else
                log_info "Terraform validation passed ✓"
            fi
        else
            log_warn "Could not initialize Terraform for validation"
        fi
    else
        log_warn "Terraform not installed, skipping validation"
    fi
    
    cd - > /dev/null
}

# Validate Ansible configuration
validate_ansible() {
    log_info "Validating Ansible configuration..."
    
    local ansible_dir="${CLOUD_DIR}/ansible"
    
    if [ ! -d "${ansible_dir}" ]; then
        log_warn "Ansible directory not found, skipping..."
        return 0
    fi
    
    # Check for playbooks
    if [ ! -f "${ansible_dir}/playbooks/deploy.yml" ]; then
        log_warn "Ansible playbook not found"
    else
        log_info "Ansible playbook found ✓"
    fi
    
    # Validate Ansible syntax if ansible-playbook is available
    if command -v ansible-playbook &> /dev/null; then
        if [ -f "${ansible_dir}/playbooks/deploy.yml" ]; then
            log_info "Validating Ansible playbook syntax..."
            if ansible-playbook --syntax-check "${ansible_dir}/playbooks/deploy.yml" > /dev/null 2>&1; then
                log_info "Ansible playbook syntax is valid ✓"
            else
                log_error "Ansible playbook syntax validation failed"
                ansible-playbook --syntax-check "${ansible_dir}/playbooks/deploy.yml"
                ((VALIDATION_ERRORS++))
            fi
        fi
        
        # Check ansible-lint if available
        if command -v ansible-lint &> /dev/null; then
            log_info "Running ansible-lint..."
            if ansible-lint "${ansible_dir}/playbooks/deploy.yml" > /dev/null 2>&1; then
                log_info "Ansible linting passed ✓"
            else
                log_warn "Ansible linting found issues"
                ansible-lint "${ansible_dir}/playbooks/deploy.yml" || true
            fi
        fi
    else
        log_warn "Ansible not installed, skipping validation"
    fi
}

# Validate CloudFormation template
validate_cloudformation() {
    log_info "Validating CloudFormation template..."
    
    local template_file="${CLOUD_DIR}/cloudformation/stack.yaml"
    
    if [ ! -f "${template_file}" ]; then
        log_warn "CloudFormation template not found, skipping..."
        return 0
    fi
    
    if command -v aws &> /dev/null && check_aws_credentials; then
        log_info "Validating CloudFormation template with AWS..."
        if aws cloudformation validate-template \
            --template-body "file://${template_file}" \
            --region "${AWS_REGION:-us-east-1}" > /dev/null 2>&1; then
            log_info "CloudFormation template is valid ✓"
        else
            log_error "CloudFormation template validation failed"
            aws cloudformation validate-template \
                --template-body "file://${template_file}" \
                --region "${AWS_REGION:-us-east-1}"
            ((VALIDATION_ERRORS++))
        fi
    else
        log_warn "AWS CLI not configured, skipping CloudFormation validation"
    fi
}

# Validate environment variables
validate_environment() {
    log_info "Validating environment configuration..."
    
    local required_vars=("AWS_REGION" "AWS_KEY_NAME")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("${var}")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        ((VALIDATION_ERRORS++))
    else
        log_info "Required environment variables are set ✓"
    fi
    
    # Validate AWS region format
    if [ -n "${AWS_REGION:-}" ]; then
        if [[ ! "${AWS_REGION}" =~ ^[a-z0-9-]+$ ]]; then
            log_error "Invalid AWS region format: ${AWS_REGION}"
            ((VALIDATION_ERRORS++))
        fi
    fi
    
    # Validate instance type if set
    if [ -n "${AWS_INSTANCE_TYPE:-}" ]; then
        if [[ ! "${AWS_INSTANCE_TYPE}" =~ ^[a-z0-9.]+$ ]]; then
            log_warn "Instance type format may be invalid: ${AWS_INSTANCE_TYPE}"
        fi
    fi
    
    # Validate key path if provided
    if [ -n "${AWS_KEY_PATH:-}" ]; then
        if [ ! -f "${AWS_KEY_PATH}" ]; then
            log_error "SSH key file not found: ${AWS_KEY_PATH}"
            ((VALIDATION_ERRORS++))
        else
            log_info "SSH key file found ✓"
        fi
    fi
}

# Validate scripts
validate_scripts() {
    log_info "Validating deployment scripts..."
    
    local scripts_dir="${SCRIPT_DIR}"
    local scripts=("deploy.sh" "launch_ec2.sh" "health_check.sh" "update_app.sh" "view_logs.sh")
    
    for script in "${scripts[@]}"; do
        local script_path="${scripts_dir}/${script}"
        if [ ! -f "${script_path}" ]; then
            log_warn "Script not found: ${script}"
            continue
        fi
        
        # Check if script is executable
        if [ ! -x "${script_path}" ]; then
            log_warn "Script is not executable: ${script}"
        fi
        
        # Check for shellcheck if available
        if command -v shellcheck &> /dev/null; then
            log_debug "Running shellcheck on ${script}..."
            if shellcheck "${script_path}" > /dev/null 2>&1; then
                log_debug "shellcheck passed for ${script} ✓"
            else
                log_warn "shellcheck found issues in ${script}"
                shellcheck "${script_path}" || true
            fi
        fi
    done
    
    log_info "Script validation completed"
}

# Validate user data script
validate_user_data() {
    log_info "Validating user data script..."
    
    local user_data_file="${CLOUD_DIR}/user_data/init.sh"
    
    if [ ! -f "${user_data_file}" ]; then
        log_error "User data script not found: ${user_data_file}"
        ((VALIDATION_ERRORS++))
        return 1
    fi
    
    # Check if script is readable
    if [ ! -r "${user_data_file}" ]; then
        log_error "User data script is not readable"
        ((VALIDATION_ERRORS++))
        return 1
    fi
    
    # Basic syntax check
    if bash -n "${user_data_file}" 2>/dev/null; then
        log_info "User data script syntax is valid ✓"
    else
        log_error "User data script has syntax errors"
        bash -n "${user_data_file}"
        ((VALIDATION_ERRORS++))
    fi
}

# Main validation function
main() {
    log_info "Starting deployment configuration validation..."
    echo ""
    
    validate_environment
    echo ""
    
    validate_terraform
    echo ""
    
    validate_ansible
    echo ""
    
    validate_cloudformation
    echo ""
    
    validate_scripts
    echo ""
    
    validate_user_data
    echo ""
    
    # Summary
    log_info "=========================================="
    if [ ${VALIDATION_ERRORS} -eq 0 ]; then
        log_info "All validations passed! ✓"
        exit 0
    else
        log_error "Validation completed with ${VALIDATION_ERRORS} error(s)"
        exit 1
    fi
}

main "$@"

