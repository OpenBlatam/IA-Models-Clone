#!/bin/bash
# Environment check script
# Verifies that the deployment environment is properly configured

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Check operating system
check_os() {
    log_info "Checking operating system..."
    
    local os_type
    os_type=$(uname -s)
    
    case "${os_type}" in
        Linux)
            log_info "OS: Linux ✓"
            ;;
        Darwin)
            log_info "OS: macOS ✓"
            log_warn "Some features may require Linux"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            log_info "OS: Windows (Git Bash/WSL) ✓"
            log_warn "Some features may require Linux or WSL"
            ;;
        *)
            log_warn "OS: ${os_type} (may have limited support)"
            ;;
    esac
}

# Check shell
check_shell() {
    log_info "Checking shell..."
    
    local shell_name
    shell_name=$(basename "${SHELL}")
    
    if [ "${shell_name}" = "bash" ]; then
        local bash_version
        bash_version=$(bash --version | head -1)
        log_info "Shell: ${bash_version} ✓"
    else
        log_warn "Shell: ${shell_name} (bash recommended)"
    fi
}

# Check required commands
check_commands() {
    log_info "Checking required commands..."
    
    local required=("bash" "curl" "git")
    local optional=("aws" "terraform" "ansible-playbook" "docker" "docker-compose")
    
    local missing_required=()
    local missing_optional=()
    
    # Check required commands
    for cmd in "${required[@]}"; do
        if ! command -v "${cmd}" &> /dev/null; then
            missing_required+=("${cmd}")
        else
            log_info "✓ ${cmd} found"
        fi
    done
    
    # Check optional commands
    for cmd in "${optional[@]}"; do
        if ! command -v "${cmd}" &> /dev/null; then
            # Check for docker compose (newer syntax)
            if [ "${cmd}" = "docker-compose" ]; then
                if docker compose version &> /dev/null; then
                    log_info "✓ docker compose found (new syntax)"
                    continue
                fi
            fi
            missing_optional+=("${cmd}")
        else
            log_info "✓ ${cmd} found"
        fi
    done
    
    # Report results
    if [ ${#missing_required[@]} -ne 0 ]; then
        log_error "Missing required commands: ${missing_required[*]}"
        return 1
    fi
    
    if [ ${#missing_optional[@]} -ne 0 ]; then
        log_warn "Missing optional commands: ${missing_optional[*]}"
        log_info "These are optional but recommended"
    fi
    
    return 0
}

# Check file permissions
check_permissions() {
    log_info "Checking script permissions..."
    
    local scripts_dir="${SCRIPT_DIR}"
    local issues=0
    
    while IFS= read -r -d '' script; do
        if [ ! -x "${script}" ]; then
            log_warn "Script not executable: ${script}"
            ((issues++))
        fi
    done < <(find "${scripts_dir}" -type f -name "*.sh" -print0)
    
    if [ ${issues} -eq 0 ]; then
        log_info "All scripts are executable ✓"
        return 0
    else
        log_warn "Found ${issues} script(s) that are not executable"
        log_info "Run: ./scripts/fix_permissions.sh"
        return 1
    fi
}

# Check configuration files
check_config() {
    log_info "Checking configuration files..."
    
    local config_ok=true
    
    # Check .env file
    if [ ! -f "${CLOUD_DIR}/.env" ]; then
        log_warn ".env file not found"
        log_info "Run: ./scripts/setup.sh"
        config_ok=false
    else
        log_info "✓ .env file exists"
    fi
    
    # Check terraform.tfvars
    if [ ! -f "${CLOUD_DIR}/terraform/terraform.tfvars" ]; then
        log_warn "terraform.tfvars not found (optional if not using Terraform)"
    else
        log_info "✓ terraform.tfvars exists"
    fi
    
    # Check Ansible inventory
    if [ ! -f "${CLOUD_DIR}/ansible/inventory/ec2.ini" ]; then
        log_warn "Ansible inventory not found (optional if not using Ansible)"
    else
        log_info "✓ Ansible inventory exists"
    fi
    
    if [ "${config_ok}" = "true" ]; then
        return 0
    else
        return 1
    fi
}

# Check AWS credentials
check_aws() {
    log_info "Checking AWS configuration..."
    
    if ! command -v aws &> /dev/null; then
        log_warn "AWS CLI not installed (optional)"
        return 0
    fi
    
    if aws sts get-caller-identity &> /dev/null; then
        local aws_account
        aws_account=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")
        log_info "✓ AWS credentials configured (Account: ${aws_account})"
        return 0
    else
        log_warn "AWS credentials not configured"
        log_info "Run: aws configure"
        return 1
    fi
}

# Main function
main() {
    log_info "Checking deployment environment..."
    echo ""
    
    local errors=0
    local warnings=0
    
    # Run checks
    check_os
    echo ""
    
    check_shell
    echo ""
    
    if ! check_commands; then
        ((errors++))
    fi
    echo ""
    
    if ! check_permissions; then
        ((warnings++))
    fi
    echo ""
    
    if ! check_config; then
        ((warnings++))
    fi
    echo ""
    
    if ! check_aws; then
        ((warnings++))
    fi
    echo ""
    
    # Summary
    log_info "=========================================="
    if [ ${errors} -eq 0 ] && [ ${warnings} -eq 0 ]; then
        log_info "Environment check passed! ✓"
        exit 0
    elif [ ${errors} -eq 0 ]; then
        log_warn "Environment check passed with ${warnings} warning(s)"
        exit 0
    else
        log_error "Environment check failed with ${errors} error(s) and ${warnings} warning(s)"
        exit 1
    fi
}

main "$@"

