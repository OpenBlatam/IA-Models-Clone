#!/bin/bash
# Terraform validation script
# Validates Terraform configuration before deployment

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check if Terraform is installed
check_terraform() {
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed"
        log_info "Install from: https://www.terraform.io/downloads"
        return 1
    fi
    
    local version
    version=$(terraform version -json | jq -r '.terraform_version' 2>/dev/null || terraform version | head -1)
    log_info "Terraform version: ${version}"
    return 0
}

# Initialize Terraform
init_terraform() {
    log_info "Initializing Terraform..."
    
    if terraform init -backend=false > /dev/null 2>&1; then
        log_info "Terraform initialized ✓"
        return 0
    else
        log_error "Terraform initialization failed"
        terraform init -backend=false
        return 1
    fi
}

# Validate Terraform configuration
validate_config() {
    log_info "Validating Terraform configuration..."
    
    if terraform validate > /dev/null 2>&1; then
        log_info "Terraform configuration is valid ✓"
        return 0
    else
        log_error "Terraform validation failed"
        terraform validate
        return 1
    fi
}

# Format check
format_check() {
    log_info "Checking Terraform formatting..."
    
    if terraform fmt -check -recursive > /dev/null 2>&1; then
        log_info "Terraform files are properly formatted ✓"
        return 0
    else
        log_warn "Terraform files are not properly formatted"
        log_info "Run 'terraform fmt -recursive' to fix"
        terraform fmt -check -recursive
        return 1
    fi
}

# Security check (if tfsec is available)
security_check() {
    if ! command -v tfsec &> /dev/null; then
        log_warn "tfsec not installed, skipping security check"
        log_info "Install from: https://github.com/aquasecurity/tfsec"
        return 0
    fi
    
    log_info "Running security checks with tfsec..."
    
    if tfsec . > /dev/null 2>&1; then
        log_info "Security checks passed ✓"
        return 0
    else
        log_warn "Security issues found"
        tfsec .
        return 1
    fi
}

# Main function
main() {
    log_info "Starting Terraform validation..."
    echo ""
    
    local errors=0
    local warnings=0
    
    # Check Terraform
    if ! check_terraform; then
        exit 1
    fi
    echo ""
    
    # Initialize
    if ! init_terraform; then
        ((errors++))
    fi
    echo ""
    
    # Validate
    if ! validate_config; then
        ((errors++))
    fi
    echo ""
    
    # Format check
    if ! format_check; then
        ((warnings++))
    fi
    echo ""
    
    # Security check
    if ! security_check; then
        ((warnings++))
    fi
    echo ""
    
    # Summary
    log_info "=========================================="
    if [ ${errors} -eq 0 ] && [ ${warnings} -eq 0 ]; then
        log_info "All validations passed! ✓"
        exit 0
    elif [ ${errors} -eq 0 ]; then
        log_warn "Validation passed with ${warnings} warning(s)"
        exit 0
    else
        log_error "Validation failed with ${errors} error(s) and ${warnings} warning(s)"
        exit 1
    fi
}

main "$@"

