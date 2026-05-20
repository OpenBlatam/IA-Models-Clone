#!/bin/bash
# Setup script for 3D Prototype AI deployment
# Ensures all scripts are executable and environment is properly configured

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Function to make scripts executable
make_scripts_executable() {
    log_info "Making scripts executable..."
    
    local scripts_dir="${SCRIPT_DIR}"
    local scripts=(
        "deploy.sh"
        "validate.sh"
        "backup.sh"
        "rollback.sh"
        "health_check.sh"
        "monitor.sh"
        "update_app.sh"
        "view_logs.sh"
        "launch_ec2.sh"
        "cleanup.sh"
        "test_scripts.sh"
        "docker_utils.sh"
    )
    
    local lib_scripts=(
        "lib/common.sh"
    )
    
    # Make main scripts executable
    for script in "${scripts[@]}"; do
        local script_path="${scripts_dir}/${script}"
        if [ -f "${script_path}" ]; then
            chmod +x "${script_path}" 2>/dev/null || log_warn "Could not make ${script} executable (may need sudo)"
            log_debug "Made ${script} executable"
        fi
    done
    
    # Make library scripts executable
    for script in "${lib_scripts[@]}"; do
        local script_path="${scripts_dir}/${script}"
        if [ -f "${script_path}" ]; then
            chmod +x "${script_path}" 2>/dev/null || log_warn "Could not make ${script} executable"
            log_debug "Made ${script} executable"
        fi
    done
    
    log_info "Scripts made executable ✓"
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    local dirs=(
        "${CLOUD_DIR}/backups"
        "${CLOUD_DIR}/logs"
        "${CLOUD_DIR}/tmp"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "${dir}" ]; then
            mkdir -p "${dir}"
            log_debug "Created directory: ${dir}"
        fi
    done
    
    log_info "Directories created ✓"
}

# Function to create .env file if it doesn't exist
setup_env_file() {
    log_info "Setting up environment file..."
    
    local env_file="${CLOUD_DIR}/.env"
    local env_example="${CLOUD_DIR}/.env.example"
    
    if [ ! -f "${env_file}" ]; then
        if [ -f "${env_example}" ]; then
            cp "${env_example}" "${env_file}"
            log_info "Created .env file from .env.example"
            log_warn "Please edit ${env_file} with your configuration"
        else
            log_warn ".env.example not found, creating basic .env file"
            cat > "${env_file}" << 'EOF'
# AWS Configuration
AWS_REGION=us-east-1
AWS_INSTANCE_TYPE=t3.large
AWS_KEY_NAME=your-key-pair-name
AWS_KEY_PATH=~/.ssh/your-key-pair-name.pem

# Application Configuration
APP_PORT=8030
APP_HOST=0.0.0.0

# Deployment Configuration
DEPLOYMENT_METHOD=terraform
SKIP_APP_DEPLOY=false
USE_DOCKER=true
EOF
            log_info "Created basic .env file"
            log_warn "Please edit ${env_file} with your configuration"
        fi
    else
        log_info ".env file already exists"
    fi
}

# Function to setup Terraform configuration
setup_terraform() {
    log_info "Setting up Terraform configuration..."
    
    local terraform_dir="${CLOUD_DIR}/terraform"
    local tfvars_file="${terraform_dir}/terraform.tfvars"
    local tfvars_example="${terraform_dir}/terraform.tfvars.example"
    
    if [ ! -f "${tfvars_file}" ]; then
        if [ -f "${tfvars_example}" ]; then
            cp "${tfvars_example}" "${tfvars_file}"
            log_info "Created terraform.tfvars from example"
            log_warn "Please edit ${tfvars_file} with your configuration"
        else
            log_warn "terraform.tfvars.example not found"
        fi
    else
        log_info "terraform.tfvars already exists"
    fi
}

# Function to setup Ansible inventory
setup_ansible_inventory() {
    log_info "Setting up Ansible inventory..."
    
    local ansible_dir="${CLOUD_DIR}/ansible"
    local inventory_file="${ansible_dir}/inventory/ec2.ini"
    local inventory_example="${ansible_dir}/inventory/ec2.ini.example"
    
    if [ ! -f "${inventory_file}" ]; then
        if [ -f "${inventory_example}" ]; then
            cp "${inventory_example}" "${inventory_file}"
            log_info "Created Ansible inventory from example"
            log_warn "Please edit ${inventory_file} with your instance details"
        else
            log_warn "Ansible inventory example not found"
        fi
    else
        log_info "Ansible inventory already exists"
    fi
}

# Function to check and install dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    local optional_deps=()
    
    # Required dependencies
    if ! command -v aws &> /dev/null; then
        missing_deps+=("aws-cli")
    fi
    
    # Optional dependencies
    if ! command -v terraform &> /dev/null; then
        optional_deps+=("terraform")
    fi
    
    if ! command -v ansible-playbook &> /dev/null; then
        optional_deps+=("ansible")
    fi
    
    if ! command -v docker &> /dev/null; then
        optional_deps+=("docker")
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        optional_deps+=("docker-compose")
    fi
    
    if ! command -v shellcheck &> /dev/null; then
        optional_deps+=("shellcheck")
    fi
    
    # Report missing required dependencies
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            case "${dep}" in
                aws-cli)
                    log_info "  AWS CLI: https://aws.amazon.com/cli/"
                    ;;
            esac
        done
        return 1
    fi
    
    # Report optional dependencies
    if [ ${#optional_deps[@]} -ne 0 ]; then
        log_warn "Optional dependencies not found: ${optional_deps[*]}"
        log_info "These are optional but recommended for full functionality"
    else
        log_info "All dependencies found ✓"
    fi
    
    return 0
}

# Function to verify script syntax
verify_scripts() {
    log_info "Verifying script syntax..."
    
    local scripts_dir="${SCRIPT_DIR}"
    local errors=0
    
    # Check if bash is available
    if ! command -v bash &> /dev/null; then
        log_error "bash not found, cannot verify scripts"
        return 1
    fi
    
    # Find all shell scripts
    while IFS= read -r -d '' script; do
        if ! bash -n "${script}" 2>/dev/null; then
            log_error "Syntax error in: ${script}"
            ((errors++))
        fi
    done < <(find "${scripts_dir}" -type f -name "*.sh" -print0)
    
    if [ ${errors} -eq 0 ]; then
        log_info "All scripts have valid syntax ✓"
        return 0
    else
        log_error "Found ${errors} script(s) with syntax errors"
        return 1
    fi
}

# Function to create user data script executable
setup_user_data() {
    log_info "Setting up user data script..."
    
    local user_data_file="${CLOUD_DIR}/user_data/init.sh"
    
    if [ -f "${user_data_file}" ]; then
        chmod +x "${user_data_file}" 2>/dev/null || log_warn "Could not make user_data/init.sh executable"
        log_info "User data script is ready ✓"
    else
        log_warn "User data script not found: ${user_data_file}"
    fi
}

# Function to display setup summary
display_summary() {
    cat << EOF

${GREEN}==========================================${NC}
${GREEN}Setup Summary${NC}
${GREEN}==========================================${NC}

Setup completed! Next steps:

1. Configure environment:
   ${YELLOW}Edit ${CLOUD_DIR}/.env${NC}

2. Configure Terraform (if using):
   ${YELLOW}Edit ${CLOUD_DIR}/terraform/terraform.tfvars${NC}

3. Configure Ansible (if using):
   ${YELLOW}Edit ${CLOUD_DIR}/ansible/inventory/ec2.ini${NC}

4. Validate configuration:
   ${YELLOW}./scripts/validate.sh${NC}

5. Deploy:
   ${YELLOW}./scripts/deploy.sh${NC}

Or use Makefile:
   ${YELLOW}make setup${NC}
   ${YELLOW}make validate${NC}
   ${YELLOW}make deploy${NC}

${GREEN}==========================================${NC}

EOF
}

# Main function
main() {
    log_info "Starting setup for 3D Prototype AI deployment..."
    echo ""
    
    # Check dependencies
    if ! check_dependencies; then
        log_error "Setup cannot continue without required dependencies"
        exit 1
    fi
    echo ""
    
    # Make scripts executable
    make_scripts_executable
    echo ""
    
    # Create directories
    create_directories
    echo ""
    
    # Setup configuration files
    setup_env_file
    echo ""
    
    setup_terraform
    echo ""
    
    setup_ansible_inventory
    echo ""
    
    # Setup user data
    setup_user_data
    echo ""
    
    # Verify scripts
    verify_scripts
    echo ""
    
    # Display summary
    display_summary
    
    log_info "Setup completed successfully! ✓"
}

main "$@"

