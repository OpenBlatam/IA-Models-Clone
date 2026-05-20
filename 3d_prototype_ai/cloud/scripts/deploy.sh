#!/bin/bash
# Main deployment script for 3D Prototype AI on AWS EC2
# This script orchestrates the entire deployment process
# Following DevOps best practices: error handling, validation, logging, modularity

set -o errexit
set -o nounset
set -o pipefail

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Initialize
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLOUD_DIR="${SCRIPT_DIR}/.."
setup_trap
create_temp_dir

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly AWS_REGION="${AWS_REGION:-us-east-1}"
readonly INSTANCE_TYPE="${AWS_INSTANCE_TYPE:-t3.large}"
readonly KEY_NAME="${AWS_KEY_NAME:-}"
readonly DEPLOYMENT_METHOD="${DEPLOYMENT_METHOD:-terraform}"
readonly SKIP_APP_DEPLOY="${SKIP_APP_DEPLOY:-false}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"

# Global variables
INSTANCE_IP=""
INSTANCE_ID=""

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy 3D Prototype AI application to AWS EC2.

OPTIONS:
    -m, --method METHOD      Deployment method (terraform|cloudformation|ansible)
    -r, --region REGION       AWS region (default: us-east-1)
    -t, --type TYPE          Instance type (default: t3.large)
    -k, --key-name NAME      AWS key pair name
    -p, --key-path PATH      Path to SSH private key
    -s, --skip-app           Skip application deployment
    -d, --debug              Enable debug mode
    -h, --help               Show this help message

ENVIRONMENT VARIABLES:
    AWS_REGION               AWS region
    AWS_INSTANCE_TYPE        EC2 instance type
    AWS_KEY_NAME             AWS key pair name
    AWS_KEY_PATH             Path to SSH private key
    DEPLOYMENT_METHOD        Deployment method
    SKIP_APP_DEPLOY          Skip app deployment (true/false)

EXAMPLES:
    $0 --method terraform --region us-east-1
    $0 --method cloudformation --key-name my-key
    $0 --skip-app  # Only deploy infrastructure

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--method)
                DEPLOYMENT_METHOD="$2"
                shift 2
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -t|--type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            -k|--key-name)
                KEY_NAME="$2"
                shift 2
                ;;
            -p|--key-path)
                AWS_KEY_PATH="$2"
                shift 2
                ;;
            -s|--skip-app)
                SKIP_APP_DEPLOY="true"
                shift
                ;;
            -d|--debug)
                export DEBUG="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("aws")
    case "${DEPLOYMENT_METHOD}" in
        terraform)
            required_commands+=("terraform")
            ;;
        ansible)
            required_commands+=("ansible-playbook")
            ;;
    esac
    
    local missing_tools=()
    for cmd in "${required_commands[@]}"; do
        if ! check_command "${cmd}"; then
            missing_tools+=("${cmd}")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        error_exit 1 "Missing required tools: ${missing_tools[*]}"
    fi
    
    # Check AWS credentials
    check_aws_credentials || error_exit 1 "AWS credentials validation failed"
    
    # Validate key path if provided
    if [ -n "${AWS_KEY_PATH}" ]; then
        validate_file "${AWS_KEY_PATH}" "SSH private key" || error_exit 1
        # Check key permissions
        local key_perms
        key_perms=$(stat -c "%a" "${AWS_KEY_PATH}" 2>/dev/null || stat -f "%OLp" "${AWS_KEY_PATH}" 2>/dev/null || echo "000")
        if [ "${key_perms}" != "600" ] && [ "${key_perms}" != "400" ]; then
            log_warn "SSH key permissions should be 600 or 400, current: ${key_perms}"
        fi
    fi
    
    # Validate required environment variables
    if [ -z "${KEY_NAME}" ]; then
        error_exit 1 "AWS_KEY_NAME is required"
    fi
    
    log_info "Prerequisites check passed ✓"
}

# Deploy with Terraform
deploy_with_terraform() {
    log_info "Deploying with Terraform..."
    
    local terraform_dir="${CLOUD_DIR}/terraform"
    validate_directory "${terraform_dir}" "Terraform directory"
    
    # Check for terraform.tfvars
    if [ ! -f "${terraform_dir}/terraform.tfvars" ]; then
        log_warn "terraform.tfvars not found. Creating from example..."
        if [ -f "${terraform_dir}/terraform.tfvars.example" ]; then
            cp "${terraform_dir}/terraform.tfvars.example" "${terraform_dir}/terraform.tfvars"
            log_warn "Please edit ${terraform_dir}/terraform.tfvars with your values"
            error_exit 1 "Configuration file created, please update it"
        else
            error_exit 1 "terraform.tfvars.example not found"
        fi
    fi
    
    cd "${terraform_dir}"
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init -upgrade || error_exit 1 "Terraform initialization failed"
    
    # Validate Terraform configuration
    log_info "Validating Terraform configuration..."
    terraform validate || error_exit 1 "Terraform validation failed"
    
    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan -out="${TMP_DIR}/tfplan" || error_exit 1 "Terraform plan failed"
    
    # Apply configuration
    log_info "Applying Terraform configuration..."
    terraform apply "${TMP_DIR}/tfplan" || error_exit 1 "Terraform apply failed"
    
    # Get outputs
    log_info "Getting deployment outputs..."
    terraform output
    
    # Extract instance details
    INSTANCE_IP=$(terraform output -raw instance_public_ip 2>/dev/null || echo "")
    INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null || echo "")
    
    if [ -z "${INSTANCE_IP}" ]; then
        error_exit 1 "Failed to get instance IP from Terraform output"
    fi
    
    validate_ip "${INSTANCE_IP}" || error_exit 1 "Invalid instance IP"
    
    log_info "Instance deployed: ${INSTANCE_ID}"
    log_info "Public IP: ${INSTANCE_IP}"
    
    cd - > /dev/null
}

# Deploy with CloudFormation
deploy_with_cloudformation() {
    log_info "Deploying with CloudFormation..."
    
    local stack_name="${PROJECT_NAME:-3d-prototype-ai}-stack-$(date +%s)"
    local template_file="${CLOUD_DIR}/cloudformation/stack.yaml"
    
    validate_file "${template_file}" "CloudFormation template"
    
    # Validate template
    log_info "Validating CloudFormation template..."
    aws cloudformation validate-template \
        --template-body "file://${template_file}" \
        --region "${AWS_REGION}" > /dev/null || error_exit 1 "CloudFormation template validation failed"
    
    # Deploy stack
    log_info "Deploying CloudFormation stack: ${stack_name}"
    aws cloudformation deploy \
        --template-file "${template_file}" \
        --stack-name "${stack_name}" \
        --capabilities CAPABILITY_IAM \
        --region "${AWS_REGION}" \
        --parameter-overrides \
            InstanceType="${INSTANCE_TYPE}" \
            KeyName="${KEY_NAME}" \
        || error_exit 1 "CloudFormation deployment failed"
    
    # Get stack outputs
    log_info "Getting stack outputs..."
    local outputs
    outputs=$(aws cloudformation describe-stacks \
        --stack-name "${stack_name}" \
        --region "${AWS_REGION}" \
        --query 'Stacks[0].Outputs' \
        --output json)
    
    INSTANCE_IP=$(echo "${outputs}" | jq -r '.[] | select(.OutputKey=="PublicIP") | .OutputValue')
    INSTANCE_ID=$(echo "${outputs}" | jq -r '.[] | select(.OutputKey=="InstanceId") | .OutputValue')
    
    if [ -z "${INSTANCE_IP}" ] || [ "${INSTANCE_IP}" = "null" ]; then
        error_exit 1 "Failed to get instance IP from CloudFormation output"
    fi
    
    validate_ip "${INSTANCE_IP}" || error_exit 1 "Invalid instance IP"
    
    aws cloudformation describe-stacks \
        --stack-name "${stack_name}" \
        --region "${AWS_REGION}" \
        --query 'Stacks[0].Outputs' \
        --output table
}

# Wait for instance to be ready
wait_for_instance() {
    log_info "Waiting for instance to be ready..."
    
    if [ -z "${AWS_KEY_PATH}" ]; then
        error_exit 1 "AWS_KEY_PATH is required for instance readiness check"
    fi
    
    local max_attempts=30
    local attempt=0
    local delay=10
    
    while [ ${attempt} -lt ${max_attempts} ]; do
        if ssh -i "${AWS_KEY_PATH}" \
            -o StrictHostKeyChecking=no \
            -o ConnectTimeout=5 \
            -o UserKnownHostsFile=/dev/null \
            ubuntu@${INSTANCE_IP} \
            "test -f /var/log/user-data-complete.log" 2>/dev/null; then
            log_info "Instance is ready ✓"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log_info "Waiting... (${attempt}/${max_attempts})"
        sleep ${delay}
    done
    
    log_warn "Instance may not be fully ready, but continuing..."
    return 0
}

# Deploy application
deploy_application() {
    log_info "Deploying application to instance..."
    
    if [ -z "${INSTANCE_IP}" ]; then
        error_exit 1 "Instance IP not set"
    fi
    
    if [ -z "${AWS_KEY_PATH}" ]; then
        error_exit 1 "AWS_KEY_PATH is required for application deployment"
    fi
    
    validate_file "${AWS_KEY_PATH}" "SSH private key"
    validate_ip "${INSTANCE_IP}"
    
    # Wait for instance
    wait_for_instance
    
    # Check disk space on remote
    log_info "Checking disk space on remote instance..."
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} \
        "df -h / | tail -1" || log_warn "Could not check disk space"
    
    # Copy application files
    log_info "Copying application files..."
    rsync -avz \
        -e "ssh -i ${AWS_KEY_PATH} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.env' \
        --exclude='cloud/' \
        --exclude='*.md' \
        --progress \
        "${PROJECT_ROOT}/" \
        ubuntu@${INSTANCE_IP}:/opt/3d-prototype-ai/ || error_exit 1 "Failed to copy application files"
    
    # Deploy on remote instance
    log_info "Running deployment on remote instance..."
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} << 'REMOTE_EOF'
set -e
cd /opt/3d-prototype-ai
sudo chown -R ubuntu:ubuntu /opt/3d-prototype-ai

# Install Python dependencies if not using Docker
if [ -f "requirements.txt" ] && [ ! -f "docker-compose.yml" ]; then
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Deploy with Docker Compose if available
if [ -f "docker-compose.yml" ]; then
    docker-compose down || true
    docker-compose pull
    docker-compose up -d --build
    docker-compose ps
else
    # Use systemd service
    sudo systemctl daemon-reload
    sudo systemctl restart 3d-prototype-ai || true
    sudo systemctl status 3d-prototype-ai || true
fi

# Health check
sleep 5
if curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "✓ Application is healthy"
else
    echo "✗ Health check failed"
    exit 1
fi
REMOTE_EOF
    
    if [ $? -eq 0 ]; then
        log_info "Application deployed successfully ✓"
    else
        error_exit 1 "Application deployment failed"
    fi
}

# Show deployment information
show_deployment_info() {
    cat << EOF

${GREEN}==========================================${NC}
${GREEN}Deployment Summary${NC}
${GREEN}==========================================${NC}
Instance ID: ${INSTANCE_ID}
Public IP: ${INSTANCE_IP}
Application URL: http://${INSTANCE_IP}:8030
Health Check: http://${INSTANCE_IP}/health
API Docs: http://${INSTANCE_IP}/docs

SSH Command:
  ssh -i ${AWS_KEY_PATH} ubuntu@${INSTANCE_IP}

View logs:
  ssh -i ${AWS_KEY_PATH} ubuntu@${INSTANCE_IP} 'docker-compose logs -f'

Or use helper script:
  ./scripts/view_logs.sh --ip ${INSTANCE_IP} --key ${AWS_KEY_PATH}
${GREEN}==========================================${NC}

EOF
}

# Main execution
main() {
    parse_args "$@"
    
    log_info "Starting 3D Prototype AI deployment..."
    log_info "Deployment method: ${DEPLOYMENT_METHOD}"
    log_info "AWS Region: ${AWS_REGION}"
    log_info "Instance Type: ${INSTANCE_TYPE}"
    
    check_prerequisites
    
    case "${DEPLOYMENT_METHOD}" in
        terraform)
            deploy_with_terraform
            ;;
        cloudformation)
            deploy_with_cloudformation
            ;;
        ansible)
            log_error "Ansible deployment should be run separately"
            log_info "Use: ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml"
            exit 1
            ;;
        *)
            error_exit 1 "Unknown deployment method: ${DEPLOYMENT_METHOD}"
            ;;
    esac
    
    # Deploy application
    if [ "${SKIP_APP_DEPLOY}" != "true" ]; then
        deploy_application
    else
        log_info "Skipping application deployment (--skip-app flag set)"
    fi
    
    show_deployment_info
    
    send_notification "Deployment completed successfully - Instance: ${INSTANCE_ID}, IP: ${INSTANCE_IP}" "success"
    log_info "Deployment completed successfully! 🎉"
}

# Run main function
main "$@"
