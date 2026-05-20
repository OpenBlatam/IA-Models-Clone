#!/bin/bash
# Automatic deployment script for EC2
# Can be used in CI/CD pipelines or manually
# Follows DevOps best practices with error handling and rollback

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly INSTANCE_IP="${INSTANCE_IP:-}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"
readonly SKIP_BACKUP="${SKIP_BACKUP:-false}"
readonly SKIP_TESTS="${SKIP_TESTS:-false}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automatic deployment script for EC2 instance.

OPTIONS:
    -i, --ip IP              Instance IP address (required)
    -k, --key-path PATH      Path to SSH private key (required)
    -b, --skip-backup        Skip backup before deployment
    -t, --skip-tests         Skip tests before deployment
    -h, --help               Show this help message

ENVIRONMENT VARIABLES:
    INSTANCE_IP              EC2 instance IP
    AWS_KEY_PATH             Path to SSH private key
    SKIP_BACKUP              Skip backup (true/false)
    SKIP_TESTS               Skip tests (true/false)

EXAMPLES:
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem
    $0 --ip 1.2.3.4 --key-path ~/.ssh/key.pem --skip-backup

EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--ip)
                INSTANCE_IP="$2"
                shift 2
                ;;
            -k|--key-path)
                AWS_KEY_PATH="$2"
                shift 2
                ;;
            -b|--skip-backup)
                SKIP_BACKUP="true"
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS="true"
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
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    if [ -z "${INSTANCE_IP}" ]; then
        error_exit 1 "INSTANCE_IP is required"
    fi
    
    if [ -z "${AWS_KEY_PATH}" ]; then
        error_exit 1 "AWS_KEY_PATH is required"
    fi
    
    validate_file "${AWS_KEY_PATH}" "SSH private key"
    validate_ip "${INSTANCE_IP}"
    
    # Check SSH access
    log_info "Testing SSH access..."
    if ! ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=5 \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} \
        "echo 'SSH connection successful'" &> /dev/null; then
        error_exit 1 "Cannot connect to instance via SSH"
    fi
    
    log_info "Prerequisites validated ✓"
}

# Run tests
run_tests() {
    if [ "${SKIP_TESTS}" = "true" ]; then
        log_info "Skipping tests (--skip-tests flag set)"
        return 0
    fi
    
    log_info "Running tests..."
    
    if [ -d "${PROJECT_ROOT}/tests" ]; then
        cd "${PROJECT_ROOT}"
        
        if command -v pytest &> /dev/null; then
            if pytest tests/ -v --tb=short; then
                log_info "Tests passed ✓"
            else
                log_error "Tests failed"
                return 1
            fi
        else
            log_warn "pytest not found, skipping tests"
        fi
    else
        log_warn "Tests directory not found, skipping"
    fi
}

# Create backup
create_backup() {
    if [ "${SKIP_BACKUP}" = "true" ]; then
        log_info "Skipping backup (--skip-backup flag set)"
        return 0
    fi
    
    log_info "Creating backup before deployment..."
    
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} << 'REMOTE_EOF'
set -e
cd /opt/3d-prototype-ai

BACKUP_FILE="/tmp/app_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

if [ -f "docker-compose.yml" ]; then
  # Backup with Docker
  docker-compose exec -T api tar -czf ${BACKUP_FILE} . 2>/dev/null || \
  tar -czf ${BACKUP_FILE} storage/ logs/ .env docker-compose.yml 2>/dev/null || true
else
  # Backup without Docker
  tar -czf ${BACKUP_FILE} storage/ logs/ .env 2>/dev/null || true
fi

if [ -f "${BACKUP_FILE}" ]; then
  echo "Backup created: ${BACKUP_FILE}"
else
  echo "Warning: Backup creation may have failed"
fi
REMOTE_EOF
    
    log_info "Backup created ✓"
}

# Deploy application
deploy_application() {
    log_info "Deploying application to ${INSTANCE_IP}..."
    
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
        --exclude='tests/' \
        --exclude='.github/' \
        --delete \
        --progress \
        "${PROJECT_ROOT}/" \
        ubuntu@${INSTANCE_IP}:/opt/3d-prototype-ai/ || error_exit 1 "Failed to copy files"
    
    # Deploy on remote instance
    log_info "Updating application on remote instance..."
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} << 'REMOTE_EOF'
set -e
cd /opt/3d-prototype-ai

echo "Setting ownership..."
sudo chown -R ubuntu:ubuntu /opt/3d-prototype-ai

# Deploy with Docker Compose
if [ -f "docker-compose.yml" ]; then
  echo "Deploying with Docker Compose..."
  docker-compose pull || true
  docker-compose up -d --build
  
  echo "Waiting for services to start..."
  sleep 10
  
  # Health check
  if docker-compose exec -T api curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "✓ Application is healthy"
  else
    echo "✗ Health check failed"
    docker-compose logs --tail=50
    exit 1
  fi
else
  # Deploy without Docker
  echo "Deploying without Docker..."
  
  if [ -f "requirements.txt" ]; then
    if [ ! -d "venv" ]; then
      python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
  fi
  
  sudo systemctl daemon-reload
  sudo systemctl restart 3d-prototype-ai || true
  
  sleep 5
  
  if curl -f http://localhost:8030/health > /dev/null 2>&1; then
    echo "✓ Application is healthy"
  else
    echo "✗ Health check failed"
    sudo journalctl -u 3d-prototype-ai -n 50 --no-pager
    exit 1
  fi
fi

echo "Deployment completed successfully!"
REMOTE_EOF
    
    if [ $? -eq 0 ]; then
        log_info "Application deployed successfully ✓"
    else
        error_exit 1 "Application deployment failed"
    fi
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    local max_attempts=10
    local attempt=0
    
    while [ ${attempt} -lt ${max_attempts} ]; do
        if curl -f -m 10 "http://${INSTANCE_IP}:8030/health" > /dev/null 2>&1; then
            log_info "Deployment verified - Application is healthy ✓"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log_info "Waiting for application... (${attempt}/${max_attempts})"
        sleep 5
    done
    
    log_error "Deployment verification failed"
    return 1
}

# Rollback function
rollback_deployment() {
    log_warn "Deployment failed, attempting rollback..."
    
    ssh -i "${AWS_KEY_PATH}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${INSTANCE_IP} << 'REMOTE_EOF' || true
cd /opt/3d-prototype-ai

# Find latest backup
BACKUP_FILE=$(ls -t /tmp/app_backup_*.tar.gz 2>/dev/null | head -1)

if [ -n "${BACKUP_FILE}" ] && [ -f "${BACKUP_FILE}" ]; then
  echo "Restoring from backup: ${BACKUP_FILE}"
  tar -xzf "${BACKUP_FILE}" -C /opt/3d-prototype-ai
  
  if [ -f "docker-compose.yml" ]; then
    docker-compose restart
  else
    sudo systemctl restart 3d-prototype-ai
  fi
  
  echo "Rollback completed"
else
  echo "No backup found, cannot rollback"
fi
REMOTE_EOF
}

# Main function
main() {
    parse_args "$@"
    
    log_info "Starting automatic deployment..."
    log_info "Target instance: ${INSTANCE_IP}"
    echo ""
    
    setup_trap
    
    # Validate prerequisites
    validate_prerequisites
    echo ""
    
    # Run tests
    if ! run_tests; then
        error_exit 1 "Tests failed, deployment aborted"
    fi
    echo ""
    
    # Create backup
    create_backup
    echo ""
    
    # Deploy application
    if ! deploy_application; then
        rollback_deployment
        error_exit 1 "Deployment failed and rollback attempted"
    fi
    echo ""
    
    # Verify deployment
    if ! verify_deployment; then
        rollback_deployment
        error_exit 1 "Deployment verification failed and rollback attempted"
    fi
    echo ""
    
    # Success
    log_info "=========================================="
    log_info "Deployment Summary"
    log_info "=========================================="
    log_info "Instance IP: ${INSTANCE_IP}"
    log_info "Application URL: http://${INSTANCE_IP}:8030"
    log_info "Health Check: http://${INSTANCE_IP}/health"
    log_info "API Docs: http://${INSTANCE_IP}/docs"
    log_info "=========================================="
    echo ""
    
    send_notification "Deployment completed successfully - Instance: ${INSTANCE_IP}" "success"
    log_info "Automatic deployment completed successfully! 🎉"
}

# Set up error handling
trap 'rollback_deployment; exit 1' ERR

main "$@"

