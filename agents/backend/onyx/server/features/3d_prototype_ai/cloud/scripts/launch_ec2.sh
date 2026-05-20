#!/bin/bash
# Quick EC2 launch script - Launches an EC2 instance with user data
# This is a simpler alternative to Terraform for quick deployments

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load configuration
if [ -f "${CLOUD_DIR}/.env" ]; then
    source "${CLOUD_DIR}/.env"
fi

# Default values
AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${AWS_INSTANCE_TYPE:-t3.large}"
KEY_NAME="${AWS_KEY_NAME}"
SECURITY_GROUP="${AWS_SECURITY_GROUP}"
AMI_ID="${AWS_AMI_ID}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not installed"
    exit 1
fi

if [ -z "${KEY_NAME}" ]; then
    log_error "AWS_KEY_NAME not set in .env file"
    exit 1
fi

# Get latest Ubuntu AMI if not specified
if [ -z "${AMI_ID}" ]; then
    log_info "Finding latest Ubuntu 22.04 AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/h2-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text \
        --region "${AWS_REGION}")
    log_info "Using AMI: ${AMI_ID}"
fi

# Create security group if not provided
if [ -z "${SECURITY_GROUP}" ]; then
    log_info "Creating security group..."
    SECURITY_GROUP=$(aws ec2 create-security-group \
        --group-name "3d-prototype-ai-sg-$(date +%s)" \
        --description "Security group for 3D Prototype AI" \
        --region "${AWS_REGION}" \
        --query 'GroupId' \
        --output text)
    
    # Add rules
    aws ec2 authorize-security-group-ingress \
        --group-id "${SECURITY_GROUP}" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "${AWS_REGION}" > /dev/null
    
    aws ec2 authorize-security-group-ingress \
        --group-id "${SECURITY_GROUP}" \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region "${AWS_REGION}" > /dev/null
    
    aws ec2 authorize-security-group-ingress \
        --group-id "${SECURITY_GROUP}" \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region "${AWS_REGION}" > /dev/null
    
    aws ec2 authorize-security-group-ingress \
        --group-id "${SECURITY_GROUP}" \
        --protocol tcp \
        --port 8030 \
        --cidr 0.0.0.0/0 \
        --region "${AWS_REGION}" > /dev/null
    
    log_info "Security group created: ${SECURITY_GROUP}"
fi

# Read user data script
USER_DATA_FILE="${CLOUD_DIR}/user_data/init.sh"
if [ ! -f "${USER_DATA_FILE}" ]; then
    log_error "User data script not found: ${USER_DATA_FILE}"
    exit 1
fi

# Launch instance
log_info "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SECURITY_GROUP}" \
    --user-data "file://${USER_DATA_FILE}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=3d-prototype-ai},{Key=Project,Value=3D-Prototype-AI}]" \
    --region "${AWS_REGION}" \
    --query 'Instances[0].InstanceId' \
    --output text)

log_info "Instance launched: ${INSTANCE_ID}"

# Wait for instance to be running
log_info "Waiting for instance to be running..."
aws ec2 wait instance-running \
    --instance-ids "${INSTANCE_ID}" \
    --region "${AWS_REGION}"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --region "${AWS_REGION}" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

log_info "Instance is running!"
log_info "Instance ID: ${INSTANCE_ID}"
log_info "Public IP: ${PUBLIC_IP}"
log_info ""
log_info "SSH command:"
log_info "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
log_info ""
log_info "Waiting for initialization to complete (this may take a few minutes)..."
log_info "You can check progress with:"
log_info "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/user-data.log'"

