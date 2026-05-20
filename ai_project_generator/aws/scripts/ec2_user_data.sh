#!/bin/bash

###############################################################################
# EC2 User Data Script for AI Project Generator
# This script runs on EC2 instance launch to perform initial setup
###############################################################################

set -euo pipefail

# Log all output
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

PROJECT_NAME="${project_name:-ai-project-generator}"
ENVIRONMENT="${environment:-production}"
APP_PORT="${app_port:-8020}"
REDIS_URL="${redis_url:-redis://localhost:6379}"

log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

###############################################################################
# System Update
###############################################################################

log_info "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

###############################################################################
# Install Essential Packages
###############################################################################

log_info "Installing essential packages..."
apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    htop \
    net-tools \
    vim \
    jq \
    ufw \
    redis-server \
    libpq-dev \
    libssl-dev \
    libffi-dev

###############################################################################
# Configure Firewall
###############################################################################

log_info "Configuring firewall..."
ufw --force enable
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow ${APP_PORT}/tcp

###############################################################################
# Install Docker
###############################################################################

log_info "Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Add Docker's official GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    
    # Set up Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Add ubuntu user to docker group
    usermod -aG docker ubuntu
    
    # Start Docker service
    systemctl enable docker
    systemctl start docker
    
    log_info "Docker installed successfully."
else
    log_info "Docker is already installed."
fi

###############################################################################
# Install Docker Compose (standalone)
###############################################################################

log_info "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)
    curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    log_info "Docker Compose installed successfully."
else
    log_info "Docker Compose is already installed."
fi

###############################################################################
# Configure Redis
###############################################################################

log_info "Configuring Redis..."
systemctl enable redis-server
systemctl start redis-server

# Configure Redis for application use
sed -i 's/^bind 127.0.0.1/bind 127.0.0.1/' /etc/redis/redis.conf
sed -i 's/^# maxmemory <bytes>/maxmemory 512mb/' /etc/redis/redis.conf
sed -i 's/^# maxmemory-policy noeviction/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf
systemctl restart redis-server

log_info "Redis configured and started."

###############################################################################
# Configure Timezone
###############################################################################

log_info "Configuring timezone..."
timedatectl set-timezone UTC

###############################################################################
# Install CloudWatch Agent
###############################################################################

log_info "Installing CloudWatch agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb -O /tmp/amazon-cloudwatch-agent.deb
dpkg -i -E /tmp/amazon-cloudwatch-agent.deb || true
rm /tmp/amazon-cloudwatch-agent.deb

###############################################################################
# Create Application Directory
###############################################################################

log_info "Creating application directory..."
mkdir -p /opt/${PROJECT_NAME}
chown ubuntu:ubuntu /opt/${PROJECT_NAME}

###############################################################################
# Configure Automatic Security Updates
###############################################################################

log_info "Configuring automatic security updates..."
apt-get install -y unattended-upgrades
cat > /etc/apt/apt.conf.d/50unattended-upgrades <<EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

###############################################################################
# Install AWS CLI
###############################################################################

log_info "Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
    log_info "AWS CLI installed successfully."
else
    log_info "AWS CLI is already installed."
fi

###############################################################################
# Create Python Virtual Environment
###############################################################################

log_info "Creating Python virtual environment..."
mkdir -p /opt/venv
python3.11 -m venv /opt/venv
/opt/venv/bin/pip install --upgrade pip setuptools wheel

###############################################################################
# Final Steps
###############################################################################

log_info "User data script completed successfully."
log_info "Project: ${PROJECT_NAME}"
log_info "Environment: ${ENVIRONMENT}"
log_info "Application Port: ${APP_PORT}"
log_info "Redis URL: ${REDIS_URL}"
log_info "Instance is ready for application deployment."

