#!/bin/bash
# EC2 User Data Script - Initializes the server for 3D Prototype AI deployment
# This script runs automatically when the EC2 instance is launched

set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=========================================="
echo "3D Prototype AI - EC2 Initialization"
echo "=========================================="
echo "Started at: $(date)"

# Variables
APP_PORT="${app_port:-8030}"
APP_HOST="${app_host:-0.0.0.0}"
PROJECT_NAME="${project_name:-3d-prototype-ai}"
APP_USER="ubuntu"
APP_DIR="/opt/${PROJECT_NAME}"

# Update system
echo "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

# Install essential packages
echo "Installing essential packages..."
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
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    nginx \
    supervisor \
    htop \
    net-tools \
    jq \
    awscli

# Install Docker
echo "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ${APP_USER}
fi

# Install Docker Compose (standalone)
echo "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
fi

# Create application directory
echo "Creating application directory..."
mkdir -p ${APP_DIR}
mkdir -p ${APP_DIR}/storage/prototypes
mkdir -p ${APP_DIR}/storage/backups
mkdir -p ${APP_DIR}/logs
chown -R ${APP_USER}:${APP_USER} ${APP_DIR}

# Configure Nginx
echo "Configuring Nginx..."
cat > /etc/nginx/sites-available/${PROJECT_NAME} <<EOF
server {
    listen 80;
    server_name _;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /health {
        proxy_pass http://127.0.0.1:${APP_PORT}/health;
        access_log off;
    }
}
EOF

# Enable Nginx site
ln -sf /etc/nginx/sites-available/${PROJECT_NAME} /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# Configure firewall (UFW)
echo "Configuring firewall..."
ufw --force enable
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow ${APP_PORT}/tcp

# Install CloudWatch agent (optional)
echo "Installing CloudWatch agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb -O /tmp/amazon-cloudwatch-agent.deb
dpkg -i -E /tmp/amazon-cloudwatch-agent.deb || true
rm -f /tmp/amazon-cloudwatch-agent.deb

# Create systemd service for application (if not using Docker)
cat > /etc/systemd/system/${PROJECT_NAME}.service <<EOF
[Unit]
Description=3D Prototype AI Application
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 -m uvicorn main:app --host ${APP_HOST} --port ${APP_PORT}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create deployment script
cat > ${APP_DIR}/deploy.sh <<'DEPLOY_EOF'
#!/bin/bash
set -e

APP_DIR="/opt/3d-prototype-ai"
cd ${APP_DIR}

echo "Deploying 3D Prototype AI..."

# Pull latest code (if using git)
if [ -d ".git" ]; then
    git pull origin main
fi

# Build and start with Docker Compose
if [ -f "docker-compose.yml" ]; then
    docker-compose pull
    docker-compose up -d --build
    docker-compose ps
else
    # Fallback to systemd
    systemctl daemon-reload
    systemctl restart 3d-prototype-ai
fi

echo "Deployment completed!"
DEPLOY_EOF

chmod +x ${APP_DIR}/deploy.sh
chown ${APP_USER}:${APP_USER} ${APP_DIR}/deploy.sh

# Create health check script
cat > /usr/local/bin/health-check.sh <<'HEALTH_EOF'
#!/bin/bash
curl -f http://localhost:${APP_PORT}/health || exit 1
HEALTH_EOF

chmod +x /usr/local/bin/health-check.sh

# Set up log rotation
cat > /etc/logrotate.d/${PROJECT_NAME} <<EOF
${APP_DIR}/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 ${APP_USER} ${APP_USER}
    sharedscripts
}
EOF

# Final setup
echo "Finalizing setup..."
systemctl enable nginx
systemctl enable docker

# Create info file
cat > ${APP_DIR}/instance-info.txt <<EOF
Instance initialized: $(date)
Project: ${PROJECT_NAME}
App Port: ${APP_PORT}
App Host: ${APP_HOST}
Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)
Region: $(curl -s http://169.254.169.254/latest/meta-data/placement/region)
EOF

echo "=========================================="
echo "Initialization completed at: $(date)"
echo "Next steps:"
echo "1. SSH into the instance"
echo "2. Copy your application files to ${APP_DIR}"
echo "3. Run ${APP_DIR}/deploy.sh"
echo "=========================================="

# Signal completion
touch /var/log/user-data-complete.log
echo "User data script completed successfully" > /var/log/user-data-complete.log

