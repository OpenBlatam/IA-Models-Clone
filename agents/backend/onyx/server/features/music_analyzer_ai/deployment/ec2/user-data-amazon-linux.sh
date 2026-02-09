#!/bin/bash
# EC2 User Data Script for Amazon Linux 2
# This script runs automatically when the EC2 instance starts

set -e

# Log everything
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=========================================="
echo "Music Analyzer AI - EC2 Setup"
echo "Starting installation..."
echo "=========================================="

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

# Install Git
yum install -y git

# Install other utilities
yum install -y curl wget unzip

# Create application directory
APP_DIR="/opt/music-analyzer-ai"
mkdir -p $APP_DIR
cd $APP_DIR

# Clone repository (or you can use S3, CodeDeploy, etc.)
# For now, we'll create a script that can be run manually
# In production, use CodeDeploy, CodePipeline, or S3

# Create environment file template
cat > $APP_DIR/.env << 'EOF'
ENVIRONMENT=production
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
LOG_LEVEL=INFO
CACHE_ENABLED=true
POSTGRES_PASSWORD=changeme
REDIS_PASSWORD=changeme
GRAFANA_PASSWORD=admin
DATABASE_URL=postgresql://music_analyzer:changeme@postgres:5432/music_analyzer_db
EOF

# Create startup script
cat > $APP_DIR/start.sh << 'EOF'
#!/bin/bash
cd /opt/music-analyzer-ai
docker-compose -f deployment/docker-compose.prod.yml up -d
EOF

chmod +x $APP_DIR/start.sh

# Create systemd service for auto-start
cat > /etc/systemd/system/music-analyzer-ai.service << 'EOF'
[Unit]
Description=Music Analyzer AI
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/music-analyzer-ai
ExecStart=/opt/music-analyzer-ai/start.sh
ExecStop=/usr/bin/docker-compose -f /opt/music-analyzer-ai/deployment/docker-compose.prod.yml down
User=ec2-user
Group=ec2-user

[Install]
WantedBy=multi-user.target
EOF

# Enable service (but don't start yet - need to deploy code first)
systemctl daemon-reload
# systemctl enable music-analyzer-ai.service

# Configure firewall (if using Security Groups, this may not be needed)
# Allow HTTP, HTTPS, and custom port
# firewall-cmd --permanent --add-service=http
# firewall-cmd --permanent --add-service=https
# firewall-cmd --permanent --add-port=8010/tcp
# firewall-cmd --reload

# Set up log rotation
cat > /etc/logrotate.d/music-analyzer-ai << 'EOF'
/var/log/music-analyzer-ai/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ec2-user ec2-user
}
EOF

echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Deploy your application code to: $APP_DIR"
echo "2. Update .env file with your credentials"
echo "3. Run: cd $APP_DIR && ./start.sh"
echo "   Or enable service: systemctl enable music-analyzer-ai.service"
echo ""
echo "Application will be available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8010"
echo "=========================================="




