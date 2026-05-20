#!/bin/bash
# Prometheus & Grafana Setup Script
# Advanced observability with Prometheus metrics and Grafana dashboards

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMETHEUS_DIR="/opt/prometheus"
GRAFANA_DIR="/opt/grafana"
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Install Prometheus
install_prometheus() {
    log_info "Installing Prometheus..."
    
    local prometheus_version="2.45.0"
    local prometheus_url="https://github.com/prometheus/prometheus/releases/download/v${prometheus_version}/prometheus-${prometheus_version}.linux-amd64.tar.gz"
    
    sudo mkdir -p "$PROMETHEUS_DIR"
    cd /tmp
    
    wget -q "$prometheus_url" -O prometheus.tar.gz
    tar -xzf prometheus.tar.gz
    sudo mv prometheus-${prometheus_version}.linux-amd64/* "$PROMETHEUS_DIR/"
    
    sudo chown -R ubuntu:ubuntu "$PROMETHEUS_DIR"
    
    log_info "✅ Prometheus installed"
}

# Install Grafana
install_grafana() {
    log_info "Installing Grafana..."
    
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
    
    wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y grafana
    
    sudo systemctl enable grafana-server
    sudo systemctl start grafana-server
    
    log_info "✅ Grafana installed"
}

# Configure Prometheus
configure_prometheus() {
    log_info "Configuring Prometheus..."
    
    sudo tee "$PROMETHEUS_DIR/prometheus.yml" > /dev/null <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'ai-project-generator'
    environment: 'production'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'ai-project-generator'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8020']
        labels:
          service: 'ai-project-generator'
          environment: 'production'

  - job_name: 'nginx'
    static_configs:
      - targets: ['localhost:9113']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']
EOF
    
    sudo chown ubuntu:ubuntu "$PROMETHEUS_DIR/prometheus.yml"
    
    log_info "✅ Prometheus configured"
}

# Create Prometheus systemd service
create_prometheus_service() {
    log_info "Creating Prometheus systemd service..."
    
    sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=$PROMETHEUS_DIR/prometheus \\
    --config.file=$PROMETHEUS_DIR/prometheus.yml \\
    --storage.tsdb.path=$PROMETHEUS_DIR/data \\
    --web.console.libraries=$PROMETHEUS_DIR/console_libraries \\
    --web.console.templates=$PROMETHEUS_DIR/consoles \\
    --web.listen-address=0.0.0.0:$PROMETHEUS_PORT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable prometheus
    sudo systemctl start prometheus
    
    log_info "✅ Prometheus service created and started"
}

# Configure Grafana datasource
configure_grafana_datasource() {
    log_info "Configuring Grafana datasource..."
    
    # Wait for Grafana to be ready
    sleep 10
    
    # Get Grafana admin password
    local admin_password=$(sudo cat /etc/grafana/grafana.ini | grep "^admin_password" | cut -d'=' -f2 | tr -d ' ' || echo "admin")
    
    # Add Prometheus datasource
    curl -X POST "http://localhost:$GRAFANA_PORT/api/datasources" \
        -H "Content-Type: application/json" \
        -u "admin:$admin_password" \
        -d '{
            "name": "Prometheus",
            "type": "prometheus",
            "url": "http://localhost:9090",
            "access": "proxy",
            "isDefault": true
        }' 2>/dev/null || log_warn "Datasource may already exist"
    
    log_info "✅ Grafana datasource configured"
}

# Install exporters
install_exporters() {
    log_info "Installing Prometheus exporters..."
    
    # Node Exporter
    local node_exporter_version="1.6.1"
    wget -q "https://github.com/prometheus/node_exporter/releases/download/v${node_exporter_version}/node_exporter-${node_exporter_version}.linux-amd64.tar.gz" -O /tmp/node_exporter.tar.gz
    tar -xzf /tmp/node_exporter.tar.gz -C /tmp
    sudo mv /tmp/node_exporter-${node_exporter_version}.linux-amd64/node_exporter /usr/local/bin/
    sudo chmod +x /usr/local/bin/node_exporter
    
    # Create node exporter service
    sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=/usr/local/bin/node_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable node_exporter
    sudo systemctl start node_exporter
    
    log_info "✅ Node exporter installed"
}

# Setup Grafana dashboards
setup_grafana_dashboards() {
    log_info "Setting up Grafana dashboards..."
    
    # Dashboard IDs for common dashboards
    local dashboards=(
        "1860"  # Node Exporter Full
        "11074" # Docker Container & Host Metrics
        "12708" # Redis Dashboard
        "12708" # Nginx Metrics
    )
    
    log_info "Import dashboard IDs: ${dashboards[*]}"
    log_info "Import these dashboards manually from grafana.com/dashboards"
    
    log_info "✅ Grafana dashboard setup instructions provided"
}

# Main function
main() {
    case "${1:-install}" in
        install)
            install_prometheus
            install_grafana
            configure_prometheus
            create_prometheus_service
            install_exporters
            configure_grafana_datasource
            setup_grafana_dashboards
            log_info "✅ Prometheus & Grafana setup completed"
            log_info "Prometheus: http://localhost:$PROMETHEUS_PORT"
            log_info "Grafana: http://localhost:$GRAFANA_PORT (admin/admin)"
            ;;
        start)
            sudo systemctl start prometheus
            sudo systemctl start grafana-server
            sudo systemctl start node_exporter
            ;;
        stop)
            sudo systemctl stop prometheus
            sudo systemctl stop grafana-server
            sudo systemctl stop node_exporter
            ;;
        status)
            sudo systemctl status prometheus
            sudo systemctl status grafana-server
            sudo systemctl status node_exporter
            ;;
        *)
            echo "Usage: $0 {install|start|stop|status}"
            exit 1
            ;;
    esac
}

main "$@"



