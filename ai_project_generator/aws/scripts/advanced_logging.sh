#!/bin/bash
# Advanced Logging and Tracing Script
# Centralized logging with structured logs and distributed tracing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/var/log/ai-project-generator"
LOG_LEVEL="${LOG_LEVEL:-INFO}"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
ENABLE_TRACING="${ENABLE_TRACING:-true}"
CLOUDWATCH_ENABLED="${CLOUDWATCH_ENABLED:-true}"

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

# Setup logging directory
setup_logging() {
    log_info "Setting up advanced logging..."
    
    sudo mkdir -p "$LOG_DIR"
    sudo chown -R ubuntu:ubuntu "$LOG_DIR"
    
    # Create log rotation config
    sudo tee /etc/logrotate.d/ai-project-generator > /dev/null <<EOF
$LOG_DIR/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 ubuntu ubuntu
    sharedscripts
    postrotate
        systemctl reload ai-project-generator > /dev/null 2>&1 || true
    endscript
}
EOF
    
    log_info "✅ Logging directory configured"
}

# Configure structured logging
configure_structured_logging() {
    log_info "Configuring structured logging (JSON format)..."
    
    # Create logging configuration
    sudo tee /opt/ai-project-generator/config/logging.json > /dev/null <<EOF
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "json": {
      "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
      "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d"
    },
    "detailed": {
      "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    }
  },
  "handlers": {
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "filename": "$LOG_DIR/app.log",
      "maxBytes": 10485760,
      "backupCount": 10,
      "formatter": "json",
      "level": "$LOG_LEVEL"
    },
    "console": {
      "class": "logging.StreamHandler",
      "formatter": "detailed",
      "level": "$LOG_LEVEL"
    },
    "cloudwatch": {
      "class": "watchtower.CloudWatchLogHandler",
      "log_group": "/aws/ec2/ai-project-generator",
      "stream_name": "app",
      "formatter": "json",
      "level": "$LOG_LEVEL"
    }
  },
  "root": {
    "level": "$LOG_LEVEL",
    "handlers": ["file", "console"]
  },
  "loggers": {
    "ai_project_generator": {
      "level": "$LOG_LEVEL",
      "handlers": ["file", "console", "cloudwatch"],
      "propagate": false
    }
  }
}
EOF
    
    sudo chown ubuntu:ubuntu /opt/ai-project-generator/config/logging.json
    
    log_info "✅ Structured logging configured"
}

# Setup CloudWatch logging
setup_cloudwatch() {
    if [ "$CLOUDWATCH_ENABLED" != "true" ]; then
        log_info "CloudWatch logging disabled"
        return 0
    fi
    
    log_info "Setting up CloudWatch logging..."
    
    # Install watchtower if not available
    if ! python3 -c "import watchtower" 2>/dev/null; then
        log_info "Installing watchtower..."
        pip3 install watchtower || {
            log_error "Failed to install watchtower"
            return 1
        }
    fi
    
    # Create CloudWatch log group
    aws logs create-log-group \
        --log-group-name "/aws/ec2/ai-project-generator" \
        2>/dev/null || log_warn "Log group may already exist"
    
    log_info "✅ CloudWatch logging configured"
}

# Setup distributed tracing
setup_tracing() {
    if [ "$ENABLE_TRACING" != "true" ]; then
        log_info "Distributed tracing disabled"
        return 0
    fi
    
    log_info "Setting up distributed tracing..."
    
    # Install OpenTelemetry if not available
    if ! python3 -c "import opentelemetry" 2>/dev/null; then
        log_info "Installing OpenTelemetry..."
        pip3 install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi || {
            log_error "Failed to install OpenTelemetry"
            return 1
        }
    fi
    
    # Create tracing configuration
    sudo tee /opt/ai-project-generator/config/tracing.json > /dev/null <<EOF
{
  "service_name": "ai-project-generator",
  "service_version": "1.0.0",
  "exporter": {
    "type": "otlp",
    "endpoint": "http://localhost:4318",
    "headers": {}
  },
  "sampling": {
    "type": "trace_id_ratio",
    "ratio": 0.1
  },
  "instrumentation": {
    "fastapi": true,
    "httpx": true,
    "redis": true,
    "postgresql": true
  }
}
EOF
    
    sudo chown ubuntu:ubuntu /opt/ai-project-generator/config/tracing.json
    
    log_info "✅ Distributed tracing configured"
}

# Analyze logs
analyze_logs() {
    local log_file="${1:-$LOG_DIR/app.log}"
    local pattern="${2:-}"
    
    log_info "Analyzing logs: $log_file"
    
    if [ ! -f "$log_file" ]; then
        log_error "Log file not found: $log_file"
        return 1
    fi
    
    # Error analysis
    log_info "=== Error Analysis ==="
    grep -i "error\|exception\|failed" "$log_file" | tail -20
    
    # Performance analysis
    log_info "=== Performance Analysis ==="
    grep -E "duration|response_time|latency" "$log_file" | \
        awk '{print $NF}' | \
        awk '{sum+=$1; count++} END {if(count>0) print "Avg:", sum/count, "Max:", max}'
    
    # Request analysis
    log_info "=== Request Analysis ==="
    grep -E "GET|POST|PUT|DELETE" "$log_file" | \
        awk '{print $NF}' | \
        sort | uniq -c | sort -rn | head -10
    
    if [ -n "$pattern" ]; then
        log_info "=== Pattern Search: $pattern ==="
        grep -i "$pattern" "$log_file" | tail -20
    fi
}

# Export logs to S3
export_logs_to_s3() {
    local s3_bucket="${1:-ai-project-generator-logs}"
    local date_prefix=$(date +%Y/%m/%d)
    
    log_info "Exporting logs to S3: s3://$s3_bucket/$date_prefix/"
    
    # Create S3 bucket if it doesn't exist
    aws s3 mb "s3://$s3_bucket" 2>/dev/null || log_warn "Bucket may already exist"
    
    # Upload logs
    aws s3 sync "$LOG_DIR" "s3://$s3_bucket/$date_prefix/" \
        --exclude "*.gz" \
        --exclude "*.zip" || {
        log_error "Failed to upload logs to S3"
        return 1
    }
    
    log_info "✅ Logs exported to S3"
}

# Main function
main() {
    case "${1:-setup}" in
        setup)
            setup_logging
            configure_structured_logging
            setup_cloudwatch
            setup_tracing
            log_info "✅ Advanced logging setup completed"
            ;;
        analyze)
            analyze_logs "${2:-}" "${3:-}"
            ;;
        export)
            export_logs_to_s3 "${2:-}"
            ;;
        *)
            echo "Usage: $0 {setup|analyze|export} [log_file] [pattern|s3_bucket]"
            echo "Environment variables:"
            echo "  LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)"
            echo "  ENABLE_TRACING: true or false (default: true)"
            echo "  CLOUDWATCH_ENABLED: true or false (default: true)"
            exit 1
            ;;
    esac
}

main "$@"



