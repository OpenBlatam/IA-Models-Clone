#!/bin/bash
# Alert manager script
# Manages alerts and notifications

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly ALERT_CONFIG="${ALERT_CONFIG:-${CLOUD_DIR}/.alerts.conf}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Manage alerts and notifications.

COMMANDS:
    setup                Setup alert configuration
    test                 Test alert channels
    send TYPE MESSAGE     Send alert
    list                 List configured alerts
    enable ALERT         Enable alert
    disable ALERT        Disable alert

OPTIONS:
    -c, --config FILE     Alert configuration file
    -h, --help           Show this help message

EXAMPLES:
    $0 setup
    $0 test
    $0 send critical "Application is down"
    $0 list

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    ALERT_TYPE=""
    MESSAGE=""
    ALERT_NAME=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                ALERT_CONFIG="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            setup|test|list)
                COMMAND="$1"
                shift
                break
                ;;
            send)
                COMMAND="$1"
                ALERT_TYPE="$2"
                MESSAGE="$3"
                shift 3
                break
                ;;
            enable|disable)
                COMMAND="$1"
                ALERT_NAME="$2"
                shift 2
                break
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Setup alert configuration
setup_alerts() {
    log_info "Setting up alert configuration..."
    
    cat > "${ALERT_CONFIG}" << EOF
# Alert Configuration
# Generated: $(date)

# Slack Configuration
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
SLACK_CHANNEL="${SLACK_CHANNEL:-#alerts}"

# Email Configuration
EMAIL_SMTP_SERVER="${EMAIL_SMTP_SERVER:-}"
EMAIL_SMTP_PORT="${EMAIL_SMTP_PORT:-587}"
EMAIL_FROM="${EMAIL_FROM:-}"
EMAIL_TO="${EMAIL_TO:-}"

# Webhook Configuration
WEBHOOK_URL="${WEBHOOK_URL:-}"

# Alert Thresholds
CPU_THRESHOLD_HIGH=80
CPU_THRESHOLD_CRITICAL=95
MEMORY_THRESHOLD_HIGH=85
MEMORY_THRESHOLD_CRITICAL=95
DISK_THRESHOLD_HIGH=85
DISK_THRESHOLD_CRITICAL=95

# Alert Settings
ALERT_COOLDOWN=300
ALERT_ENABLED=true
EOF
    
    log_info "Alert configuration created: ${ALERT_CONFIG}"
    log_info "Please edit the file to configure your alert channels"
}

# Send alert
send_alert() {
    local alert_type="${1}"
    local message="${2}"
    
    if [ ! -f "${ALERT_CONFIG}" ]; then
        log_warn "Alert configuration not found. Run 'setup' first."
        return 1
    fi
    
    source "${ALERT_CONFIG}"
    
    local color
    local emoji
    case "${alert_type}" in
        critical)
            color="danger"
            emoji="🚨"
            ;;
        warning)
            color="warning"
            emoji="⚠️"
            ;;
        info)
            color="good"
            emoji="ℹ️"
            ;;
        *)
            color="good"
            emoji="ℹ️"
            ;;
    esac
    
    # Send Slack alert
    if [ -n "${SLACK_WEBHOOK_URL}" ]; then
        local payload
        payload=$(cat << EOF
{
  "text": "${emoji} ${message}",
  "channel": "${SLACK_CHANNEL}",
  "attachments": [
    {
      "color": "${color}",
      "fields": [
        {
          "title": "Type",
          "value": "${alert_type}",
          "short": true
        },
        {
          "title": "Time",
          "value": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
          "short": true
        }
      ]
    }
  ]
}
EOF
)
        curl -X POST -H 'Content-type: application/json' \
            --data "${payload}" \
            "${SLACK_WEBHOOK_URL}" 2>/dev/null && log_info "Slack alert sent" || log_warn "Failed to send Slack alert"
    fi
    
    # Send webhook alert
    if [ -n "${WEBHOOK_URL}" ]; then
        local payload
        payload=$(cat << EOF
{
  "type": "${alert_type}",
  "message": "${message}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
)
        curl -X POST -H 'Content-type: application/json' \
            --data "${payload}" \
            "${WEBHOOK_URL}" 2>/dev/null && log_info "Webhook alert sent" || log_warn "Failed to send webhook alert"
    fi
    
    log_info "Alert sent: ${alert_type} - ${message}"
}

# Test alerts
test_alerts() {
    log_info "Testing alert channels..."
    
    send_alert "info" "This is a test alert from the alert manager"
    
    log_info "Test alerts sent. Check your configured channels."
}

# List alerts
list_alerts() {
    if [ ! -f "${ALERT_CONFIG}" ]; then
        log_warn "Alert configuration not found. Run 'setup' first."
        return 1
    fi
    
    log_info "Alert Configuration:"
    cat "${ALERT_CONFIG}"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        setup)
            setup_alerts
            ;;
        test)
            test_alerts
            ;;
        send)
            if [ -z "${ALERT_TYPE}" ] || [ -z "${MESSAGE}" ]; then
                error_exit 1 "Alert type and message are required"
            fi
            send_alert "${ALERT_TYPE}" "${MESSAGE}"
            ;;
        list)
            list_alerts
            ;;
        enable|disable)
            log_info "${COMMAND} functionality - implement as needed"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


