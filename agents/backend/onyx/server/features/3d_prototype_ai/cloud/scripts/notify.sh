#!/bin/bash
# Notification script
# Sends deployment notifications via various channels

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] MESSAGE

Send deployment notifications.

OPTIONS:
    -t, --type TYPE          Notification type (success|failure|info)
    -c, --channel CHANNEL    Channel (slack|email|webhook)
    -u, --url URL            Webhook URL
    -h, --help               Show this help message

EXAMPLES:
    $0 --type success --channel slack "Deployment completed"
    $0 --type failure "Deployment failed"

EOF
}

# Parse arguments
parse_args() {
    NOTIFICATION_TYPE="info"
    CHANNEL=""
    WEBHOOK_URL=""
    MESSAGE=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                NOTIFICATION_TYPE="$2"
                shift 2
                ;;
            -c|--channel)
                CHANNEL="$2"
                shift 2
                ;;
            -u|--url)
                WEBHOOK_URL="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                MESSAGE="$*"
                break
                ;;
        esac
    done
}

# Send Slack notification
send_slack() {
    local message="${1}"
    local webhook_url="${2}"
    local type="${3}"
    
    local color
    case "${type}" in
        success)
            color="good"
            emoji="✅"
            ;;
        failure)
            color="danger"
            emoji="❌"
            ;;
        *)
            color="#36a64f"
            emoji="ℹ️"
            ;;
    esac
    
    local payload
    payload=$(cat << EOF
{
  "text": "${emoji} ${message}",
  "attachments": [
    {
      "color": "${color}",
      "fields": [
        {
          "title": "Time",
          "value": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
          "short": true
        },
        {
          "title": "Type",
          "value": "${type}",
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
        "${webhook_url}" 2>/dev/null || log_warn "Failed to send Slack notification"
}

# Send webhook notification
send_webhook() {
    local message="${1}"
    local webhook_url="${2}"
    local type="${3}"
    
    local payload
    payload=$(cat << EOF
{
  "message": "${message}",
  "type": "${type}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
        --data "${payload}" \
        "${webhook_url}" 2>/dev/null || log_warn "Failed to send webhook notification"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${MESSAGE}" ]; then
        error_exit 1 "Message is required"
    fi
    
    # Send notification based on channel
    case "${CHANNEL}" in
        slack)
            if [ -z "${WEBHOOK_URL}" ]; then
                WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
            fi
            if [ -n "${WEBHOOK_URL}" ]; then
                send_slack "${MESSAGE}" "${WEBHOOK_URL}" "${NOTIFICATION_TYPE}"
                log_info "Slack notification sent"
            else
                log_warn "Slack webhook URL not configured"
            fi
            ;;
        webhook)
            if [ -z "${WEBHOOK_URL}" ]; then
                error_exit 1 "Webhook URL is required for webhook channel"
            fi
            send_webhook "${MESSAGE}" "${WEBHOOK_URL}" "${NOTIFICATION_TYPE}"
            log_info "Webhook notification sent"
            ;;
        *)
            # Default: just log
            log_info "Notification [${NOTIFICATION_TYPE}]: ${MESSAGE}"
            ;;
    esac
}

main "$@"


