#!/bin/bash
# Audit trail script
# Tracks and logs all deployment activities

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Default values
readonly AUDIT_LOG="${AUDIT_LOG:-${CLOUD_DIR}/audit.log}"
readonly AUDIT_RETENTION_DAYS="${AUDIT_RETENTION_DAYS:-365}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Manage audit trail.

COMMANDS:
    log ACTION DETAILS    Log an action
    view                  View audit log
    search PATTERN        Search audit log
    export                Export audit log
    clean                 Clean old audit entries

OPTIONS:
    -l, --log-file FILE   Audit log file (default: ./audit.log)
    -r, --retention DAYS  Retention days (default: 365)
    -h, --help           Show this help message

EXAMPLES:
    $0 log "deployment" "Deployed version 1.2.3"
    $0 view
    $0 search "deployment"
    $0 export

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    ACTION=""
    DETAILS=""
    PATTERN=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -l|--log-file)
                AUDIT_LOG="$2"
                shift 2
                ;;
            -r|--retention)
                AUDIT_RETENTION_DAYS="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            log|view|search|export|clean)
                COMMAND="$1"
                if [ "$COMMAND" = "log" ]; then
                    ACTION="$2"
                    DETAILS="$3"
                    shift 3
                elif [ "$COMMAND" = "search" ]; then
                    PATTERN="$2"
                    shift 2
                else
                    shift
                fi
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

# Log action
log_action() {
    local action="${1}"
    local details="${2}"
    local timestamp
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local user
    user="${USER:-$(whoami)}"
    local hostname
    hostname=$(hostname 2>/dev/null || echo "unknown")
    
    # Create audit log entry
    local entry
    entry="${timestamp}|${user}@${hostname}|${action}|${details}"
    
    # Append to audit log
    echo "${entry}" >> "${AUDIT_LOG}"
    
    log_info "Audit log entry created: ${action}"
}

# View audit log
view_log() {
    if [ ! -f "${AUDIT_LOG}" ]; then
        log_warn "Audit log file not found: ${AUDIT_LOG}"
        return 1
    fi
    
    log_info "Audit Log (${AUDIT_LOG}):"
    echo ""
    printf "%-20s %-30s %-20s %s\n" "Timestamp" "User@Host" "Action" "Details"
    echo "--------------------------------------------------------------------------------"
    
    while IFS='|' read -r timestamp user_host action details; do
        printf "%-20s %-30s %-20s %s\n" "${timestamp}" "${user_host}" "${action}" "${details}"
    done < "${AUDIT_LOG}"
}

# Search audit log
search_log() {
    local pattern="${1}"
    
    if [ ! -f "${AUDIT_LOG}" ]; then
        log_warn "Audit log file not found: ${AUDIT_LOG}"
        return 1
    fi
    
    log_info "Searching for: ${pattern}"
    echo ""
    
    grep -i "${pattern}" "${AUDIT_LOG}" || echo "No matches found"
}

# Export audit log
export_log() {
    local export_file="${CLOUD_DIR}/audit_export_$(date +%Y%m%d_%H%M%S).csv"
    
    if [ ! -f "${AUDIT_LOG}" ]; then
        log_warn "Audit log file not found: ${AUDIT_LOG}"
        return 1
    fi
    
    log_info "Exporting audit log to: ${export_file}"
    
    # Convert to CSV
    echo "Timestamp,User@Host,Action,Details" > "${export_file}"
    
    while IFS='|' read -r timestamp user_host action details; do
        echo "${timestamp},${user_host},${action},\"${details}\"" >> "${export_file}"
    done < "${AUDIT_LOG}"
    
    log_info "Export completed: ${export_file}"
}

# Clean old entries
clean_log() {
    local retention_days="${1}"
    
    if [ ! -f "${AUDIT_LOG}" ]; then
        log_warn "Audit log file not found: ${AUDIT_LOG}"
        return 1
    fi
    
    log_info "Cleaning audit log (retention: ${retention_days} days)..."
    
    # Create backup
    cp "${AUDIT_LOG}" "${AUDIT_LOG}.backup"
    
    # Filter entries within retention period
    local cutoff_date
    cutoff_date=$(date -u -d "${retention_days} days ago" +%Y-%m-%d)
    
    awk -v cutoff="${cutoff_date}" -F'|' '
        $1 >= cutoff {print}
    ' "${AUDIT_LOG}" > "${AUDIT_LOG}.tmp"
    
    mv "${AUDIT_LOG}.tmp" "${AUDIT_LOG}"
    
    log_info "Audit log cleaned"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    # Ensure audit log directory exists
    mkdir -p "$(dirname "${AUDIT_LOG}")"
    
    case "${COMMAND}" in
        log)
            if [ -z "${ACTION}" ] || [ -z "${DETAILS}" ]; then
                error_exit 1 "Action and details are required"
            fi
            log_action "${ACTION}" "${DETAILS}"
            ;;
        view)
            view_log
            ;;
        search)
            if [ -z "${PATTERN}" ]; then
                error_exit 1 "Search pattern is required"
            fi
            search_log "${PATTERN}"
            ;;
        export)
            export_log
            ;;
        clean)
            clean_log "${AUDIT_RETENTION_DAYS}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


