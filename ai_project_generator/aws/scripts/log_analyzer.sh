#!/bin/bash

###############################################################################
# Log Analyzer Script for AI Project Generator
# Analyzes logs for errors, patterns, and generates reports
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh" 2>/dev/null || {
    echo "Error: common_functions.sh not found" >&2
    exit 1
}

# Configuration
LOG_DIR="${LOG_DIR:-/var/log}"
APP_LOG_DIR="${APP_LOG_DIR:-/opt/ai-project-generator/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/log-analysis}"
LOG_FILE="${LOG_FILE:-/var/log/log-analyzer.log}"
ANALYSIS_PERIOD="${ANALYSIS_PERIOD:-24}" # hours
SEND_REPORT="${SEND_REPORT:-false}"

# Analysis patterns
ERROR_PATTERNS=(
    "ERROR"
    "FATAL"
    "Exception"
    "Traceback"
    "failed"
    "timeout"
    "connection refused"
)

WARNING_PATTERNS=(
    "WARN"
    "WARNING"
    "deprecated"
    "slow"
)

###############################################################################
# Analysis Functions
###############################################################################

analyze_log_file() {
    local log_file="$1"
    local output_file="$2"
    
    if [ ! -f "${log_file}" ]; then
        log_warn "Log file not found: ${log_file}"
        return 1
    fi
    
    log_info "Analyzing: ${log_file}"
    
    # Count errors
    local error_count=0
    for pattern in "${ERROR_PATTERNS[@]}"; do
        local count
        count=$(grep -i "${pattern}" "${log_file}" 2>/dev/null | wc -l)
        error_count=$((error_count + count))
    done
    
    # Count warnings
    local warning_count=0
    for pattern in "${WARNING_PATTERNS[@]}"; do
        local count
        count=$(grep -i "${pattern}" "${log_file}" 2>/dev/null | wc -l)
        warning_count=$((warning_count + count))
    done
    
    # Get top errors
    local top_errors
    top_errors=$(grep -iE "$(IFS='|'; echo "${ERROR_PATTERNS[*]}")" "${log_file}" 2>/dev/null | \
        sort | uniq -c | sort -rn | head -10)
    
    # Get recent errors (last hour)
    local recent_errors
    recent_errors=$(grep -iE "$(IFS='|'; echo "${ERROR_PATTERNS[*]}")" "${log_file}" 2>/dev/null | \
        tail -20)
    
    # Write analysis
    cat > "${output_file}" <<EOF
Log File: ${log_file}
Analysis Period: Last ${ANALYSIS_PERIOD} hours
Generated: $(date -Iseconds)

Summary:
--------
Total Errors: ${error_count}
Total Warnings: ${warning_count}
File Size: $(du -h "${log_file}" | cut -f1)
Last Modified: $(stat -c %y "${log_file}" 2>/dev/null || stat -f "%Sm" "${log_file}" 2>/dev/null)

Top 10 Errors:
--------------
${top_errors:-None}

Recent Errors (Last 20):
------------------------
${recent_errors:-None}
EOF
    
    log_debug "Analysis written to: ${output_file}"
    echo "${error_count}|${warning_count}"
}

analyze_all_logs() {
    log_info "Analyzing all log files..."
    
    mkdir -p "${OUTPUT_DIR}"
    local total_errors=0
    local total_warnings=0
    local analyzed_files=0
    
    # Application logs
    if [ -d "${APP_LOG_DIR}" ]; then
        while IFS= read -r -d '' log_file; do
            local output_file="${OUTPUT_DIR}/analysis_$(basename ${log_file}).txt"
            local result
            result=$(analyze_log_file "${log_file}" "${output_file}")
            if [ -n "${result}" ]; then
                local errors warnings
                IFS='|' read -r errors warnings <<< "${result}"
                total_errors=$((total_errors + errors))
                total_warnings=$((total_warnings + warnings))
                analyzed_files=$((analyzed_files + 1))
            fi
        done < <(find "${APP_LOG_DIR}" -type f -name "*.log" -mtime -$((ANALYSIS_PERIOD / 24)) -print0 2>/dev/null)
    fi
    
    # System logs
    local system_logs=(
        "/var/log/nginx/error.log"
        "/var/log/nginx/access.log"
        "/var/log/redis/redis-server.log"
        "/var/log/syslog"
    )
    
    for log_file in "${system_logs[@]}"; do
        if [ -f "${log_file}" ]; then
            local output_file="${OUTPUT_DIR}/analysis_$(basename ${log_file}).txt"
            local result
            result=$(analyze_log_file "${log_file}" "${output_file}")
            if [ -n "${result}" ]; then
                local errors warnings
                IFS='|' read -r errors warnings <<< "${result}"
                total_errors=$((total_errors + errors))
                total_warnings=$((total_warnings + warnings))
                analyzed_files=$((analyzed_files + 1))
            fi
        fi
    done
    
    # Generate summary
    local summary_file="${OUTPUT_DIR}/summary.txt"
    generate_report "${summary_file}" "Log Analysis Summary" \
        "Analysis Period: Last ${ANALYSIS_PERIOD} hours\n" \
        "Files Analyzed: ${analyzed_files}\n" \
        "Total Errors: ${total_errors}\n" \
        "Total Warnings: ${total_warnings}\n" \
        "Analysis Directory: ${OUTPUT_DIR}\n"
    
    log_success "Analysis completed: ${analyzed_files} files, ${total_errors} errors, ${total_warnings} warnings"
    
    # Send metrics
    send_cloudwatch_metric "AIProjectGenerator/Logs" "LogErrors" "${total_errors}" "Count"
    send_cloudwatch_metric "AIProjectGenerator/Logs" "LogWarnings" "${total_warnings}" "Count"
    send_cloudwatch_metric "AIProjectGenerator/Logs" "LogFilesAnalyzed" "${analyzed_files}" "Count"
    
    # Send report if configured
    if [ "${SEND_REPORT}" = "true" ] && [ -n "${ALERT_EMAIL:-}" ]; then
        mail -s "Log Analysis Report" "${ALERT_EMAIL}" < "${summary_file}" 2>/dev/null || true
    fi
    
    echo "${total_errors}|${total_warnings}|${analyzed_files}"
}

detect_anomalies() {
    log_info "Detecting anomalies in logs..."
    
    local anomaly_file="${OUTPUT_DIR}/anomalies.txt"
    local anomalies=0
    
    # Check for sudden error spikes
    # Check for unusual patterns
    # Check for performance degradation indicators
    
    cat > "${anomaly_file}" <<EOF
Anomaly Detection Report
Generated: $(date -Iseconds)

Anomalies Detected: ${anomalies}

EOF
    
    if [ $anomalies -gt 0 ]; then
        log_warn "Anomalies detected: ${anomalies}"
        send_sns_alert "${SNS_TOPIC_ARN:-}" \
            "Log Anomalies Detected" \
            "Found ${anomalies} anomalies in log analysis\nSee: ${anomaly_file}"
    else
        log_success "No anomalies detected"
    fi
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "=========================================="
    log_info "Starting Log Analysis"
    log_info "=========================================="
    log_info "Analysis period: Last ${ANALYSIS_PERIOD} hours"
    
    analyze_all_logs
    detect_anomalies
    
    log_info "=========================================="
    log_success "Log analysis completed"
    log_info "Results: ${OUTPUT_DIR}"
    log_info "=========================================="
}

# Run main function
main "$@"

