#!/bin/bash

###############################################################################
# Security Audit Script for AI Project Generator
# Performs security checks and hardening recommendations
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh" 2>/dev/null || {
    echo "Error: common_functions.sh not found" >&2
    exit 1
}

# Configuration
LOG_FILE="${LOG_FILE:-/var/log/security-audit.log}"
REPORT_DIR="${REPORT_DIR:-/tmp/security-audit}"
APP_DIR="${APP_DIR:-/opt/ai-project-generator}"

# Security checks
CHECK_SSH="${CHECK_SSH:-true}"
CHECK_FIREWALL="${CHECK_FIREWALL:-true}"
CHECK_PERMISSIONS="${CHECK_PERMISSIONS:-true}"
CHECK_SECRETS="${CHECK_SECRETS:-true}"
CHECK_UPDATES="${CHECK_UPDATES:-true}"

###############################################################################
# Security Check Functions
###############################################################################

check_ssh_security() {
    log_info "Checking SSH security..."
    
    local issues=0
    local ssh_config="/etc/ssh/sshd_config"
    
    if [ ! -f "${ssh_config}" ]; then
        log_warn "SSH config not found"
        return 1
    fi
    
    # Check PasswordAuthentication
    if grep -q "^PasswordAuthentication yes" "${ssh_config}"; then
        log_warn "SSH password authentication is enabled (should be disabled)"
        issues=$((issues + 1))
    fi
    
    # Check PermitRootLogin
    if grep -q "^PermitRootLogin yes" "${ssh_config}"; then
        log_warn "SSH root login is enabled (should be disabled)"
        issues=$((issues + 1))
    fi
    
    # Check for weak ciphers
    if grep -qE "^(Ciphers|MACs).*md5" "${ssh_config}"; then
        log_warn "Weak SSH ciphers detected"
        issues=$((issues + 1))
    fi
    
    if [ $issues -eq 0 ]; then
        log_success "SSH security: OK"
    else
        log_warn "SSH security: ${issues} issues found"
    fi
    
    return $issues
}

check_firewall() {
    log_info "Checking firewall configuration..."
    
    local issues=0
    
    # Check UFW status
    if command -v ufw &> /dev/null; then
        if ! ufw status | grep -q "Status: active"; then
            log_warn "UFW firewall is not active"
            issues=$((issues + 1))
        else
            log_success "UFW firewall is active"
        fi
    else
        log_warn "UFW not installed"
        issues=$((issues + 1))
    fi
    
    # Check iptables if UFW not available
    if [ $issues -gt 0 ] && command -v iptables &> /dev/null; then
        local rule_count
        rule_count=$(iptables -L | grep -v "^Chain" | grep -v "^target" | wc -l)
        if [ $rule_count -lt 3 ]; then
            log_warn "Few iptables rules found"
            issues=$((issues + 1))
        fi
    fi
    
    return $issues
}

check_file_permissions() {
    log_info "Checking file permissions..."
    
    local issues=0
    
    # Check application directory permissions
    if [ -d "${APP_DIR}" ]; then
        local perms
        perms=$(stat -c %a "${APP_DIR}" 2>/dev/null || stat -f "%OLp" "${APP_DIR}" 2>/dev/null)
        if [ "${perms}" != "755" ] && [ "${perms}" != "750" ]; then
            log_warn "Application directory has insecure permissions: ${perms}"
            issues=$((issues + 1))
        fi
    fi
    
    # Check for world-writable files
    local world_writable
    world_writable=$(find "${APP_DIR}" -type f -perm -002 2>/dev/null | wc -l)
    if [ $world_writable -gt 0 ]; then
        log_warn "Found ${world_writable} world-writable files"
        issues=$((issues + 1))
    fi
    
    # Check .env file permissions
    if [ -f "${APP_DIR}/.env" ]; then
        local env_perms
        env_perms=$(stat -c %a "${APP_DIR}/.env" 2>/dev/null || stat -f "%OLp" "${APP_DIR}/.env" 2>/dev/null)
        if [ "${env_perms}" != "600" ] && [ "${env_perms}" != "640" ]; then
            log_warn ".env file has insecure permissions: ${env_perms} (should be 600)"
            issues=$((issues + 1))
        fi
    fi
    
    if [ $issues -eq 0 ]; then
        log_success "File permissions: OK"
    else
        log_warn "File permissions: ${issues} issues found"
    fi
    
    return $issues
}

check_secrets() {
    log_info "Checking for exposed secrets..."
    
    local issues=0
    
    # Check for hardcoded secrets in code
    local secret_patterns=(
        "password.*=.*['\"].*['\"]"
        "secret.*=.*['\"].*['\"]"
        "api_key.*=.*['\"].*['\"]"
        "aws_secret.*=.*['\"].*['\"]"
    )
    
    if [ -d "${APP_DIR}" ]; then
        for pattern in "${secret_patterns[@]}"; do
            local matches
            matches=$(grep -rE "${pattern}" "${APP_DIR}" --exclude-dir=.git \
                --exclude="*.pyc" --exclude="*.log" 2>/dev/null | wc -l)
            if [ $matches -gt 0 ]; then
                log_warn "Potential secrets found with pattern: ${pattern}"
                issues=$((issues + matches))
            fi
        done
    fi
    
    # Check .env file for default values
    if [ -f "${APP_DIR}/.env" ]; then
        if grep -qE "(CHANGE_THIS|default|password|secret)" "${APP_DIR}/.env" | \
            grep -vE "^#" | grep -vE "^$"; then
            log_warn "Potential default/weak secrets in .env file"
            issues=$((issues + 1))
        fi
    fi
    
    if [ $issues -eq 0 ]; then
        log_success "Secrets check: OK"
    else
        log_warn "Secrets check: ${issues} potential issues found"
    fi
    
    return $issues
}

check_system_updates() {
    log_info "Checking system updates..."
    
    local issues=0
    
    if command -v apt &> /dev/null; then
        # Check for available updates
        apt list --upgradable 2>/dev/null | grep -v "^Listing" | wc -l > /tmp/updates_count.txt
        local update_count
        update_count=$(cat /tmp/updates_count.txt)
        
        if [ $update_count -gt 10 ]; then
            log_warn "Many system updates available: ${update_count}"
            issues=$((issues + 1))
        elif [ $update_count -gt 0 ]; then
            log_info "System updates available: ${update_count}"
        else
            log_success "System is up to date"
        fi
        
        rm -f /tmp/updates_count.txt
    fi
    
    return $issues
}

check_docker_security() {
    log_info "Checking Docker security..."
    
    local issues=0
    
    if ! check_command "docker"; then
        return 0
    fi
    
    # Check for containers running as root
    local root_containers
    root_containers=$(docker ps --format "{{.Names}}" | while read name; do
        docker inspect --format='{{.Config.User}}' "${name}" 2>/dev/null | grep -q "^$" && echo "${name}"
    done | wc -l)
    
    if [ $root_containers -gt 0 ]; then
        log_warn "Found ${root_containers} containers running as root"
        issues=$((issues + 1))
    fi
    
    # Check for exposed ports
    local exposed_ports
    exposed_ports=$(docker ps --format "{{.Ports}}" | grep -oE "[0-9]+:[0-9]+" | wc -l)
    if [ $exposed_ports -gt 5 ]; then
        log_warn "Many exposed Docker ports: ${exposed_ports}"
        issues=$((issues + 1))
    fi
    
    if [ $issues -eq 0 ]; then
        log_success "Docker security: OK"
    else
        log_warn "Docker security: ${issues} issues found"
    fi
    
    return $issues
}

generate_security_report() {
    log_info "Generating security audit report..."
    
    mkdir -p "${REPORT_DIR}"
    local report_file="${REPORT_DIR}/security_audit_$(date +%Y%m%d_%H%M%S).txt"
    
    local total_issues=0
    
    # Run all checks
    check_ssh_security && total_issues=$((total_issues + $?)) || total_issues=$((total_issues + $?))
    check_firewall && total_issues=$((total_issues + $?)) || total_issues=$((total_issues + $?))
    check_file_permissions && total_issues=$((total_issues + $?)) || total_issues=$((total_issues + $?))
    check_secrets && total_issues=$((total_issues + $?)) || total_issues=$((total_issues + $?))
    check_system_updates && total_issues=$((total_issues + $?)) || total_issues=$((total_issues + $?))
    check_docker_security && total_issues=$((total_issues + $?)) || total_issues=$((total_issues + $?))
    
    # Generate report
    generate_report "${report_file}" "Security Audit Report" \
        "Generated: $(date -Iseconds)\n" \
        "Hostname: $(hostname)\n" \
        "Total Issues Found: ${total_issues}\n" \
        "\nRecommendations:\n" \
        "- Review and fix identified security issues\n" \
        "- Enable automatic security updates\n" \
        "- Review firewall rules\n" \
        "- Rotate secrets regularly\n" \
        "- Keep system packages updated\n"
    
    log_info "Security audit report: ${report_file}"
    
    # Send metrics
    send_cloudwatch_metric "AIProjectGenerator/Security" "SecurityIssues" \
        "${total_issues}" "Count"
    
    # Send alert if issues found
    if [ $total_issues -gt 0 ]; then
        send_sns_alert "${SNS_TOPIC_ARN:-}" \
            "Security Audit: Issues Found" \
            "Security audit found ${total_issues} issues\nReport: ${report_file}"
    fi
    
    return $total_issues
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_info "=========================================="
    log_info "Starting Security Audit"
    log_info "=========================================="
    
    generate_security_report
    local exit_code=$?
    
    log_info "=========================================="
    if [ $exit_code -eq 0 ]; then
        log_success "Security audit completed - No issues found"
    else
        log_warn "Security audit completed - ${exit_code} issues found"
    fi
    log_info "=========================================="
    
    exit $exit_code
}

# Run main function
main "$@"

