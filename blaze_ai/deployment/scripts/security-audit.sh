#!/bin/bash

# Security Audit Script for Blaze AI Production
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="/var/log/blaze-ai"
AUDIT_DIR="/var/log/blaze-ai/security"
ALERT_EMAIL="security@blazeai.com"
CRITICAL_ISSUES=0
HIGH_ISSUES=0
MEDIUM_ISSUES=0
LOW_ISSUES=0

# Security thresholds
SSL_EXPIRY_WARNING=30
FIREWALL_PORTS=(22 80 443 8000 8001 8002)
UNAUTHORIZED_PORTS=(21 23 25 110 143 993 995)

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((MEDIUM_ISSUES++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((HIGH_ISSUES++))
}

log_critical() {
    echo -e "${RED}[CRITICAL]${NC} $1"
    ((CRITICAL_ISSUES++))
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

check_prerequisites() {
    log_info "Checking security audit prerequisites..."
    
    # Check if required tools are installed
    if ! command -v nmap &> /dev/null; then
        log_warn "nmap not installed, installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y nmap
        elif command -v yum &> /dev/null; then
            sudo yum install -y nmap
        elif command -v brew &> /dev/null; then
            brew install nmap
        else
            log_error "Cannot install nmap automatically"
            exit 1
        fi
    fi
    
    if ! command -v openssl &> /dev/null; then
        log_error "openssl is not installed"
        exit 1
    fi
    
    if ! command -v netstat &> /dev/null; then
        log_error "netstat is not installed"
        exit 1
    fi
    
    # Create audit directories
    mkdir -p $AUDIT_DIR
    
    log_info "Prerequisites check passed"
}

audit_ssl_certificates() {
    log_info "Auditing SSL certificates..."
    
    # Check main SSL certificate
    if [ -f "deployment/nginx/ssl/cert.pem" ]; then
        log_info "Checking main SSL certificate..."
        
        # Get certificate details
        CERT_SUBJECT=$(openssl x509 -in deployment/nginx/ssl/cert.pem -subject -noout | cut -d= -f2-)
        CERT_ISSUER=$(openssl x509 -in deployment/nginx/ssl/cert.pem -issuer -noout | cut -d= -f2-)
        CERT_EXPIRY=$(openssl x509 -in deployment/nginx/ssl/cert.pem -enddate -noout | cut -d= -f2)
        CERT_DATE=$(date -d "$CERT_EXPIRY" +%s)
        CURRENT_DATE=$(date +%s)
        DAYS_LEFT=$(( (CERT_DATE - CURRENT_DATE) / 86400 ))
        
        log_info "Certificate Details:"
        log_info "- Subject: $CERT_SUBJECT"
        log_info "- Issuer: $CERT_ISSUER"
        log_info "- Expires: $CERT_EXPIRY"
        log_info "- Days until expiry: $DAYS_LEFT"
        
        # Check expiry
        if [ $DAYS_LEFT -lt 0 ]; then
            log_critical "SSL certificate has EXPIRED!"
        elif [ $DAYS_LEFT -lt 7 ]; then
            log_critical "SSL certificate expires in $DAYS_LEFT days!"
        elif [ $DAYS_LEFT -lt $SSL_EXPIRY_WARNING ]; then
            log_warn "SSL certificate expires in $DAYS_LEFT days"
        fi
        
        # Check certificate strength
        CERT_BITS=$(openssl x509 -in deployment/nginx/ssl/cert.pem -text -noout | grep "Public Key Algorithm" | awk '{print $NF}')
        if [[ "$CERT_BITS" == *"2048"* ]] || [[ "$CERT_BITS" == *"4096"* ]]; then
            log_info "Certificate strength: $CERT_BITS (Good)"
        else
            log_warn "Certificate strength: $CERT_BITS (Consider upgrading)"
        fi
        
        # Check for weak ciphers
        SSL_CIPHERS=$(openssl x509 -in deployment/nginx/ssl/cert.pem -text -noout | grep -A 10 "X509v3 Extended Key Usage")
        if echo "$SSL_CIPHERS" | grep -q "TLS Web Server Authentication"; then
            log_info "Certificate usage: TLS Web Server Authentication (Good)"
        else
            log_warn "Certificate usage: Check extended key usage"
        fi
    else
        log_error "Main SSL certificate not found"
    fi
    
    # Check for other certificates
    find /etc/ssl/certs /etc/pki -name "*.pem" -o -name "*.crt" 2>/dev/null | while read cert; do
        if [ -f "$cert" ]; then
            log_debug "Found additional certificate: $cert"
        fi
    done
}

audit_firewall_configuration() {
    log_info "Auditing firewall configuration..."
    
    # Check UFW status
    if command -v ufw &> /dev/null; then
        UFW_STATUS=$(ufw status | grep "Status:" | awk '{print $2}')
        
        if [ "$UFW_STATUS" = "active" ]; then
            log_info "UFW firewall: ACTIVE"
            
            # Check UFW rules
            UFW_RULES=$(ufw status numbered | grep -E "^\[[0-9]+\]")
            log_info "UFW Rules:"
            echo "$UFW_RULES" | while read rule; do
                log_info "  $rule"
            done
        else
            log_critical "UFW firewall: INACTIVE"
        fi
    else
        log_warn "UFW not installed"
    fi
    
    # Check iptables
    if command -v iptables &> /dev/null; then
        IPTABLES_RULES=$(iptables -L -n | grep -E "(ACCEPT|DROP|REJECT)")
        if [ -n "$IPTABLES_RULES" ]; then
            log_info "iptables rules found"
        else
            log_warn "No iptables rules found"
        fi
    fi
    
    # Check for unauthorized open ports
    log_info "Checking for unauthorized open ports..."
    for port in "${UNAUTHORIZED_PORTS[@]}"; do
        if netstat -tuln | grep ":$port " > /dev/null; then
            log_error "Unauthorized port $port is open"
        fi
    done
    
    # Check required ports
    log_info "Checking required ports..."
    for port in "${FIREWALL_PORTS[@]}"; do
        if netstat -tuln | grep ":$port " > /dev/null; then
            log_info "Required port $port: OPEN"
        else
            log_warn "Required port $port: CLOSED"
        fi
    done
}

audit_network_security() {
    log_info "Auditing network security..."
    
    # Check listening ports
    LISTENING_PORTS=$(netstat -tuln | grep LISTEN | awk '{print $4}' | cut -d: -f2 | sort -n | uniq)
    
    log_info "Currently listening ports:"
    echo "$LISTENING_PORTS" | while read port; do
        if [ -n "$port" ]; then
            SERVICE=$(grep ":$port " /etc/services 2>/dev/null | head -1 | awk '{print $1}' || echo "Unknown")
            log_info "  Port $port: $SERVICE"
        fi
    done
    
    # Check for suspicious connections
    log_info "Checking for suspicious connections..."
    ESTABLISHED_CONNS=$(netstat -tuln | grep ESTABLISHED | wc -l)
    log_info "Established connections: $ESTABLISHED_CONNS"
    
    # Check for listening on all interfaces
    ALL_INTERFACE_PORTS=$(netstat -tuln | grep "0.0.0.0:" | awk '{print $4}' | cut -d: -f2)
    if [ -n "$ALL_INTERFACE_PORTS" ]; then
        log_warn "Services listening on all interfaces:"
        echo "$ALL_INTERFACE_PORTS" | while read port; do
            log_warn "  Port $port"
        done
    fi
}

audit_user_accounts() {
    log_info "Auditing user accounts..."
    
    # Check for users with UID 0 (root)
    ROOT_USERS=$(awk -F: '$3 == 0 {print $1}' /etc/passwd)
    ROOT_COUNT=$(echo "$ROOT_USERS" | wc -l)
    
    if [ $ROOT_COUNT -eq 1 ] && [ "$ROOT_USERS" = "root" ]; then
        log_info "Root user configuration: OK (only root has UID 0)"
    else
        log_critical "Multiple users with UID 0 found: $ROOT_USERS"
    fi
    
    # Check for users without passwords
    NO_PASSWORD_USERS=$(awk -F: '$2 == "" {print $1}' /etc/shadow 2>/dev/null || echo "")
    if [ -n "$NO_PASSWORD_USERS" ]; then
        log_critical "Users without passwords: $NO_PASSWORD_USERS"
    else
        log_info "Password policy: OK (all users have passwords)"
    fi
    
    # Check for users with shell access
    SHELL_USERS=$(awk -F: '$7 ~ /\/bin\/(bash|sh|zsh|fish)/ {print $1}' /etc/passwd)
    SHELL_COUNT=$(echo "$SHELL_USERS" | wc -l)
    log_info "Users with shell access: $SHELL_COUNT"
    
    # Check for service accounts
    SERVICE_ACCOUNTS=$(awk -F: '$3 < 1000 && $3 > 0 {print $1}' /etc/passwd)
    log_info "Service accounts found:"
    echo "$SERVICE_ACCOUNTS" | while read account; do
        if [ -n "$account" ]; then
            log_info "  $account"
        fi
    done
}

audit_file_permissions() {
    log_info "Auditing file permissions..."
    
    # Check critical file permissions
    CRITICAL_FILES=(
        "/etc/passwd:644"
        "/etc/shadow:600"
        "/etc/group:644"
        "/etc/gshadow:600"
        "/etc/ssh/sshd_config:600"
    )
    
    for file_perm in "${CRITICAL_FILES[@]}"; do
        file=$(echo $file_perm | cut -d: -f1)
        expected_perm=$(echo $file_perm | cut -d: -f2)
        
        if [ -f "$file" ]; then
            actual_perm=$(stat -c %a "$file")
            owner=$(stat -c %U "$file")
            
            if [ "$actual_perm" = "$expected_perm" ]; then
                log_info "File $file: OK (permissions: $actual_perm, owner: $owner)"
            else
                log_error "File $file: WRONG PERMISSIONS (expected: $expected_perm, actual: $actual_perm)"
            fi
            
            if [ "$owner" != "root" ]; then
                log_warn "File $file: WRONG OWNER (expected: root, actual: $owner)"
            fi
        else
            log_warn "File $file: NOT FOUND"
        fi
    done
    
    # Check application file permissions
    if [ -d "deployment" ]; then
        log_info "Checking application file permissions..."
        
        # Check for world-writable files
        WORLD_WRITABLE=$(find deployment -type f -perm -002 2>/dev/null | head -10)
        if [ -n "$WORLD_WRITABLE" ]; then
            log_warn "World-writable files found:"
            echo "$WORLD_WRITABLE" | while read file; do
                log_warn "  $file"
            done
        else
            log_info "No world-writable files found"
        fi
        
        # Check for files owned by root
        ROOT_OWNED=$(find deployment -user root 2>/dev/null | head -10)
        if [ -n "$ROOT_OWNED" ]; then
            log_warn "Files owned by root:"
            echo "$ROOT_OWNED" | while read file; do
                log_warn "  $file"
            done
        fi
    fi
}

audit_ssh_configuration() {
    log_info "Auditing SSH configuration..."
    
    SSH_CONFIG="/etc/ssh/sshd_config"
    
    if [ -f "$SSH_CONFIG" ]; then
        # Check SSH protocol version
        SSH_PROTOCOL=$(grep "^Protocol" $SSH_CONFIG | awk '{print $2}')
        if [ "$SSH_PROTOCOL" = "2" ]; then
            log_info "SSH Protocol: 2 (Good)"
        else
            log_critical "SSH Protocol: $SSH_PROTOCOL (Should be 2)"
        fi
        
        # Check root login
        ROOT_LOGIN=$(grep "^PermitRootLogin" $SSH_CONFIG | awk '{print $2}')
        if [ "$ROOT_LOGIN" = "no" ]; then
            log_info "Root login: DISABLED (Good)"
        else
            log_critical "Root login: ENABLED (Should be disabled)"
        fi
        
        # Check password authentication
        PASSWORD_AUTH=$(grep "^PasswordAuthentication" $SSH_CONFIG | awk '{print $2}')
        if [ "$PASSWORD_AUTH" = "no" ]; then
            log_info "Password authentication: DISABLED (Good)"
        else
            log_warn "Password authentication: ENABLED (Consider disabling)"
        fi
        
        # Check key-based authentication
        KEY_AUTH=$(grep "^PubkeyAuthentication" $SSH_CONFIG | awk '{print $2}')
        if [ "$KEY_AUTH" = "yes" ]; then
            log_info "Key-based authentication: ENABLED (Good)"
        else
            log_warn "Key-based authentication: DISABLED (Consider enabling)"
        fi
        
        # Check SSH service status
        if systemctl is-active --quiet sshd; then
            log_info "SSH service: ACTIVE"
        else
            log_warn "SSH service: INACTIVE"
        fi
    else
        log_warn "SSH configuration file not found"
    fi
}

audit_docker_security() {
    log_info "Auditing Docker security..."
    
    if command -v docker &> /dev/null; then
        # Check Docker daemon configuration
        DOCKER_DAEMON_CONFIG="/etc/docker/daemon.json"
        if [ -f "$DOCKER_DAEMON_CONFIG" ]; then
            log_info "Docker daemon configuration found"
            
            # Check for security options
            if grep -q "userland-proxy" "$DOCKER_DAEMON_CONFIG"; then
                log_info "Userland proxy: configured"
            else
                log_warn "Userland proxy: not configured"
            fi
        else
            log_warn "Docker daemon configuration not found"
        fi
        
        # Check running containers
        RUNNING_CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}")
        if [ -n "$RUNNING_CONTAINERS" ]; then
            log_info "Running containers:"
            echo "$RUNNING_CONTAINERS" | while read container; do
                if [ -n "$container" ]; then
                    log_info "  $container"
                fi
            done
        else
            log_info "No running containers"
        fi
        
        # Check for privileged containers
        PRIVILEGED_CONTAINERS=$(docker ps --format "{{.Names}}" --filter "label=privileged=true" 2>/dev/null || echo "")
        if [ -n "$PRIVILEGED_CONTAINERS" ]; then
            log_warn "Privileged containers found: $PRIVILEGED_CONTAINERS"
        else
            log_info "No privileged containers found"
        fi
    else
        log_info "Docker not installed"
    fi
}

audit_application_security() {
    log_info "Auditing application security..."
    
    # Check for environment variables
    if [ -f ".env" ]; then
        log_info "Environment file found"
        
        # Check for sensitive data in .env
        SENSITIVE_VARS=$(grep -E "(PASSWORD|SECRET|KEY|TOKEN)" .env | grep -v "^#" || echo "")
        if [ -n "$SENSITIVE_VARS" ]; then
            log_warn "Sensitive environment variables found:"
            echo "$SENSITIVE_VARS" | while read var; do
                log_warn "  $var"
            done
        fi
    else
        log_warn "Environment file not found"
    fi
    
    # Check for hardcoded secrets in code
    log_info "Checking for hardcoded secrets..."
    HARDCODED_SECRETS=$(grep -r -E "(password|secret|key|token)" . --include="*.py" --include="*.js" --include="*.sh" 2>/dev/null | grep -v "password=" | grep -v "secret=" | head -10 || echo "")
    if [ -n "$HARDCODED_SECRETS" ]; then
        log_warn "Potential hardcoded secrets found:"
        echo "$HARDCODED_SECRETS" | while read secret; do
            log_warn "  $secret"
        done
    else
        log_info "No obvious hardcoded secrets found"
    fi
    
    # Check for exposed ports in configuration
    log_info "Checking for exposed ports in configuration..."
    EXPOSED_PORTS=$(grep -r -E "port.*[0-9]{4}" . --include="*.yml" --include="*.yaml" --include="*.py" 2>/dev/null | head -10 || echo "")
    if [ -n "$EXPOSED_PORTS" ]; then
        log_info "Port configurations found:"
        echo "$EXPOSED_PORTS" | while read port; do
            log_info "  $port"
        done
    fi
}

generate_security_report() {
    log_info "Generating security audit report..."
    
    REPORT_FILE="$AUDIT_DIR/security_audit_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > $REPORT_FILE << EOF
Blaze AI Security Audit Report
==============================
Generated: $(date)
Hostname: $(hostname)
Auditor: $(whoami)

Security Issues Summary:
- Critical: $CRITICAL_ISSUES
- High: $HIGH_ISSUES
- Medium: $MEDIUM_ISSUES
- Low: $LOW_ISSUES

Overall Security Score: $(if [ $CRITICAL_ISSUES -eq 0 ] && [ $HIGH_ISSUES -eq 0 ]; then echo "GOOD"; elif [ $CRITICAL_ISSUES -eq 0 ]; then echo "FAIR"; else echo "POOR"; fi)

Detailed Findings:
==================

SSL Certificate Security:
- Main certificate: $(if [ -f "deployment/nginx/ssl/cert.pem" ]; then echo "Found"; else echo "Not Found"; fi)
- Expiry warning: ${SSL_EXPIRY_WARNING} days
- Days until expiry: ${DAYS_LEFT:-"Unknown"}

Firewall Configuration:
- UFW status: ${UFW_STATUS:-"Unknown"}
- Required ports: ${FIREWALL_PORTS[*]}
- Unauthorized ports: ${UNAUTHORIZED_PORTS[*]}

Network Security:
- Listening ports: $(netstat -tuln | grep LISTEN | wc -l)
- Established connections: ${ESTABLISHED_CONNS:-"Unknown"}

User Account Security:
- Root users: ${ROOT_COUNT:-"Unknown"}
- Users without passwords: $(if [ -n "$NO_PASSWORD_USERS" ]; then echo "Yes"; else echo "No"; fi)
- Shell access users: ${SHELL_COUNT:-"Unknown"}

File Permissions:
- Critical files checked: ${#CRITICAL_FILES[@]}
- World-writable files: $(if [ -n "$WORLD_WRITABLE" ]; then echo "Found"; else echo "None"; fi)

SSH Configuration:
- Protocol version: ${SSH_PROTOCOL:-"Unknown"}
- Root login: ${ROOT_LOGIN:-"Unknown"}
- Password auth: ${PASSWORD_AUTH:-"Unknown"}

Docker Security:
- Docker installed: $(if command -v docker &> /dev/null; then echo "Yes"; else echo "No"; fi)
- Privileged containers: $(if [ -n "$PRIVILEGED_CONTAINERS" ]; then echo "Found"; else echo "None"; fi)

Application Security:
- Environment file: $(if [ -f ".env" ]; then echo "Found"; else echo "Not Found"; fi)
- Hardcoded secrets: $(if [ -n "$HARDCODED_SECRETS" ]; then echo "Found"; else echo "None"; fi)

Recommendations:
================
$(if [ $CRITICAL_ISSUES -gt 0 ]; then echo "- IMMEDIATE ACTION REQUIRED: Fix critical security issues"; fi)
$(if [ $HIGH_ISSUES -gt 0 ]; then echo "- HIGH PRIORITY: Address high-risk security vulnerabilities"; fi)
$(if [ $MEDIUM_ISSUES -gt 0 ]; then echo "- MEDIUM PRIORITY: Review and fix medium-risk issues"; fi)
$(if [ $LOW_ISSUES -gt 0 ]; then echo "- LOW PRIORITY: Consider addressing low-risk issues"; fi)

$(if [ "$UFW_STATUS" != "active" ]; then echo "- Enable and configure UFW firewall"; fi)
$(if [ "$ROOT_LOGIN" != "no" ]; then echo "- Disable root SSH login"; fi)
$(if [ "$PASSWORD_AUTH" != "no" ]; then echo "- Disable password authentication for SSH"; fi)
$(if [ $DAYS_LEFT -lt $SSL_EXPIRY_WARNING ]; then echo "- Renew SSL certificate soon"; fi)
$(if [ -n "$WORLD_WRITABLE" ]; then echo "- Review and fix world-writable file permissions"; fi)

Next Steps:
===========
1. Address critical and high-priority issues immediately
2. Review medium and low-priority issues within 30 days
3. Schedule regular security audits (monthly recommended)
4. Implement security monitoring and alerting
5. Conduct penetration testing annually

Report generated by: Blaze AI Security Audit Script v1.0
EOF
    
    log_info "Security audit report generated: $REPORT_FILE"
}

send_security_alerts() {
    log_info "Checking if security alerts need to be sent..."
    
    if [ $CRITICAL_ISSUES -gt 0 ] || [ $HIGH_ISSUES -gt 5 ]; then
        log_critical "Sending security alert email..."
        
        ALERT_SUBJECT="SECURITY ALERT - Critical Issues Detected on $(hostname)"
        ALERT_BODY="Critical security issues detected during audit:
        
Critical Issues: $CRITICAL_ISSUES
High Issues: $HIGH_ISSUES
Medium Issues: $MEDIUM_ISSUES
Low Issues: $LOW_ISSUES

Please review the security audit report immediately.
Report location: $REPORT_FILE

This is an automated security alert from Blaze AI."
        
        echo -e "$ALERT_BODY" | mail -s "$ALERT_SUBJECT" $ALERT_EMAIL
        log_info "Security alert sent to $ALERT_EMAIL"
    else
        log_info "No security alerts needed"
    fi
}

# Main security audit logic
main() {
    log_info "Starting Blaze AI security audit..."
    
    check_prerequisites
    
    # Perform security audits
    audit_ssl_certificates
    audit_firewall_configuration
    audit_network_security
    audit_user_accounts
    audit_file_permissions
    audit_ssh_configuration
    audit_docker_security
    audit_application_security
    
    # Generate report and send alerts
    generate_security_report
    send_security_alerts
    
    # Summary
    log_info "Security audit completed!"
    log_info "Issues found: Critical=$CRITICAL_ISSUES, High=$HIGH_ISSUES, Medium=$MEDIUM_ISSUES, Low=$LOW_ISSUES"
    
    if [ $CRITICAL_ISSUES -gt 0 ]; then
        log_critical "CRITICAL SECURITY ISSUES DETECTED - IMMEDIATE ACTION REQUIRED!"
        exit 1
    elif [ $HIGH_ISSUES -gt 0 ]; then
        log_error "High security issues detected - review and fix soon"
    else
        log_info "Security audit passed - no critical issues found"
    fi
}

# Run main function
main "$@"
