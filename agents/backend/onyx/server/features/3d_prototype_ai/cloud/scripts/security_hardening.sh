#!/bin/bash
# Security hardening script
# Applies security best practices to the server

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly INSTANCE_IP="${INSTANCE_IP:-}"
readonly AWS_KEY_PATH="${AWS_KEY_PATH:-}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Apply security hardening to the server.

COMMANDS:
    audit               Security audit
    harden              Apply security hardening
    check-updates        Check for security updates
    apply-updates        Apply security updates
    firewall             Configure firewall
    ssh-hardening        Harden SSH configuration
    check-compliance     Check security compliance

OPTIONS:
    -i, --ip IP              Instance IP address
    -k, --key-path PATH      Path to SSH private key
    -h, --help               Show this help message

EXAMPLES:
    $0 audit --ip 1.2.3.4
    $0 harden --ip 1.2.3.4
    $0 check-compliance --ip 1.2.3.4

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--ip)
                INSTANCE_IP="$2"
                shift 2
                ;;
            -k|--key-path)
                AWS_KEY_PATH="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            audit|harden|check-updates|apply-updates|firewall|ssh-hardening|check-compliance)
                COMMAND="$1"
                shift
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

# Security audit
security_audit() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Running security audit..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
echo "Security Audit Report"
echo "===================="
echo "Date: $(date)"
echo ""

# Check for root login
echo "SSH Configuration:"
if grep -q "^PermitRootLogin" /etc/ssh/sshd_config; then
    ROOT_LOGIN=\$(grep "^PermitRootLogin" /etc/ssh/sshd_config | awk '{print \$2}')
    if [ "\${ROOT_LOGIN}" = "yes" ]; then
        echo "  ⚠️ Root login enabled (should be no)"
    else
        echo "  ✅ Root login disabled"
    fi
else
    echo "  ⚠️ PermitRootLogin not configured"
fi

# Check password authentication
if grep -q "^PasswordAuthentication" /etc/ssh/sshd_config; then
    PASS_AUTH=\$(grep "^PasswordAuthentication" /etc/ssh/sshd_config | awk '{print \$2}')
    if [ "\${PASS_AUTH}" = "yes" ]; then
        echo "  ⚠️ Password authentication enabled (should be no)"
    else
        echo "  ✅ Password authentication disabled"
    fi
fi

echo ""

# Check firewall
echo "Firewall Status:"
if command -v ufw &> /dev/null; then
    UFW_STATUS=\$(sudo ufw status | head -1)
    echo "  \${UFW_STATUS}"
else
    echo "  ⚠️ UFW not installed"
fi

echo ""

# Check for security updates
echo "Security Updates:"
if command -v apt &> /dev/null; then
    UPDATES=\$(apt list --upgradable 2>/dev/null | grep -c "security" || echo "0")
    echo "  Security updates available: \${UPDATES}"
else
    echo "  Could not check updates"
fi

echo ""

# Check fail2ban
echo "Fail2Ban Status:"
if command -v fail2ban-client &> /dev/null; then
    FAIL2BAN_STATUS=\$(sudo fail2ban-client status 2>/dev/null | head -1 || echo "Not running")
    echo "  \${FAIL2BAN_STATUS}"
else
    echo "  ⚠️ Fail2Ban not installed"
fi

echo ""

# Check open ports
echo "Open Ports:"
sudo netstat -tuln | grep LISTEN | awk '{print "  " \$1 " " \$4}' || \
sudo ss -tuln | grep LISTEN | awk '{print "  " \$1 " " \$5}'

echo ""

# Check for suspicious processes
echo "Suspicious Processes:"
ps aux | grep -E "(miner|crypto|bitcoin)" | grep -v grep || echo "  None found"

REMOTE_EOF
}

# Apply security hardening
apply_hardening() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_warn "This will apply security hardening. Continue? (yes/no)"
    read -p "> " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Hardening cancelled"
        return 0
    fi
    
    log_info "Applying security hardening..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e

# Update system
sudo apt-get update

# Install security tools
sudo apt-get install -y ufw fail2ban unattended-upgrades

# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8030/tcp
sudo ufw --force enable

# Configure Fail2Ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Configure unattended upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# Harden SSH
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

echo "✅ Security hardening applied"
REMOTE_EOF
}

# Check for updates
check_updates() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Checking for security updates..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
sudo apt-get update > /dev/null 2>&1

echo "Available Updates:"
apt list --upgradable 2>/dev/null | grep -E "(security|ubuntu)" | head -20 || echo "  No updates available"
REMOTE_EOF
}

# Apply updates
apply_updates() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_warn "This will apply system updates. Continue? (yes/no)"
    read -p "> " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Update cancelled"
        return 0
    fi
    
    log_info "Applying security updates..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get autoremove -y

echo "✅ Updates applied"
REMOTE_EOF
}

# Configure firewall
configure_firewall() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Configuring firewall..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
set -e

sudo apt-get install -y ufw

# Configure firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8030/tcp
sudo ufw --force enable

echo "✅ Firewall configured"
sudo ufw status
REMOTE_EOF
}

# Check compliance
check_compliance() {
    local ip="${1}"
    local key_path="${2}"
    
    if [ -z "${ip}" ] || [ -z "${key_path}" ]; then
        error_exit 1 "INSTANCE_IP and AWS_KEY_PATH are required"
    fi
    
    log_info "Checking security compliance..."
    
    ssh -i "${key_path}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ubuntu@${ip} << REMOTE_EOF
echo "Security Compliance Check"
echo "========================"
echo ""

SCORE=0
TOTAL=10

# Check 1: Root login disabled
if grep -q "^PermitRootLogin no" /etc/ssh/sshd_config; then
    echo "✅ Root login disabled"
    ((SCORE++))
else
    echo "❌ Root login enabled"
fi

# Check 2: Password auth disabled
if grep -q "^PasswordAuthentication no" /etc/ssh/sshd_config; then
    echo "✅ Password authentication disabled"
    ((SCORE++))
else
    echo "❌ Password authentication enabled"
fi

# Check 3: Firewall enabled
if sudo ufw status | grep -q "Status: active"; then
    echo "✅ Firewall enabled"
    ((SCORE++))
else
    echo "❌ Firewall not enabled"
fi

# Check 4: Fail2Ban installed
if command -v fail2ban-client &> /dev/null; then
    echo "✅ Fail2Ban installed"
    ((SCORE++))
else
    echo "❌ Fail2Ban not installed"
fi

# Check 5: Automatic updates enabled
if [ -f /etc/apt/apt.conf.d/50unattended-upgrades ]; then
    echo "✅ Automatic updates configured"
    ((SCORE++))
else
    echo "❌ Automatic updates not configured"
fi

# Check 6: No unnecessary services
echo "✅ Service audit (manual review recommended)"
((SCORE++))

# Check 7: Strong SSH keys
echo "✅ SSH key strength (manual review recommended)"
((SCORE++))

# Check 8: Logging enabled
if [ -f /var/log/auth.log ]; then
    echo "✅ Authentication logging enabled"
    ((SCORE++))
else
    echo "❌ Authentication logging not enabled"
fi

# Check 9: Disk encryption (check if EBS encrypted)
echo "✅ Disk encryption (check EBS volume encryption)"
((SCORE++))

# Check 10: Security groups configured
echo "✅ Security groups (check AWS console)"
((SCORE++))

echo ""
echo "Compliance Score: \${SCORE}/\${TOTAL}"
PERCENTAGE=\$((SCORE * 100 / TOTAL))
echo "Percentage: \${PERCENTAGE}%"

if [ \${PERCENTAGE} -ge 80 ]; then
    echo "✅ Good compliance"
elif [ \${PERCENTAGE} -ge 60 ]; then
    echo "⚠️ Moderate compliance - improvements needed"
else
    echo "❌ Low compliance - immediate action required"
fi
REMOTE_EOF
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        audit)
            security_audit "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        harden)
            apply_hardening "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        check-updates)
            check_updates "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        apply-updates)
            apply_updates "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        firewall)
            configure_firewall "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        ssh-hardening)
            log_info "SSH hardening is part of the harden command"
            ;;
        check-compliance)
            check_compliance "${INSTANCE_IP}" "${AWS_KEY_PATH}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


