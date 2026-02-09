#!/bin/bash
# Advanced Security Hardening Script
# Implements security best practices and compliance checks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_CONFIG_DIR="/opt/ai-project-generator/config/security"

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

# Configure firewall (UFW)
configure_firewall() {
    log_info "Configuring firewall..."
    
    sudo ufw --force reset
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow SSH
    sudo ufw allow 22/tcp
    
    # Allow HTTP/HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    
    # Allow application port
    sudo ufw allow 8020/tcp
    
    # Allow Prometheus (if installed)
    sudo ufw allow 9090/tcp comment 'Prometheus'
    
    # Allow Grafana (if installed)
    sudo ufw allow 3000/tcp comment 'Grafana'
    
    sudo ufw --force enable
    
    log_info "✅ Firewall configured"
}

# Configure fail2ban
configure_fail2ban() {
    log_info "Configuring fail2ban..."
    
    sudo apt-get update
    sudo apt-get install -y fail2ban
    
    sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = root@localhost
sendername = Fail2Ban
action = %(action_)s

[sshd]
enabled = true
port = 22
logpath = %(sshd_log)s
maxretry = 3

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
EOF
    
    sudo systemctl enable fail2ban
    sudo systemctl restart fail2ban
    
    log_info "✅ fail2ban configured"
}

# Configure automatic security updates
configure_auto_updates() {
    log_info "Configuring automatic security updates..."
    
    sudo apt-get install -y unattended-upgrades
    
    sudo tee /etc/apt/apt.conf.d/50unattended-upgrades > /dev/null <<EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}:\${distro_codename}-updates";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF
    
    sudo tee /etc/apt/apt.conf.d/20auto-upgrades > /dev/null <<EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOF
    
    log_info "✅ Automatic security updates configured"
}

# Harden SSH
harden_ssh() {
    log_info "Hardening SSH configuration..."
    
    sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup
    
    sudo tee -a /etc/ssh/sshd_config > /dev/null <<EOF

# Security hardening
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
Protocol 2
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
X11Forwarding no
PermitEmptyPasswords no
AllowUsers ubuntu
EOF
    
    sudo systemctl restart sshd
    
    log_info "✅ SSH hardened"
}

# Configure auditd
configure_auditd() {
    log_info "Configuring auditd..."
    
    sudo apt-get install -y auditd audispd-plugins
    
    sudo tee /etc/audit/rules.d/audit.rules > /dev/null <<EOF
# Delete all rules
-D

# Buffer size
-b 8192

# Failure mode
-f 1

# Make the configuration immutable
-e 2
EOF
    
    sudo systemctl enable auditd
    sudo systemctl start auditd
    
    log_info "✅ auditd configured"
}

# Configure AppArmor
configure_apparmor() {
    log_info "Configuring AppArmor..."
    
    sudo apt-get install -y apparmor apparmor-utils
    
    # Enable AppArmor
    sudo systemctl enable apparmor
    sudo systemctl start apparmor
    
    # Set profiles to enforce mode
    sudo aa-enforce /etc/apparmor.d/*
    
    log_info "✅ AppArmor configured"
}

# Configure log rotation
configure_log_rotation() {
    log_info "Configuring log rotation..."
    
    sudo tee /etc/logrotate.d/ai-project-generator-security > /dev/null <<EOF
/var/log/security-*.log {
    daily
    rotate 90
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root root
    sharedscripts
}
EOF
    
    log_info "✅ Log rotation configured"
}

# Install security tools
install_security_tools() {
    log_info "Installing security tools..."
    
    sudo apt-get update
    sudo apt-get install -y \
        rkhunter \
        chkrootkit \
        lynis \
        aide \
        clamav \
        clamav-daemon
    
    # Update ClamAV database
    sudo freshclam
    
    log_info "✅ Security tools installed"
}

# Run security scan
run_security_scan() {
    log_info "Running security scan..."
    
    local report_file="/tmp/security_scan_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "=== Security Scan Report ==="
        echo "Date: $(date)"
        echo ""
        
        echo "=== rkhunter Scan ==="
        sudo rkhunter --check --skip-keypress --report-warnings-only || true
        
        echo ""
        echo "=== chkrootkit Scan ==="
        sudo chkrootkit -q || true
        
        echo ""
        echo "=== Lynis Audit ==="
        sudo lynis audit system --quick || true
        
        echo ""
        echo "=== ClamAV Scan ==="
        sudo clamscan -r /opt/ai-project-generator --infected --remove --quiet || true
        
    } | tee "$report_file"
    
    log_info "✅ Security scan completed: $report_file"
}

# Configure WAF rules (Nginx ModSecurity)
configure_waf() {
    log_info "Configuring WAF (ModSecurity)..."
    
    sudo apt-get install -y nginx-module-security
    
    # Enable ModSecurity in Nginx
    if [ -f /etc/nginx/modules-enabled/50-mod-http-security.conf ]; then
        log_info "ModSecurity module enabled"
    else
        log_warn "ModSecurity module not found, skipping WAF configuration"
    fi
    
    log_info "✅ WAF configuration attempted"
}

# Main function
main() {
    case "${1:-all}" in
        all)
            configure_firewall
            configure_fail2ban
            configure_auto_updates
            harden_ssh
            configure_auditd
            configure_apparmor
            configure_log_rotation
            install_security_tools
            log_info "✅ Security hardening completed"
            ;;
        firewall)
            configure_firewall
            ;;
        fail2ban)
            configure_fail2ban
            ;;
        ssh)
            harden_ssh
            ;;
        scan)
            run_security_scan
            ;;
        *)
            echo "Usage: $0 {all|firewall|fail2ban|ssh|scan}"
            exit 1
            ;;
    esac
}

main "$@"



