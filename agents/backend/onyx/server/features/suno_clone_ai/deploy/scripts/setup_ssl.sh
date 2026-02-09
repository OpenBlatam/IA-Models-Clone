#!/bin/bash
# SSL/TLS Setup script for Suno Clone AI
# Sets up Let's Encrypt certificates with Certbot

set -euo pipefail

# Configuration
readonly DOMAIN="${1:-}"
readonly EMAIL="${2:-admin@${DOMAIN}}"
readonly NGINX_CONF="/etc/nginx/nginx.conf"
readonly CERT_DIR="/etc/nginx/ssl"

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root (use sudo)"
        exit 1
    fi
}

# Install Certbot
install_certbot() {
    log_info "Installing Certbot..."
    
    if command -v certbot &> /dev/null; then
        log_info "Certbot already installed"
        return 0
    fi
    
    apt-get update
    apt-get install -y certbot python3-certbot-nginx || {
        log_error "Failed to install Certbot"
        return 1
    }
    
    log_info "Certbot installed"
}

# Generate self-signed certificate (for testing)
generate_self_signed() {
    log_info "Generating self-signed certificate..."
    
    mkdir -p "${CERT_DIR}"
    
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "${CERT_DIR}/key.pem" \
        -out "${CERT_DIR}/cert.pem" \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=${DOMAIN:-localhost}" || {
        log_error "Failed to generate self-signed certificate"
        return 1
    }
    
    chmod 600 "${CERT_DIR}/key.pem"
    chmod 644 "${CERT_DIR}/cert.pem"
    
    log_info "Self-signed certificate generated"
    log_warn "This is for testing only. Use Let's Encrypt for production."
}

# Obtain Let's Encrypt certificate
obtain_certificate() {
    if [ -z "${DOMAIN}" ]; then
        log_error "Domain name required"
        echo "Usage: $0 <domain> [email]"
        exit 1
    fi
    
    log_info "Obtaining Let's Encrypt certificate for ${DOMAIN}..."
    
    # Ensure Nginx is configured and running
    systemctl start nginx || true
    
    # Obtain certificate
    certbot certonly --nginx \
        -d "${DOMAIN}" \
        --email "${EMAIL}" \
        --agree-tos \
        --non-interactive || {
        log_error "Failed to obtain certificate"
        return 1
    }
    
    # Create symlinks
    mkdir -p "${CERT_DIR}"
    ln -sf "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" "${CERT_DIR}/cert.pem"
    ln -sf "/etc/letsencrypt/live/${DOMAIN}/privkey.pem" "${CERT_DIR}/key.pem"
    
    log_info "Certificate obtained and linked"
}

# Setup auto-renewal
setup_renewal() {
    log_info "Setting up certificate auto-renewal..."
    
    # Create renewal script
    cat > /etc/cron.monthly/certbot-renew << 'EOF'
#!/bin/bash
certbot renew --quiet --post-hook "systemctl reload nginx"
EOF
    
    chmod +x /etc/cron.monthly/certbot-renew
    
    log_info "Auto-renewal configured"
}

# Test certificate
test_certificate() {
    log_info "Testing certificate..."
    
    if [ -f "${CERT_DIR}/cert.pem" ] && [ -f "${CERT_DIR}/key.pem" ]; then
        openssl x509 -in "${CERT_DIR}/cert.pem" -text -noout | grep -E "Subject:|Issuer:|Not After"
        log_info "Certificate is valid"
    else
        log_error "Certificate files not found"
        return 1
    fi
}

# Main function
main() {
    check_root
    
    if [ "${1:-}" == "--self-signed" ]; then
        generate_self_signed
        test_certificate
    elif [ -n "${DOMAIN}" ]; then
        install_certbot
        obtain_certificate
        setup_renewal
        test_certificate
        log_info "✅ SSL/TLS setup completed!"
        log_info "Please restart Nginx: systemctl restart nginx"
    else
        echo "Usage:"
        echo "  $0 <domain> [email]          # Let's Encrypt certificate"
        echo "  $0 --self-signed              # Self-signed certificate (testing)"
        exit 1
    fi
}

main "$@"




