#!/bin/bash

# SSL Certificate Setup Script for Blaze AI Production
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SSL_DIR="deployment/nginx/ssl"
DOMAINS=("api.blazeai.com" "gradio.blazeai.com" "metrics.blazeai.com")
EMAIL="admin@blazeai.com"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking SSL setup prerequisites..."
    
    # Check if certbot is installed
    if ! command -v certbot &> /dev/null; then
        log_warn "certbot is not installed, attempting to install..."
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y certbot
        elif command -v yum &> /dev/null; then
            sudo yum install -y certbot
        elif command -v brew &> /dev/null; then
            brew install certbot
        else
            log_error "Cannot install certbot automatically. Please install it manually."
            exit 1
        fi
    fi
    
    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        log_warn "nginx is not installed, attempting to install..."
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get install -y nginx
        elif command -v yum &> /dev/null; then
            sudo yum install -y nginx
        elif command -v brew &> /dev/null; then
            brew install nginx
        else
            log_error "Cannot install nginx automatically. Please install it manually."
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed"
}

create_ssl_directory() {
    log_info "Creating SSL directory..."
    
    mkdir -p $SSL_DIR
    chmod 700 $SSL_DIR
    
    log_info "SSL directory created: $SSL_DIR"
}

generate_self_signed_cert() {
    log_info "Generating self-signed certificate for testing..."
    
    # Generate private key
    openssl genrsa -out $SSL_DIR/key.pem 2048
    
    # Generate certificate signing request
    openssl req -new -key $SSL_DIR/key.pem -out $SSL_DIR/cert.csr -subj "/C=US/ST=State/L=City/O=BlazeAI/CN=blazeai.com"
    
    # Generate self-signed certificate
    openssl x509 -req -in $SSL_DIR/cert.csr -signkey $SSL_DIR/key.pem -out $SSL_DIR/cert.pem -days 365
    
    # Clean up CSR
    rm $SSL_DIR/cert.csr
    
    log_info "Self-signed certificate generated"
    log_warn "This certificate is for testing only. Use Let's Encrypt for production."
}

generate_letsencrypt_cert() {
    log_info "Generating Let's Encrypt certificate..."
    
    # Stop nginx temporarily
    sudo systemctl stop nginx 2>/dev/null || true
    
    # Generate certificate using certbot
    certbot certonly --standalone \
        --email $EMAIL \
        --agree-tos \
        --no-eff-email \
        --domains $(IFS=,; echo "${DOMAINS[*]}") \
        --cert-path $SSL_DIR/cert.pem \
        --key-path $SSL_DIR/key.pem
    
    # Copy certificates to SSL directory
    sudo cp /etc/letsencrypt/live/blazeai.com/fullchain.pem $SSL_DIR/cert.pem
    sudo cp /etc/letsencrypt/live/blazeai.com/privkey.pem $SSL_DIR/key.pem
    
    # Set proper permissions
    sudo chown $USER:$USER $SSL_DIR/*
    chmod 600 $SSL_DIR/*
    
    log_info "Let's Encrypt certificate generated successfully"
}

setup_auto_renewal() {
    log_info "Setting up automatic certificate renewal..."
    
    # Create renewal script
    cat > $SSL_DIR/renew.sh << 'EOF'
#!/bin/bash
# Certificate renewal script for Blaze AI

SSL_DIR="$(dirname "$0")"
DOMAINS=("api.blazeai.com" "gradio.blazeai.com" "metrics.blazeai.com")
EMAIL="admin@blazeai.com"

# Renew certificate
certbot renew --quiet

# Copy renewed certificates
sudo cp /etc/letsencrypt/live/blazeai.com/fullchain.pem $SSL_DIR/cert.pem
sudo cp /etc/letsencrypt/live/blazeai.com/privkey.pem $SSL_DIR/key.pem

# Set proper permissions
sudo chown $USER:$USER $SSL_DIR/*
chmod 600 $SSL_DIR/*

# Reload nginx
sudo systemctl reload nginx

echo "Certificate renewal completed at $(date)"
EOF
    
    chmod +x $SSL_DIR/renew.sh
    
    # Add to crontab for automatic renewal
    (crontab -l 2>/dev/null; echo "0 12 * * * $SSL_DIR/renew.sh") | crontab -
    
    log_info "Automatic renewal setup completed"
}

verify_certificate() {
    log_info "Verifying certificate..."
    
    if [ -f "$SSL_DIR/cert.pem" ] && [ -f "$SSL_DIR/key.pem" ]; then
        # Check certificate validity
        if openssl x509 -in $SSL_DIR/cert.pem -text -noout | grep -q "Not After"; then
            log_info "Certificate is valid"
            openssl x509 -in $SSL_DIR/cert.pem -text -noout | grep "Not After"
        else
            log_error "Certificate verification failed"
            exit 1
        fi
    else
        log_error "Certificate files not found"
        exit 1
    fi
}

show_ssl_info() {
    log_info "SSL Setup Information:"
    echo "Certificate: $SSL_DIR/cert.pem"
    echo "Private Key: $SSL_DIR/key.pem"
    echo "Domains: ${DOMAINS[*]}"
    echo "Email: $EMAIL"
    echo ""
    echo "Next steps:"
    echo "1. Update your DNS records to point to your server"
    echo "2. Configure nginx with the SSL certificates"
    echo "3. Test the SSL configuration"
}

# Main SSL setup logic
main() {
    log_info "Starting SSL certificate setup for Blaze AI..."
    
    check_prerequisites
    create_ssl_directory
    
    echo "Choose certificate type:"
    echo "1) Self-signed (for testing)"
    echo "2) Let's Encrypt (for production)"
    read -p "Enter your choice (1 or 2): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            generate_self_signed_cert
            ;;
        2)
            generate_letsencrypt_cert
            setup_auto_renewal
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
    
    verify_certificate
    show_ssl_info
    
    log_info "SSL setup completed successfully!"
}

# Run main function
main "$@"
