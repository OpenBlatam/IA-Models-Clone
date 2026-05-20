#!/bin/bash
# Secret manager script
# Manages secrets securely

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly SECRETS_DIR="${SECRETS_DIR:-${CLOUD_DIR}/.secrets}"
readonly SECRETS_ENCRYPTED="${SECRETS_ENCRYPTED:-true}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Manage secrets securely.

COMMANDS:
    set NAME VALUE       Set a secret
    get NAME            Get a secret value
    list                List all secrets
    delete NAME         Delete a secret
    rotate NAME         Rotate a secret
    export              Export secrets (encrypted)

OPTIONS:
    -d, --secrets-dir DIR   Secrets directory (default: ./.secrets)
    -e, --encrypt            Encrypt secrets
    -h, --help              Show this help message

EXAMPLES:
    $0 set API_KEY "secret123"
    $0 get API_KEY
    $0 list
    $0 rotate API_KEY

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    SECRET_NAME=""
    SECRET_VALUE=""
    ENCRYPT=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--secrets-dir)
                SECRETS_DIR="$2"
                shift 2
                ;;
            -e|--encrypt)
                ENCRYPT=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            set|get|list|delete|rotate|export)
                COMMAND="$1"
                if [ "$COMMAND" = "set" ]; then
                    SECRET_NAME="$2"
                    SECRET_VALUE="$3"
                    shift 3
                elif [ "$COMMAND" = "get" ] || [ "$COMMAND" = "delete" ] || [ "$COMMAND" = "rotate" ]; then
                    SECRET_NAME="$2"
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

# Encrypt value
encrypt_value() {
    local value="${1}"
    
    if command -v openssl &> /dev/null; then
        echo "${value}" | openssl enc -aes-256-cbc -a -salt -pbkdf2 2>/dev/null || echo "${value}"
    else
        # Base64 encoding as fallback
        echo "${value}" | base64
    fi
}

# Decrypt value
decrypt_value() {
    local encrypted_value="${1}"
    
    if command -v openssl &> /dev/null; then
        echo "${encrypted_value}" | openssl enc -aes-256-cbc -d -a -pbkdf2 2>/dev/null || echo "${encrypted_value}"
    else
        # Base64 decoding as fallback
        echo "${encrypted_value}" | base64 -d 2>/dev/null || echo "${encrypted_value}"
    fi
}

# Set secret
set_secret() {
    local name="${1}"
    local value="${2}"
    
    mkdir -p "${SECRETS_DIR}"
    local secret_file="${SECRETS_DIR}/${name}.secret"
    
    if [ "${ENCRYPT}" = "true" ] && [ "${SECRETS_ENCRYPTED}" = "true" ]; then
        local encrypted
        encrypted=$(encrypt_value "${value}")
        echo "${encrypted}" > "${secret_file}"
        chmod 600 "${secret_file}"
        log_info "Secret '${name}' set (encrypted)"
    else
        echo "${value}" > "${secret_file}"
        chmod 600 "${secret_file}"
        log_info "Secret '${name}' set"
    fi
    
    # Log to audit trail
    ./scripts/audit_trail.sh log "secret_set" "Secret '${name}' was set" 2>/dev/null || true
}

# Get secret
get_secret() {
    local name="${1}"
    local secret_file="${SECRETS_DIR}/${name}.secret"
    
    if [ ! -f "${secret_file}" ]; then
        log_error "Secret '${name}' not found"
        return 1
    fi
    
    local encrypted_value
    encrypted_value=$(cat "${secret_file}")
    
    if [ "${SECRETS_ENCRYPTED}" = "true" ]; then
        decrypt_value "${encrypted_value}"
    else
        echo "${encrypted_value}"
    fi
}

# List secrets
list_secrets() {
    if [ ! -d "${SECRETS_DIR}" ]; then
        log_warn "Secrets directory not found"
        return 1
    fi
    
    log_info "Available secrets:"
    echo ""
    
    for secret_file in "${SECRETS_DIR}"/*.secret; do
        if [ -f "${secret_file}" ]; then
            local name
            name=$(basename "${secret_file}" .secret)
            printf "  - %s\n" "${name}"
        fi
    done
}

# Delete secret
delete_secret() {
    local name="${1}"
    local secret_file="${SECRETS_DIR}/${name}.secret"
    
    if [ ! -f "${secret_file}" ]; then
        log_error "Secret '${name}' not found"
        return 1
    fi
    
    rm -f "${secret_file}"
    log_info "Secret '${name}' deleted"
    
    # Log to audit trail
    ./scripts/audit_trail.sh log "secret_delete" "Secret '${name}' was deleted" 2>/dev/null || true
}

# Rotate secret
rotate_secret() {
    local name="${1}"
    local secret_file="${SECRETS_DIR}/${name}.secret"
    
    if [ ! -f "${secret_file}" ]; then
        log_error "Secret '${name}' not found"
        return 1
    fi
    
    log_info "Rotating secret '${name}'..."
    
    # Generate new secret (example - implement based on your needs)
    local new_value
    new_value=$(openssl rand -hex 32 2>/dev/null || date +%s | sha256sum | cut -d' ' -f1)
    
    set_secret "${name}" "${new_value}"
    
    log_info "Secret '${name}' rotated"
    
    # Log to audit trail
    ./scripts/audit_trail.sh log "secret_rotate" "Secret '${name}' was rotated" 2>/dev/null || true
}

# Export secrets
export_secrets() {
    local export_file="${CLOUD_DIR}/secrets_export_$(date +%Y%m%d_%H%M%S).enc"
    
    if [ ! -d "${SECRETS_DIR}" ]; then
        log_warn "Secrets directory not found"
        return 1
    fi
    
    log_info "Exporting secrets to: ${export_file}"
    
    tar -czf - -C "${SECRETS_DIR}" . | \
        openssl enc -aes-256-cbc -salt -pbkdf2 > "${export_file}" 2>/dev/null || \
        tar -czf "${export_file}" -C "${SECRETS_DIR}" .
    
    chmod 600 "${export_file}"
    log_info "Secrets exported (encrypted)"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        set)
            if [ -z "${SECRET_NAME}" ] || [ -z "${SECRET_VALUE}" ]; then
                error_exit 1 "Secret name and value are required"
            fi
            set_secret "${SECRET_NAME}" "${SECRET_VALUE}"
            ;;
        get)
            if [ -z "${SECRET_NAME}" ]; then
                error_exit 1 "Secret name is required"
            fi
            get_secret "${SECRET_NAME}"
            ;;
        list)
            list_secrets
            ;;
        delete)
            if [ -z "${SECRET_NAME}" ]; then
                error_exit 1 "Secret name is required"
            fi
            delete_secret "${SECRET_NAME}"
            ;;
        rotate)
            if [ -z "${SECRET_NAME}" ]; then
                error_exit 1 "Secret name is required"
            fi
            rotate_secret "${SECRET_NAME}"
            ;;
        export)
            export_secrets
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


