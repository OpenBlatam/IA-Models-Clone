#!/bin/bash
# Fix permissions for all scripts
# Ensures all scripts are executable and have correct permissions

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_DIR="${SCRIPT_DIR}/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

# Function to fix script permissions
fix_script_permissions() {
    local file_path="${1}"
    
    if [ -f "${file_path}" ]; then
        # Make executable
        chmod +x "${file_path}" 2>/dev/null && {
            log_info "Fixed permissions: ${file_path}"
            return 0
        } || {
            log_warn "Could not fix permissions: ${file_path} (may need sudo)"
            return 1
        }
    else
        log_warn "File not found: ${file_path}"
        return 1
    fi
}

# Main function
main() {
    log_info "Fixing script permissions..."
    echo ""
    
    local fixed=0
    local failed=0
    
    # Fix scripts in scripts directory
    log_info "Fixing scripts in scripts/ directory..."
    while IFS= read -r -d '' script; do
        if fix_script_permissions "${script}"; then
            ((fixed++))
        else
            ((failed++))
        fi
    done < <(find "${SCRIPT_DIR}" -type f -name "*.sh" -print0)
    
    # Fix user data script
    log_info "Fixing user data script..."
    local user_data="${CLOUD_DIR}/user_data/init.sh"
    if [ -f "${user_data}" ]; then
        if fix_script_permissions "${user_data}"; then
            ((fixed++))
        else
            ((failed++))
        fi
    fi
    
    echo ""
    log_info "=========================================="
    log_info "Fixed: ${fixed} file(s)"
    if [ ${failed} -gt 0 ]; then
        log_warn "Failed: ${failed} file(s) (may need sudo)"
    fi
    log_info "=========================================="
    
    if [ ${failed} -eq 0 ]; then
        log_info "All permissions fixed successfully! ✓"
        exit 0
    else
        log_warn "Some permissions could not be fixed"
        log_info "Try running with sudo if needed"
        exit 1
    fi
}

main "$@"

