#!/bin/bash

###############################################################################
# Script Validator
# Validates all automation scripts for quality, portability, and best practices
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions_enhanced.sh" 2>/dev/null || {
    source "${SCRIPT_DIR}/common_functions.sh" 2>/dev/null || {
        echo "Error: common functions not found" >&2
        exit 1
    }
}

# Configuration
VALIDATION_DIR="${SCRIPT_DIR}"
LOG_FILE="${LOG_FILE:-/var/log/script-validation.log}"
ENABLE_SHELLCHECK="${ENABLE_SHELLCHECK:-true}"
ENABLE_SYNTAX_CHECK="${ENABLE_SYNTAX_CHECK:-true}"
ENABLE_FUNCTION_CHECK="${ENABLE_FUNCTION_CHECK:-true}"

# Results
TOTAL_SCRIPTS=0
PASSED_SCRIPTS=0
FAILED_SCRIPTS=0
WARNINGS=0
ERRORS=0

###############################################################################
# Validation Functions
###############################################################################

check_shebang() {
    local script="$1"
    if ! head -1 "${script}" | grep -qE '^#!/bin/bash'; then
        log_warn "Missing or incorrect shebang in: ${script}"
        return 1
    fi
    return 0
}

check_syntax() {
    local script="$1"
    if bash -n "${script}" 2>&1; then
        return 0
    else
        log_error "Syntax error in: ${script}"
        return 1
    fi
}

check_shellcheck() {
    local script="$1"
    
    if [ "${ENABLE_SHELLCHECK}" != "true" ]; then
        return 0
    fi
    
    if ! check_command "shellcheck"; then
        log_warn "shellcheck not installed, skipping"
        return 0
    fi
    
    if shellcheck "${script}" >/dev/null 2>&1; then
        return 0
    else
        log_warn "shellcheck issues in: ${script}"
        shellcheck "${script}" 2>&1 | head -5
        return 1
    fi
}

check_functions() {
    local script="$1"
    local functions
    functions=$(grep -E '^[a-z_]+\(\)' "${script}" 2>/dev/null | wc -l)
    
    if [ $functions -eq 0 ]; then
        log_warn "No functions defined in: ${script}"
        return 1
    fi
    
    log_debug "Found ${functions} functions in: ${script}"
    return 0
}

check_error_handling() {
    local script="$1"
    local has_trap=false
    local has_set_e=false
    
    if grep -qE '^set -e' "${script}" || grep -qE '^set -euo' "${script}"; then
        has_set_e=true
    fi
    
    if grep -qE '^trap ' "${script}"; then
        has_trap=true
    fi
    
    if [ "${has_set_e}" = "false" ]; then
        log_warn "Missing 'set -e' in: ${script}"
        return 1
    fi
    
    if [ "${has_trap}" = "false" ]; then
        log_warn "Missing 'trap' for error handling in: ${script}"
        return 1
    fi
    
    return 0
}

check_logging() {
    local script="$1"
    if ! grep -qE '(log_info|log_error|echo.*\[)' "${script}"; then
        log_warn "No logging found in: ${script}"
        return 1
    fi
    return 0
}

check_validation() {
    local script="$1"
    if ! grep -qE '(check_|validate_)' "${script}"; then
        log_warn "No input validation found in: ${script}"
        return 1
    fi
    return 0
}

validate_script() {
    local script="$1"
    local script_name
    script_name=$(basename "${script}")
    local issues=0
    
    log_info "Validating: ${script_name}"
    
    TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
    
    # Run all checks
    check_shebang "${script}" || issues=$((issues + 1))
    
    if [ "${ENABLE_SYNTAX_CHECK}" = "true" ]; then
        check_syntax "${script}" || {
            issues=$((issues + 1))
            ERRORS=$((ERRORS + 1))
        }
    fi
    
    check_shellcheck "${script}" || WARNINGS=$((WARNINGS + 1))
    
    if [ "${ENABLE_FUNCTION_CHECK}" = "true" ]; then
        check_functions "${script}" || WARNINGS=$((WARNINGS + 1))
    fi
    
    check_error_handling "${script}" || WARNINGS=$((WARNINGS + 1))
    check_logging "${script}" || WARNINGS=$((WARNINGS + 1))
    check_validation "${script}" || WARNINGS=$((WARNINGS + 1))
    
    if [ $issues -eq 0 ]; then
        log_success "Validation passed: ${script_name}"
        PASSED_SCRIPTS=$((PASSED_SCRIPTS + 1))
        return 0
    else
        log_error "Validation failed: ${script_name} (${issues} issues)"
        FAILED_SCRIPTS=$((FAILED_SCRIPTS + 1))
        return 1
    fi
}

validate_all_scripts() {
    log_section "Script Validation Report"
    
    # Find all shell scripts
    while IFS= read -r -d '' script; do
        # Skip common functions and this validator
        if [[ "${script}" == *"common_functions"* ]] || \
           [[ "${script}" == *"script_validator"* ]]; then
            continue
        fi
        
        validate_script "${script}"
    done < <(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name "*.sh" -print0 2>/dev/null)
    
    # Generate report
    log_section "Validation Summary"
    log_info "Total scripts: ${TOTAL_SCRIPTS}"
    log_info "Passed: ${PASSED_SCRIPTS}"
    log_info "Failed: ${FAILED_SCRIPTS}"
    log_info "Warnings: ${WARNINGS}"
    log_info "Errors: ${ERRORS}"
    
    if [ $FAILED_SCRIPTS -eq 0 ] && [ $ERRORS -eq 0 ]; then
        log_success "All scripts passed validation"
        return 0
    else
        log_error "Validation completed with issues"
        return 1
    fi
}

###############################################################################
# Main Execution
###############################################################################

main() {
    init_logging "${LOG_FILE}"
    log_section "Starting Script Validation"
    
    validate_all_scripts
    
    exit $?
}

# Run main function
main "$@"

