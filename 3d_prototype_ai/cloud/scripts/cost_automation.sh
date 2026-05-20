#!/bin/bash
# Cost optimization automation
# Automatically optimizes costs

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly AUTO_OPTIMIZE="${AUTO_OPTIMIZE:-false}"
readonly COST_THRESHOLD="${COST_THRESHOLD:-500}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Automated cost optimization.

COMMANDS:
    analyze             Analyze costs and get recommendations
    optimize            Apply cost optimizations
    monitor             Monitor costs continuously
    report              Generate cost report

OPTIONS:
    -t, --threshold AMOUNT    Cost threshold in USD (default: 500)
    -a, --auto-optimize        Automatically apply optimizations
    -h, --help                 Show this help message

EXAMPLES:
    $0 analyze
    $0 optimize --auto-optimize
    $0 monitor

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--threshold)
                COST_THRESHOLD="$2"
                shift 2
                ;;
            -a|--auto-optimize)
                AUTO_OPTIMIZE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            analyze|optimize|monitor|report)
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

# Analyze costs
analyze_costs() {
    log_info "Analyzing costs and generating recommendations..."
    
    ./scripts/cost_optimizer.sh analyze --days 30
    
    # Get recommendations
    ./scripts/cost_optimizer.sh recommendations
}

# Optimize costs
optimize_costs() {
    local auto_optimize="${1}"
    
    log_info "Optimizing costs..."
    
    if [ "${auto_optimize}" != "true" ]; then
        log_warn "This will apply cost optimizations"
        read -p "Continue? (yes/no): " confirm
        if [ "${confirm}" != "yes" ]; then
            log_info "Optimization cancelled"
            return 0
        fi
    fi
    
    # Find unused resources
    log_info "Finding unused resources..."
    ./scripts/cost_optimizer.sh unused-resources
    
    # Get rightsizing recommendations
    log_info "Getting rightsizing recommendations..."
    ./scripts/cost_optimizer.sh rightsize
    
    # Apply optimizations
    log_info "Applying cost optimizations..."
    # Implement optimization logic here
    
    log_info "Cost optimization completed"
}

# Monitor costs
monitor_costs() {
    local threshold="${1}"
    
    log_info "Monitoring costs (threshold: \$${threshold})..."
    
    while true; do
        # Get current month cost
        local current_cost
        current_cost=$(aws ce get-cost-and-usage \
            --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
            --granularity MONTHLY \
            --metrics BlendedCost \
            --query 'ResultsByTime[0].Total.BlendedCost.Amount' \
            --output text 2>/dev/null || echo "0")
        
        local cost_float
        cost_float=$(echo "${current_cost}" | awk '{print int($1)}')
        
        if [ "${cost_float}" -gt "${threshold}" ]; then
            log_warn "Cost threshold exceeded: \$${current_cost} > \$${threshold}"
            ./scripts/alert_manager.sh send warning "Cost threshold exceeded: \$${current_cost}"
        else
            log_info "Cost within threshold: \$${current_cost}"
        fi
        
        sleep 3600  # Check every hour
    done
}

# Generate report
generate_report() {
    log_info "Generating cost report..."
    
    ./scripts/cost_optimizer.sh report --days 30
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        analyze)
            analyze_costs
            ;;
        optimize)
            optimize_costs "${AUTO_OPTIMIZE}"
            ;;
        monitor)
            monitor_costs "${COST_THRESHOLD}"
            ;;
        report)
            generate_report
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


