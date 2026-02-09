#!/bin/bash
# Serverless deployment script
# Manages serverless deployments (Lambda, Functions, etc.)

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly FUNCTION_NAME="${FUNCTION_NAME:-3d-prototype-ai}"
readonly RUNTIME="${RUNTIME:-python3.11}"
readonly REGION="${AWS_REGION:-us-east-1}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Serverless deployment management.

COMMANDS:
    deploy              Deploy serverless function
    update              Update function
    invoke              Invoke function
    logs                View function logs
    delete              Delete function
    status              Show function status

OPTIONS:
    -n, --name NAME         Function name (default: 3d-prototype-ai)
    -r, --runtime RUNTIME   Runtime (default: python3.11)
    -R, --region REGION     AWS region (default: us-east-1)
    -h, --help              Show this help message

EXAMPLES:
    $0 deploy
    $0 invoke --name my-function
    $0 logs --name my-function

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--name)
                FUNCTION_NAME="$2"
                shift 2
                ;;
            -r|--runtime)
                RUNTIME="$2"
                shift 2
                ;;
            -R|--region)
                REGION="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            deploy|update|invoke|logs|delete|status)
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

# Deploy serverless function
deploy_function() {
    local function_name="${1}"
    local runtime="${2}"
    local region="${3}"
    
    log_info "Deploying serverless function: ${function_name}"
    
    # Check if SAM is available
    if command -v sam &> /dev/null; then
        log_info "Using AWS SAM for deployment..."
        sam build && sam deploy --region "${region}" || {
            log_error "SAM deployment failed"
            return 1
        }
    elif command -v serverless &> /dev/null; then
        log_info "Using Serverless Framework..."
        serverless deploy --region "${region}" || {
            log_error "Serverless deployment failed"
            return 1
        }
    else
        # Direct Lambda deployment
        log_info "Deploying directly to Lambda..."
        
        # Create deployment package
        local project_root="$(cd "${SCRIPT_DIR}/../.." && pwd)"
        local package_dir="/tmp/lambda-package-$$"
        mkdir -p "${package_dir}"
        
        # Copy application files
        cp -r "${project_root}"/* "${package_dir}/" 2>/dev/null || true
        
        # Create zip package
        cd "${package_dir}"
        zip -r "/tmp/${function_name}.zip" . > /dev/null
        
        # Deploy to Lambda
        aws lambda create-function \
            --function-name "${function_name}" \
            --runtime "${runtime}" \
            --role "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role" \
            --handler "app.handler" \
            --zip-file "fileb:///tmp/${function_name}.zip" \
            --region "${region}" 2>/dev/null || \
        aws lambda update-function-code \
            --function-name "${function_name}" \
            --zip-file "fileb:///tmp/${function_name}.zip" \
            --region "${region}" > /dev/null
        
        rm -rf "${package_dir}" "/tmp/${function_name}.zip"
        
        log_info "Function deployed successfully"
    fi
}

# Update function
update_function() {
    local function_name="${1}"
    local region="${2}"
    
    log_info "Updating function: ${function_name}"
    
    deploy_function "${function_name}" "${RUNTIME}" "${region}"
}

# Invoke function
invoke_function() {
    local function_name="${1}"
    local region="${2}"
    
    log_info "Invoking function: ${function_name}"
    
    aws lambda invoke \
        --function-name "${function_name}" \
        --region "${region}" \
        /tmp/lambda-response.json > /dev/null
    
    cat /tmp/lambda-response.json | jq . 2>/dev/null || cat /tmp/lambda-response.json
    rm -f /tmp/lambda-response.json
}

# View logs
view_logs() {
    local function_name="${1}"
    local region="${2}"
    
    log_info "Viewing logs for function: ${function_name}"
    
    aws logs tail "/aws/lambda/${function_name}" \
        --region "${region}" \
        --follow 2>/dev/null || \
    log_warn "CloudWatch Logs not available or function not found"
}

# Delete function
delete_function() {
    local function_name="${1}"
    local region="${2}"
    
    log_warn "Deleting function: ${function_name}"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Deletion cancelled"
        return 0
    fi
    
    aws lambda delete-function \
        --function-name "${function_name}" \
        --region "${region}" 2>/dev/null && \
    log_info "Function deleted" || \
    log_error "Function deletion failed"
}

# Show status
show_status() {
    local function_name="${1}"
    local region="${2}"
    
    log_info "Function status: ${function_name}"
    
    aws lambda get-function \
        --function-name "${function_name}" \
        --region "${region}" \
        --query 'Configuration.[FunctionName,Runtime,LastModified,State,CodeSize]' \
        --output table 2>/dev/null || \
    log_warn "Function not found or AWS CLI not configured"
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        deploy)
            deploy_function "${FUNCTION_NAME}" "${RUNTIME}" "${REGION}"
            ;;
        update)
            update_function "${FUNCTION_NAME}" "${REGION}"
            ;;
        invoke)
            invoke_function "${FUNCTION_NAME}" "${REGION}"
            ;;
        logs)
            view_logs "${FUNCTION_NAME}" "${REGION}"
            ;;
        delete)
            delete_function "${FUNCTION_NAME}" "${REGION}"
            ;;
        status)
            show_status "${FUNCTION_NAME}" "${REGION}"
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


