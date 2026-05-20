#!/bin/bash
# Cloud Provider Utility Library
# Reusable functions for AWS, Azure, and GCP

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# AWS functions
aws_check_credentials() {
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        return 1
    fi
    return 0
}

aws_get_instance_id() {
    local instance_name="$1"
    aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=${instance_name}" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>/dev/null
}

aws_get_instance_ip() {
    local instance_id="$1"
    aws ec2 describe-instances \
        --instance-ids "${instance_id}" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null
}

aws_upload_to_s3() {
    local local_file="$1"
    local s3_path="$2"
    
    log_info "Uploading ${local_file} to s3://${s3_path}"
    
    if aws s3 cp "${local_file}" "s3://${s3_path}"; then
        log_success "File uploaded successfully"
        return 0
    else
        log_error "Failed to upload file"
        return 1
    fi
}

aws_download_from_s3() {
    local s3_path="$1"
    local local_file="$2"
    
    log_info "Downloading s3://${s3_path} to ${local_file}"
    
    if aws s3 cp "s3://${s3_path}" "${local_file}"; then
        log_success "File downloaded successfully"
        return 0
    else
        log_error "Failed to download file"
        return 1
    fi
}

# Azure functions
azure_check_credentials() {
    if ! az account show &> /dev/null; then
        log_error "Azure credentials not configured"
        return 1
    fi
    return 0
}

azure_get_resource_group() {
    local resource_name="$1"
    az resource show \
        --name "${resource_name}" \
        --query 'resourceGroup' \
        --output tsv 2>/dev/null
}

azure_upload_to_storage() {
    local local_file="$1"
    local storage_account="$2"
    local container_name="$3"
    local blob_name="$4"
    
    log_info "Uploading ${local_file} to Azure Storage"
    
    if az storage blob upload \
        --account-name "${storage_account}" \
        --container-name "${container_name}" \
        --name "${blob_name}" \
        --file "${local_file}"; then
        log_success "File uploaded successfully"
        return 0
    else
        log_error "Failed to upload file"
        return 1
    fi
}

# GCP functions
gcp_check_credentials() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log_error "GCP credentials not configured"
        return 1
    fi
    return 0
}

gcp_get_instance_ip() {
    local instance_name="$1"
    local zone="${2:-us-central1-a}"
    
    gcloud compute instances describe "${instance_name}" \
        --zone="${zone}" \
        --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null
}

# Export functions
export -f aws_check_credentials aws_get_instance_id aws_get_instance_ip
export -f aws_upload_to_s3 aws_download_from_s3
export -f azure_check_credentials azure_get_resource_group azure_upload_to_storage
export -f gcp_check_credentials gcp_get_instance_ip




