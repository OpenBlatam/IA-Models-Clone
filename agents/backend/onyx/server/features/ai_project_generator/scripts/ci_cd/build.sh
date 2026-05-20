#!/bin/bash

###############################################################################
# CI/CD Build Script
# Automated build script for CI/CD pipelines
###############################################################################

set -euo pipefail

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/../aws/scripts/common_functions_enhanced.sh" 2>/dev/null || {
    source "${SCRIPT_DIR}/../aws/scripts/common_functions.sh" 2>/dev/null || {
        echo "Error: common functions not found" >&2
        exit 1
    }
}

# Configuration
BUILD_TYPE="${BUILD_TYPE:-docker}" # docker, python, all
IMAGE_NAME="${IMAGE_NAME:-ai-project-generator}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
BUILD_CACHE="${BUILD_CACHE:-true}"

###############################################################################
# Build Functions
###############################################################################

build_docker_image() {
    log_section "Building Docker Image"
    
    local docker_build_args=(
        "build"
        "-f" "${PROJECT_ROOT}/${DOCKERFILE}"
        "-t" "${IMAGE_NAME}:${IMAGE_TAG}"
    )
    
    # Add cache options
    if [ "${BUILD_CACHE}" = "true" ]; then
        docker_build_args+=("--cache-from" "type=gha")
        docker_build_args+=("--cache-to" "type=gha,mode=max")
    fi
    
    # Add build args
    docker_build_args+=(
        "--build-arg" "BUILD_DATE=$(date -Iseconds)"
        "--build-arg" "VCS_REF=${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo 'unknown')}"
        "--build-arg" "VERSION=${IMAGE_TAG}"
        "--build-arg" "PYTHON_VERSION=${PYTHON_VERSION}"
    )
    
    docker_build_args+=("${PROJECT_ROOT}")
    
    log_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    if docker "${docker_build_args[@]}"; then
        log_success "Docker image built successfully"
        return 0
    else
        log_error "Docker build failed"
        return 1
    fi
}

build_python_package() {
    log_section "Building Python Package"
    
    cd "${PROJECT_ROOT}" || exit 1
    
    # Check if setup.py or pyproject.toml exists
    if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
        log_warn "No Python package configuration found, skipping"
        return 0
    fi
    
    # Install build tools
    if ! check_command "python"; then
        log_error "Python not found"
        return 1
    fi
    
    python -m pip install --upgrade pip build wheel || {
        log_error "Failed to install build tools"
        return 1
    }
    
    # Build package
    log_info "Building Python package..."
    if python -m build; then
        log_success "Python package built successfully"
        return 0
    else
        log_error "Python package build failed"
        return 1
    fi
}

test_docker_image() {
    log_section "Testing Docker Image"
    
    # Test image can start
    log_info "Testing Docker image startup..."
    if docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python --version > /dev/null 2>&1; then
        log_success "Docker image test passed"
        return 0
    else
        log_error "Docker image test failed"
        return 1
    fi
}

scan_docker_image() {
    log_section "Scanning Docker Image"
    
    if ! check_command "trivy"; then
        log_warn "Trivy not installed, skipping image scan"
        return 0
    fi
    
    log_info "Scanning Docker image for vulnerabilities..."
    if trivy image --exit-code 0 --severity HIGH,CRITICAL "${IMAGE_NAME}:${IMAGE_TAG}"; then
        log_success "Docker image scan passed"
        return 0
    else
        log_warn "Docker image scan found vulnerabilities"
        return 1
    fi
}

###############################################################################
# Main Execution
###############################################################################

main() {
    init_logging "/var/log/cicd-build.log"
    log_section "CI/CD Build Process"
    log_info "Build Type: ${BUILD_TYPE}"
    log_info "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    
    local exit_code=0
    
    case "${BUILD_TYPE}" in
        docker)
            build_docker_image || exit_code=1
            if [ $exit_code -eq 0 ]; then
                test_docker_image || exit_code=1
                scan_docker_image || exit_code=1
            fi
            ;;
        python)
            build_python_package || exit_code=1
            ;;
        all)
            build_python_package || exit_code=1
            build_docker_image || exit_code=1
            if [ $exit_code -eq 0 ]; then
                test_docker_image || exit_code=1
                scan_docker_image || exit_code=1
            fi
            ;;
        *)
            log_error "Unknown build type: ${BUILD_TYPE}"
            exit 1
            ;;
    esac
    
    if [ $exit_code -eq 0 ]; then
        log_success "Build process completed successfully"
    else
        log_error "Build process failed"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"

