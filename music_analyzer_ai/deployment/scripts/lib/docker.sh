#!/bin/bash
# Docker Utility Library
# Reusable Docker functions for deployment scripts

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Docker functions
docker_is_running() {
    docker info &> /dev/null
}

docker_check_image_exists() {
    local image_name="$1"
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image_name}$"
}

docker_pull_image() {
    local image_name="$1"
    log_info "Pulling Docker image: ${image_name}"
    
    retry_with_backoff 3 5 "docker pull ${image_name}"
}

docker_build_image() {
    local context="$1"
    local dockerfile="${2:-Dockerfile}"
    local image_name="$3"
    local build_args="${4:-}"
    
    log_info "Building Docker image: ${image_name}"
    
    local build_cmd="docker build -t ${image_name} -f ${dockerfile} ${context}"
    
    if [ -n "${build_args}" ]; then
        build_cmd="${build_cmd} ${build_args}"
    fi
    
    if ! eval "${build_cmd}"; then
        log_error "Failed to build Docker image"
        return 1
    fi
    
    log_success "Docker image built successfully"
}

docker_save_image() {
    local image_name="$1"
    local output_file="$2"
    
    log_info "Saving Docker image to ${output_file}"
    
    if docker save "${image_name}" | gzip > "${output_file}"; then
        log_success "Image saved successfully"
        return 0
    else
        log_error "Failed to save Docker image"
        return 1
    fi
}

docker_load_image() {
    local image_file="$1"
    
    log_info "Loading Docker image from ${image_file}"
    
    if docker load < "${image_file}"; then
        log_success "Image loaded successfully"
        return 0
    else
        log_error "Failed to load Docker image"
        return 1
    fi
}

docker_stop_container() {
    local container_name="$1"
    
    if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        log_info "Stopping container: ${container_name}"
        docker stop "${container_name}" || true
        docker rm "${container_name}" || true
        log_success "Container stopped"
    else
        log_debug "Container ${container_name} not running"
    fi
}

docker_start_container() {
    local container_name="$1"
    local image_name="$2"
    local env_file="${3:-}"
    local ports="${4:-}"
    local volumes="${5:-}"
    
    log_info "Starting container: ${container_name}"
    
    local docker_cmd="docker run -d --name ${container_name} --restart unless-stopped"
    
    if [ -n "${env_file}" ] && [ -f "${env_file}" ]; then
        docker_cmd="${docker_cmd} --env-file ${env_file}"
    fi
    
    if [ -n "${ports}" ]; then
        docker_cmd="${docker_cmd} -p ${ports}"
    fi
    
    if [ -n "${volumes}" ]; then
        docker_cmd="${docker_cmd} -v ${volumes}"
    fi
    
    docker_cmd="${docker_cmd} ${image_name}"
    
    if eval "${docker_cmd}"; then
        log_success "Container started successfully"
        return 0
    else
        log_error "Failed to start container"
        return 1
    fi
}

docker_get_container_status() {
    local container_name="$1"
    docker inspect --format='{{.State.Status}}' "${container_name}" 2>/dev/null || echo "not-found"
}

docker_get_container_logs() {
    local container_name="$1"
    local lines="${2:-50}"
    docker logs "${container_name}" --tail "${lines}" 2>&1
}

docker_cleanup_images() {
    local image_pattern="${1:-}"
    local keep_count="${2:-2}"
    
    log_info "Cleaning up old Docker images (keeping last ${keep_count})"
    
    if [ -z "${image_pattern}" ]; then
        log_warn "No image pattern specified, skipping cleanup"
        return 0
    fi
    
    docker images "${image_pattern}" --format "{{.ID}}" | \
        tail -n +$((keep_count + 1)) | \
        xargs -r docker rmi || true
    
    log_success "Image cleanup completed"
}

docker_compose_up() {
    local compose_file="${1:-docker-compose.yml}"
    local services="${2:-}"
    
    log_info "Starting services with docker-compose"
    
    local compose_cmd="docker-compose -f ${compose_file} up -d"
    
    if [ -n "${services}" ]; then
        compose_cmd="${compose_cmd} ${services}"
    fi
    
    if eval "${compose_cmd}"; then
        log_success "Services started successfully"
        return 0
    else
        log_error "Failed to start services"
        return 1
    fi
}

docker_compose_down() {
    local compose_file="${1:-docker-compose.yml}"
    
    log_info "Stopping services with docker-compose"
    
    if docker-compose -f "${compose_file}" down; then
        log_success "Services stopped successfully"
        return 0
    else
        log_error "Failed to stop services"
        return 1
    fi
}

# Export functions
export -f docker_is_running docker_check_image_exists docker_pull_image
export -f docker_build_image docker_save_image docker_load_image
export -f docker_stop_container docker_start_container docker_get_container_status
export -f docker_get_container_logs docker_cleanup_images
export -f docker_compose_up docker_compose_down




