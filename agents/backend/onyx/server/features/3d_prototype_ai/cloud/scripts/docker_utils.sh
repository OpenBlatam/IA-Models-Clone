#!/bin/bash
# Docker utility scripts for 3D Prototype AI
# Provides convenient commands for Docker operations

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Docker utility commands for 3D Prototype AI.

COMMANDS:
    build           Build Docker images
    up              Start containers
    down            Stop containers
    restart         Restart containers
    logs            View container logs
    exec            Execute command in container
    shell           Open shell in container
    ps              List running containers
    clean           Clean Docker resources
    health          Check container health
    stats           Show container statistics

OPTIONS:
    -e, --env ENV   Environment (production|development)
    -f, --file FILE Compose file to use
    -s, --service   Service name (for exec, logs, etc.)
    -h, --help      Show this help message

EXAMPLES:
    $0 build
    $0 up -e development
    $0 logs -s api
    $0 exec -s api "python manage.py migrate"
    $0 clean

EOF
}

# Parse arguments
parse_args() {
    COMMAND="${1:-}"
    shift || true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -s|--service)
                SERVICE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Set compose file based on environment
    if [ "${ENVIRONMENT}" = "development" ]; then
        COMPOSE_FILE="docker-compose.development.yml"
    elif [ "${ENVIRONMENT}" = "production" ]; then
        COMPOSE_FILE="docker-compose.production.yml"
    fi
}

# Build Docker images
docker_build() {
    log_info "Building Docker images for ${ENVIRONMENT} environment..."
    cd "${PROJECT_ROOT}"
    
    if [ "${ENVIRONMENT}" = "production" ]; then
        docker-compose -f "${COMPOSE_FILE}" build --no-cache
    else
        docker-compose -f "${COMPOSE_FILE}" build
    fi
    
    log_info "Build completed ✓"
}

# Start containers
docker_up() {
    log_info "Starting containers for ${ENVIRONMENT} environment..."
    cd "${PROJECT_ROOT}"
    
    docker-compose -f "${COMPOSE_FILE}" up -d
    
    log_info "Containers started ✓"
    docker-compose -f "${COMPOSE_FILE}" ps
}

# Stop containers
docker_down() {
    log_info "Stopping containers..."
    cd "${PROJECT_ROOT}"
    
    docker-compose -f "${COMPOSE_FILE}" down
    
    log_info "Containers stopped ✓"
}

# Restart containers
docker_restart() {
    log_info "Restarting containers..."
    cd "${PROJECT_ROOT}"
    
    docker-compose -f "${COMPOSE_FILE}" restart
    
    log_info "Containers restarted ✓"
}

# View logs
docker_logs() {
    local service="${SERVICE:-}"
    cd "${PROJECT_ROOT}"
    
    if [ -n "${service}" ]; then
        log_info "Viewing logs for service: ${service}"
        docker-compose -f "${COMPOSE_FILE}" logs -f "${service}"
    else
        log_info "Viewing logs for all services"
        docker-compose -f "${COMPOSE_FILE}" logs -f
    fi
}

# Execute command in container
docker_exec() {
    local service="${SERVICE:-api}"
    local command="${1:-/bin/bash}"
    
    cd "${PROJECT_ROOT}"
    
    log_info "Executing command in ${service}: ${command}"
    docker-compose -f "${COMPOSE_FILE}" exec "${service}" ${command}
}

# Open shell in container
docker_shell() {
    local service="${SERVICE:-api}"
    
    cd "${PROJECT_ROOT}"
    
    log_info "Opening shell in ${service}"
    docker-compose -f "${COMPOSE_FILE}" exec "${service}" /bin/bash
}

# List containers
docker_ps() {
    cd "${PROJECT_ROOT}"
    
    log_info "Running containers:"
    docker-compose -f "${COMPOSE_FILE}" ps
}

# Clean Docker resources
docker_clean() {
    log_warn "This will remove all containers, volumes, and images"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "${confirm}" != "yes" ]; then
        log_info "Cleanup cancelled"
        return 0
    fi
    
    cd "${PROJECT_ROOT}"
    
    log_info "Stopping and removing containers..."
    docker-compose -f "${COMPOSE_FILE}" down -v
    
    log_info "Removing unused images..."
    docker image prune -f
    
    log_info "Removing unused volumes..."
    docker volume prune -f
    
    log_info "Cleanup completed ✓"
}

# Check container health
docker_health() {
    cd "${PROJECT_ROOT}"
    
    log_info "Checking container health..."
    
    local services
    services=$(docker-compose -f "${COMPOSE_FILE}" ps --services)
    
    for service in ${services}; do
        local status
        status=$(docker-compose -f "${COMPOSE_FILE}" ps -q "${service}" | xargs docker inspect --format='{{.State.Health.Status}}' 2>/dev/null || echo "no-healthcheck")
        
        if [ "${status}" = "healthy" ]; then
            log_info "✓ ${service}: healthy"
        elif [ "${status}" = "no-healthcheck" ]; then
            local running
            running=$(docker-compose -f "${COMPOSE_FILE}" ps -q "${service}" | xargs docker inspect --format='{{.State.Running}}' 2>/dev/null || echo "false")
            if [ "${running}" = "true" ]; then
                log_info "○ ${service}: running (no healthcheck)"
            else
                log_warn "✗ ${service}: not running"
            fi
        else
            log_warn "✗ ${service}: ${status}"
        fi
    done
}

# Show container statistics
docker_stats() {
    cd "${PROJECT_ROOT}"
    
    log_info "Container statistics:"
    docker stats --no-stream $(docker-compose -f "${COMPOSE_FILE}" ps -q)
}

# Main function
main() {
    if [ $# -eq 0 ]; then
        usage
        exit 1
    fi
    
    parse_args "$@"
    
    case "${COMMAND}" in
        build)
            docker_build
            ;;
        up)
            docker_up
            ;;
        down)
            docker_down
            ;;
        restart)
            docker_restart
            ;;
        logs)
            docker_logs
            ;;
        exec)
            shift
            docker_exec "$@"
            ;;
        shell)
            docker_shell
            ;;
        ps)
            docker_ps
            ;;
        clean)
            docker_clean
            ;;
        health)
            docker_health
            ;;
        stats)
            docker_stats
            ;;
        *)
            log_error "Unknown command: ${COMMAND}"
            usage
            exit 1
            ;;
    esac
}

main "$@"

