#!/bin/bash

# OS Content System Deployment Script
# Deploys the optimized system with all components

set -e

echo "🚀 Starting OS Content System Deployment..."

# Configuration
APP_NAME="os-content-system"
DOCKER_IMAGE="os-content:latest"
CONTAINER_NAME="os-content-app"
REDIS_CONTAINER="os-content-redis"
NETWORK_NAME="os-content-network"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Create Docker network
create_network() {
    print_status "Creating Docker network..."
    
    if ! docker network ls | grep -q $NETWORK_NAME; then
        docker network create $NETWORK_NAME
        print_success "Network $NETWORK_NAME created"
    else
        print_warning "Network $NETWORK_NAME already exists"
    fi
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    docker build -t $DOCKER_IMAGE .
    print_success "Docker image built successfully"
}

# Start Redis
start_redis() {
    print_status "Starting Redis..."
    
    if docker ps -q -f name=$REDIS_CONTAINER | grep -q .; then
        print_warning "Redis container already running"
    else
        docker run -d \
            --name $REDIS_CONTAINER \
            --network $NETWORK_NAME \
            -p 6379:6379 \
            redis:7-alpine
        print_success "Redis started"
    fi
}

# Start the application
start_application() {
    print_status "Starting OS Content application..."
    
    # Stop existing container if running
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_warning "Stopping existing container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
    
    # Start new container
    docker run -d \
        --name $CONTAINER_NAME \
        --network $NETWORK_NAME \
        -p 8000:8000 \
        -e REDIS_URL="redis://$REDIS_CONTAINER:6379" \
        -e LOG_LEVEL="INFO" \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/data:/app/data \
        $DOCKER_IMAGE
    
    print_success "Application started"
}

# Wait for application to be ready
wait_for_ready() {
    print_status "Waiting for application to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null; then
            print_success "Application is ready!"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "Application failed to start within expected time"
    return 1
}

# Run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Check application health
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        print_success "Application health check passed"
    else
        print_error "Application health check failed"
        return 1
    fi
    
    # Check Redis connection
    if docker exec $REDIS_CONTAINER redis-cli ping | grep -q "PONG"; then
        print_success "Redis health check passed"
    else
        print_error "Redis health check failed"
        return 1
    fi
    
    print_success "All health checks passed"
}

# Show deployment info
show_deployment_info() {
    print_success "Deployment completed successfully!"
    echo ""
    echo "📊 Deployment Information:"
    echo "   Application: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo "   Health Check: http://localhost:8000/health"
    echo "   Redis: localhost:6379"
    echo ""
    echo "🔧 Useful Commands:"
    echo "   View logs: docker logs $CONTAINER_NAME"
    echo "   Stop app: docker stop $CONTAINER_NAME"
    echo "   Restart app: docker restart $CONTAINER_NAME"
    echo "   Remove all: docker-compose down"
    echo ""
}

# Main deployment function
deploy() {
    print_status "Starting deployment process..."
    
    check_prerequisites
    create_network
    build_image
    start_redis
    start_application
    wait_for_ready
    run_health_checks
    show_deployment_info
}

# Cleanup function
cleanup() {
    print_status "Cleaning up deployment..."
    
    # Stop and remove containers
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    docker stop $REDIS_CONTAINER 2>/dev/null || true
    docker rm $REDIS_CONTAINER 2>/dev/null || true
    
    # Remove network
    docker network rm $NETWORK_NAME 2>/dev/null || true
    
    # Remove image
    docker rmi $DOCKER_IMAGE 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Show logs
show_logs() {
    print_status "Showing application logs..."
    docker logs -f $CONTAINER_NAME
}

# Show status
show_status() {
    print_status "System status:"
    echo ""
    echo "📦 Containers:"
    docker ps --filter "name=os-content" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "🌐 Network:"
    docker network ls --filter "name=os-content" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"
    echo ""
    echo "💾 Images:"
    docker images --filter "reference=os-content" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "restart")
        print_status "Restarting application..."
        docker restart $CONTAINER_NAME
        print_success "Application restarted"
        ;;
    "stop")
        print_status "Stopping application..."
        docker stop $CONTAINER_NAME
        print_success "Application stopped"
        ;;
    "start")
        print_status "Starting application..."
        docker start $CONTAINER_NAME
        print_success "Application started"
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|logs|status|restart|stop|start}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the complete system"
        echo "  cleanup  - Remove all containers and images"
        echo "  logs     - Show application logs"
        echo "  status   - Show system status"
        echo "  restart  - Restart the application"
        echo "  stop     - Stop the application"
        echo "  start    - Start the application"
        exit 1
        ;;
esac 