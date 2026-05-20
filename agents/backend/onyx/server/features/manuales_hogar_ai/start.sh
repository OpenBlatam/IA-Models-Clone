#!/bin/bash
# Simple start script - Run everything with one command
# Usage: ./start.sh [dev|prod|staging] [--no-build] [--skip-health] [--migrate]

set -e

ENVIRONMENT="${1:-dev}"
SKIP_BUILD=false
SKIP_HEALTH=false
RUN_MIGRATE=false

# Parse arguments
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-health)
            SKIP_HEALTH=true
            shift
            ;;
        --migrate)
            RUN_MIGRATE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Manuales Hogar AI...${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ docker-compose is not installed. Please install it first.${NC}"
    exit 1
fi

# Use 'docker compose' if 'docker-compose' is not available
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env file. Please edit it with your configuration.${NC}"
        echo -e "${YELLOW}   Required: OPENROUTER_API_KEY${NC}"
        if [ -t 0 ]; then  # Check if running in interactive terminal
            read -p "Press Enter to continue or Ctrl+C to edit .env first..."
        fi
    else
        echo -e "${RED}❌ .env.example not found. Please create .env manually.${NC}"
        exit 1
    fi
fi

# Load environment variables safely
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Check required variables
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENROUTER_API_KEY not set in .env${NC}"
    echo -e "${YELLOW}   The service will start but API calls will fail.${NC}"
    if [ -t 0 ]; then
        read -p "Press Enter to continue anyway or Ctrl+C to set it first..."
    fi
fi

# Check port availability
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use.${NC}"
    echo -e "${YELLOW}   The service may not start correctly.${NC}"
    if [ -t 0 ]; then
        read -p "Press Enter to continue anyway or Ctrl+C to stop the process using port 8000..."
    fi
fi

# Start services
if [ "$ENVIRONMENT" == "prod" ]; then
    echo -e "${BLUE}📦 Starting production environment...${NC}"
    if [ "$SKIP_BUILD" = true ]; then
        $DOCKER_COMPOSE -f docker-compose.prod.yml up -d
    else
        $DOCKER_COMPOSE -f docker-compose.prod.yml up -d --build
    fi
    COMPOSE_FILE="-f docker-compose.prod.yml"
elif [ "$ENVIRONMENT" == "staging" ]; then
    echo -e "${BLUE}🔧 Starting staging environment...${NC}"
    if [ "$SKIP_BUILD" = true ]; then
        $DOCKER_COMPOSE up -d
    else
        $DOCKER_COMPOSE up -d --build
    fi
    COMPOSE_FILE=""
else
    echo -e "${BLUE}🔧 Starting development environment...${NC}"
    if [ "$SKIP_BUILD" = true ]; then
        $DOCKER_COMPOSE up -d
    else
        $DOCKER_COMPOSE up -d --build
    fi
    COMPOSE_FILE=""
fi

# Run migrations if requested
if [ "$RUN_MIGRATE" = true ]; then
    echo -e "${BLUE}🔄 Running database migrations...${NC}"
    sleep 3
    $DOCKER_COMPOSE $COMPOSE_FILE exec -T app alembic upgrade head || echo -e "${YELLOW}⚠️  Migration failed, continuing...${NC}"
fi

# Wait for services to be ready
if [ "$SKIP_HEALTH" = false ]; then
    echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
    sleep 5

    # Check health
    MAX_RETRIES=30
    RETRY_COUNT=0
    HEALTHY=false

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Service is healthy!${NC}"
            HEALTHY=true
            break
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -e "${YELLOW}   Waiting... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
        sleep 2
    done

    if [ "$HEALTHY" = false ]; then
        echo -e "${YELLOW}⚠️  Service may not be ready yet. Check logs with: $DOCKER_COMPOSE $COMPOSE_FILE logs${NC}"
    fi
fi

# Show status
echo ""
echo -e "${GREEN}🎉 Manuales Hogar AI is running!${NC}"
echo ""
echo -e "${BLUE}📍 API URL:${NC} http://localhost:8000"
echo -e "${BLUE}📚 API Docs:${NC} http://localhost:8000/docs"
echo -e "${BLUE}❤️  Health:${NC} http://localhost:8000/api/v1/health"
echo ""
echo -e "${BLUE}📋 Useful commands:${NC}"
echo "   View logs:    $DOCKER_COMPOSE $COMPOSE_FILE logs -f"
echo "   Stop:         $DOCKER_COMPOSE $COMPOSE_FILE down"
echo "   Restart:      $DOCKER_COMPOSE $COMPOSE_FILE restart"
echo "   Shell:        $DOCKER_COMPOSE $COMPOSE_FILE exec app bash"
echo "   Status:       $DOCKER_COMPOSE $COMPOSE_FILE ps"
echo ""

