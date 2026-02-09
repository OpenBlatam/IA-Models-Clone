#!/bin/bash
# One-command startup script for Music Analyzer AI
# Usage: ./start.sh [dev|prod]

set -e

ENVIRONMENT=${1:-dev}
COMPOSE_FILE="docker-compose.yml"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Music Analyzer AI...${NC}"

# Select compose file
case $ENVIRONMENT in
  dev|development)
    COMPOSE_FILE="docker-compose.dev.yml"
    echo -e "${YELLOW}📦 Development mode${NC}"
    ;;
  prod|production)
    COMPOSE_FILE="docker-compose.prod.yml"
    echo -e "${YELLOW}📦 Production mode${NC}"
    ;;
  *)
    echo -e "${YELLOW}📦 Default mode${NC}"
    ;;
esac

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ docker-compose is not installed.${NC}"
    exit 1
fi

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Navigate to deployment directory
cd "$(dirname "$0")"

# Check for .env file
if [ ! -f ../.env ] && [ "$ENVIRONMENT" != "dev" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating template...${NC}"
    cat > ../.env << EOF
ENVIRONMENT=$ENVIRONMENT
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
LOG_LEVEL=INFO
CACHE_ENABLED=true
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=admin
DATABASE_URL=postgresql://music_analyzer:changeme@postgres:5432/music_analyzer_db
EOF
    echo -e "${YELLOW}📝 Please update ../.env with your actual credentials!${NC}"
fi

# Build images if needed
echo -e "${GREEN}🔨 Building Docker images...${NC}"
$DOCKER_COMPOSE -f $COMPOSE_FILE build --quiet

# Start services
echo -e "${GREEN}🚀 Starting services...${NC}"
$DOCKER_COMPOSE -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo -e "${GREEN}⏳ Waiting for services to be ready...${NC}"
sleep 5

# Check health
MAX_RETRIES=30
RETRY_COUNT=0
HEALTHY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8010/health > /dev/null 2>&1; then
        HEALTHY=true
        break
    fi
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo ""

if [ "$HEALTHY" = true ]; then
    echo -e "${GREEN}✅ All services are running!${NC}"
    echo ""
    echo -e "${GREEN}📊 Service URLs:${NC}"
    echo "  🌐 API:          http://localhost:8010"
    echo "  ❤️  Health:       http://localhost:8010/health"
    echo "  📖 Docs:          http://localhost:8010/docs"
    
    if [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "prod" ]; then
        echo "  📈 Grafana:       http://localhost:3000"
        echo "  📊 Prometheus:    http://localhost:9090"
    fi
    
    echo ""
    echo -e "${GREEN}📝 Useful commands:${NC}"
    echo "  View logs:    $DOCKER_COMPOSE -f $COMPOSE_FILE logs -f"
    echo "  Stop:        $DOCKER_COMPOSE -f $COMPOSE_FILE down"
    echo "  Restart:     $DOCKER_COMPOSE -f $COMPOSE_FILE restart"
    echo ""
else
    echo -e "${RED}⚠️  Services may still be starting. Check logs with:${NC}"
    echo "  $DOCKER_COMPOSE -f $COMPOSE_FILE logs"
fi




