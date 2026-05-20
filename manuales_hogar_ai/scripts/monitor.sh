#!/bin/bash
# Monitor script - Real-time monitoring of services
# Usage: ./scripts/monitor.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${BLUE}📊 Manuales Hogar AI - Real-time Monitor${NC}"
echo -e "${BLUE}Press Ctrl+C to exit${NC}"
echo ""

while true; do
    # Clear screen (except first time)
    if [ "$FIRST_RUN" != "true" ]; then
        clear
        echo -e "${BLUE}📊 Manuales Hogar AI - Real-time Monitor${NC}"
        echo -e "${BLUE}Press Ctrl+C to exit${NC}"
        echo ""
    fi
    FIRST_RUN="false"
    
    # Container status
    echo -e "${CYAN}=== Container Status ===${NC}"
    docker-compose ps 2>/dev/null || echo "No services running"
    echo ""
    
    # Health check
    echo -e "${CYAN}=== Health Status ===${NC}"
    if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is healthy${NC}"
        HEALTH=$(curl -s http://localhost:8000/api/v1/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "OK")
        echo "$HEALTH" | head -n 5
    else
        echo -e "${RED}❌ Service is not responding${NC}"
    fi
    echo ""
    
    # Resource usage
    echo -e "${CYAN}=== Resource Usage ===${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
        $(docker-compose ps -q) 2>/dev/null || echo "No containers running"
    echo ""
    
    # Recent logs (last 3 lines)
    echo -e "${CYAN}=== Recent Logs (last 3 lines) ===${NC}"
    docker-compose logs --tail=3 app 2>/dev/null | tail -n 3 || echo "No logs available"
    echo ""
    
    # Database connections
    echo -e "${CYAN}=== Database Status ===${NC}"
    docker-compose exec -T postgres psql -U postgres -d manuales_hogar -c \
        "SELECT count(*) as active_connections FROM pg_stat_activity WHERE datname = 'manuales_hogar';" 2>/dev/null || echo "Database not accessible"
    echo ""
    
    echo -e "${YELLOW}Refreshing in 5 seconds...${NC}"
    sleep 5
done




