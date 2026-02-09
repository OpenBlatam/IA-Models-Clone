#!/bin/bash
# Security check script - Basic security validation
# Usage: ./scripts/security-check.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"
ISSUES=0

echo -e "${BLUE}🔒 Security Check - Manuales Hogar AI${NC}"
echo "======================================"
echo ""

# Check if .env file has sensitive data exposed
echo -e "${CYAN}=== Environment File Security ===${NC}"
if [ -f .env ]; then
    if grep -q "OPENROUTER_API_KEY=$" .env || ! grep -q "OPENROUTER_API_KEY=" .env; then
        echo -e "${YELLOW}⚠️  OPENROUTER_API_KEY may not be set${NC}"
        ISSUES=$((ISSUES + 1))
    else
        echo -e "${GREEN}✅ API key is configured${NC}"
    fi
    
    # Check if .env is in .gitignore
    if [ -f .gitignore ] && grep -q "^\.env$" .gitignore; then
        echo -e "${GREEN}✅ .env is in .gitignore${NC}"
    else
        echo -e "${RED}❌ .env is NOT in .gitignore${NC}"
        ISSUES=$((ISSUES + 1))
    fi
else
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
fi
echo ""

# Check Docker security
echo -e "${CYAN}=== Docker Security ===${NC}"
if docker info > /dev/null 2>&1; then
    # Check if running as root
    if docker-compose exec app whoami 2>/dev/null | grep -q "root"; then
        echo -e "${YELLOW}⚠️  Container may be running as root${NC}"
        ISSUES=$((ISSUES + 1))
    else
        echo -e "${GREEN}✅ Container not running as root${NC}"
    fi
    
    # Check exposed ports
    EXPOSED_PORTS=$(docker-compose ps --format json 2>/dev/null | grep -o '"ports":"[^"]*"' | cut -d'"' -f4 || echo "")
    if echo "$EXPOSED_PORTS" | grep -q "0.0.0.0"; then
        echo -e "${YELLOW}⚠️  Services exposed to all interfaces${NC}"
    else
        echo -e "${GREEN}✅ Port exposure looks reasonable${NC}"
    fi
fi
echo ""

# Check API security headers
echo -e "${CYAN}=== API Security Headers ===${NC}"
HEADERS=$(curl -s -I "$API_URL/api/v1/health" 2>/dev/null || echo "")

if [ -n "$HEADERS" ]; then
    if echo "$HEADERS" | grep -qi "X-Frame-Options"; then
        echo -e "${GREEN}✅ X-Frame-Options header present${NC}"
    else
        echo -e "${YELLOW}⚠️  X-Frame-Options header missing${NC}"
    fi
    
    if echo "$HEADERS" | grep -qi "X-Content-Type-Options"; then
        echo -e "${GREEN}✅ X-Content-Type-Options header present${NC}"
    else
        echo -e "${YELLOW}⚠️  X-Content-Type-Options header missing${NC}"
    fi
    
    if echo "$HEADERS" | grep -qi "X-XSS-Protection"; then
        echo -e "${GREEN}✅ X-XSS-Protection header present${NC}"
    else
        echo -e "${YELLOW}⚠️  X-XSS-Protection header missing${NC}"
    fi
else
    echo -e "${RED}❌ Cannot check headers - API not responding${NC}"
    ISSUES=$((ISSUES + 1))
fi
echo ""

# Check for common vulnerabilities
echo -e "${CYAN}=== Common Vulnerabilities ===${NC}"

# Check if debug mode is enabled
if docker-compose exec app printenv 2>/dev/null | grep -qi "DEBUG.*true"; then
    echo -e "${RED}❌ Debug mode may be enabled${NC}"
    ISSUES=$((ISSUES + 1))
else
    echo -e "${GREEN}✅ Debug mode not detected${NC}"
fi

# Check database credentials
if docker-compose exec -T postgres psql -U postgres -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw "manuales_hogar"; then
    echo -e "${GREEN}✅ Database accessible${NC}"
else
    echo -e "${YELLOW}⚠️  Database connectivity issue${NC}"
fi
echo ""

# Summary
echo -e "${CYAN}=== Summary ===${NC}"
if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ No major security issues found${NC}"
else
    echo -e "${YELLOW}⚠️  Found $ISSUES potential security issue(s)${NC}"
    echo ""
    echo "Recommendations:"
    echo "  - Ensure .env is in .gitignore"
    echo "  - Use strong passwords for database"
    echo "  - Enable security headers in production"
    echo "  - Disable debug mode in production"
fi
echo ""




