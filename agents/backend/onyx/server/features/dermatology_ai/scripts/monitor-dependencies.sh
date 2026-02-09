#!/bin/bash
# ============================================================================
# Dependency Monitor
# Monitors dependencies and sends alerts for issues
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

LOG_FILE="dependency-monitor.log"
ALERT_THRESHOLD_DAYS=30

echo "=========================================="
echo "Dependency Monitor"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}Python: $python_version${NC}"

# Check for outdated packages
echo -e "${BLUE}Checking for outdated packages...${NC}"
outdated=$(pip list --outdated 2>/dev/null | tail -n +3 | wc -l)
if [ "$outdated" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $outdated outdated packages${NC}"
    pip list --outdated | head -10
else
    echo -e "${GREEN}✓ All packages up to date${NC}"
fi
echo ""

# Check for security vulnerabilities
echo -e "${BLUE}Checking security...${NC}"
if command -v safety &> /dev/null; then
    if safety check -r requirements.txt --short-report 2>/dev/null; then
        echo -e "${GREEN}✓ No known vulnerabilities${NC}"
    else
        echo -e "${RED}✗ Security vulnerabilities found!${NC}"
        safety check -r requirements.txt --short-report
    fi
else
    echo -e "${YELLOW}⚠ safety not installed${NC}"
fi
echo ""

# Check lock file age
if [ -f "requirements-lock.txt" ]; then
    lock_age_days=$(( ($(date +%s) - $(stat -f %m requirements-lock.txt 2>/dev/null || stat -c %Y requirements-lock.txt 2>/dev/null)) / 86400 ))
    if [ "$lock_age_days" -gt "$ALERT_THRESHOLD_DAYS" ]; then
        echo -e "${YELLOW}⚠ Lock file is $lock_age_days days old${NC}"
        echo "  Consider regenerating: pip-compile requirements.txt"
    else
        echo -e "${GREEN}✓ Lock file is recent ($lock_age_days days)${NC}"
    fi
fi
echo ""

# Check for missing packages
echo -e "${BLUE}Checking installed packages...${NC}"
missing=0
while IFS= read -r line; do
    if [[ $line =~ ^[a-zA-Z] ]] && [[ ! $line =~ ^# ]]; then
        package=$(echo "$line" | cut -d'=' -f1 | cut -d'[' -f1 | tr '[:upper:]' '[:lower:]')
        if ! pip show "$package" &>/dev/null; then
            echo -e "${RED}✗ Missing: $package${NC}"
            ((missing++))
        fi
    fi
done < requirements.txt

if [ $missing -eq 0 ]; then
    echo -e "${GREEN}✓ All packages installed${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Outdated: $outdated"
echo "Missing: $missing"
echo ""

# Log results
{
    echo "$(date): Monitor run"
    echo "  Outdated: $outdated"
    echo "  Missing: $missing"
    echo ""
} >> "$LOG_FILE"

echo "Results logged to: $LOG_FILE"



