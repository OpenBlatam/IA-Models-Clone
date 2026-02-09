#!/bin/bash
# ============================================================================
# Dependency Checker Script
# Comprehensive dependency health check
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Dermatology AI - Dependency Health Check"
echo "=========================================="
echo ""

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠ Warning: No virtual environment detected${NC}"
    echo "Consider activating a virtual environment first"
    echo ""
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo -e "${BLUE}📋 Python Version:${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python: $python_version"
required_version="3.10"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "  ${GREEN}✓ Python version is compatible${NC}"
else
    echo -e "  ${RED}✗ Python 3.10+ required${NC}"
fi
echo ""

# Check pip version
echo -e "${BLUE}📦 Pip Version:${NC}"
pip_version=$(pip --version | awk '{print $2}')
echo "  pip: $pip_version"
echo ""

# Check for outdated packages
echo -e "${BLUE}🔄 Checking for outdated packages...${NC}"
outdated_count=$(pip list --outdated 2>/dev/null | wc -l)
if [ "$outdated_count" -gt 1 ]; then
    echo -e "  ${YELLOW}⚠ Found $((outdated_count - 1)) outdated packages${NC}"
    echo "  Run 'pip list --outdated' to see details"
else
    echo -e "  ${GREEN}✓ All packages are up to date${NC}"
fi
echo ""

# Check for security vulnerabilities
echo -e "${BLUE}🔒 Checking for security vulnerabilities...${NC}"
if command_exists safety; then
    if safety check -r requirements.txt --short-report 2>/dev/null; then
        echo -e "  ${GREEN}✓ No known vulnerabilities found${NC}"
    else
        echo -e "  ${RED}✗ Security vulnerabilities detected${NC}"
        echo "  Run 'safety check -r requirements.txt' for details"
    fi
else
    echo -e "  ${YELLOW}⚠ safety not installed${NC}"
    echo "  Install with: pip install safety"
fi
echo ""

# Check for pip-audit
if command_exists pip-audit; then
    echo -e "${BLUE}🔍 Running pip-audit...${NC}"
    pip-audit -r requirements.txt --desc 2>/dev/null || true
    echo ""
fi

# Check for duplicate dependencies
echo -e "${BLUE}🔍 Checking for duplicate dependencies...${NC}"
duplicates=$(grep -E "^[a-zA-Z]" requirements.txt | cut -d'=' -f1 | cut -d'[' -f1 | sort | uniq -d)
if [ -z "$duplicates" ]; then
    echo -e "  ${GREEN}✓ No duplicate dependencies found${NC}"
else
    echo -e "  ${RED}✗ Found duplicate dependencies:${NC}"
    echo "$duplicates"
fi
echo ""

# Check for missing dependencies
echo -e "${BLUE}📚 Checking installed packages...${NC}"
missing_packages=0
while IFS= read -r line; do
    if [[ $line =~ ^[a-zA-Z] ]] && [[ ! $line =~ ^# ]]; then
        package=$(echo "$line" | cut -d'=' -f1 | cut -d'[' -f1 | tr '[:upper:]' '[:lower:]')
        if ! pip show "$package" &>/dev/null; then
            echo -e "  ${RED}✗ Missing: $package${NC}"
            ((missing_packages++))
        fi
    fi
done < requirements.txt

if [ $missing_packages -eq 0 ]; then
    echo -e "  ${GREEN}✓ All required packages are installed${NC}"
fi
echo ""

# Check file sizes
echo -e "${BLUE}📊 Requirements file sizes:${NC}"
for file in requirements*.txt; do
    if [ -f "$file" ]; then
        size=$(wc -l < "$file")
        echo "  $file: $size lines"
    fi
done
echo ""

# Check for lock file
echo -e "${BLUE}🔐 Checking for lock file...${NC}"
if [ -f "requirements-lock.txt" ]; then
    echo -e "  ${GREEN}✓ requirements-lock.txt exists${NC}"
    lock_age=$(find requirements-lock.txt -mtime +30 2>/dev/null && echo "old" || echo "recent")
    if [ "$lock_age" = "old" ]; then
        echo -e "  ${YELLOW}⚠ Lock file is older than 30 days${NC}"
        echo "  Consider regenerating with: pip-compile requirements.txt"
    fi
else
    echo -e "  ${YELLOW}⚠ requirements-lock.txt not found${NC}"
    echo "  Generate with: pip-compile requirements.txt"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${BLUE}📋 Summary:${NC}"
echo "=========================================="
echo "Python: $python_version"
echo "Pip: $pip_version"
echo "Outdated packages: $((outdated_count - 1))"
echo "Missing packages: $missing_packages"
echo ""
echo -e "${GREEN}✓ Health check completed${NC}"
echo ""
echo "Next steps:"
echo "  - Update packages: pip install --upgrade -r requirements.txt"
echo "  - Check security: safety check -r requirements.txt"
echo "  - Generate lock: pip-compile requirements.txt"



