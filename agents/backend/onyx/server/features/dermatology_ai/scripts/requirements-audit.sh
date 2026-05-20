#!/bin/bash
# ============================================================================
# Requirements Audit
# Comprehensive audit of all requirements files
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

AUDIT_REPORT="requirements-audit-report.txt"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

echo "=========================================="
echo "Requirements Audit"
echo "=========================================="
echo ""

{
    echo "Requirements Audit Report"
    echo "Generated: $TIMESTAMP"
    echo "=========================================="
    echo ""
} > "$AUDIT_REPORT"

# 1. File existence check
echo -e "${BLUE}1. Checking file existence...${NC}"
{
    echo "1. FILE EXISTENCE"
    echo "-----------------"
} >> "$AUDIT_REPORT"

for file in requirements.txt requirements-optimized.txt requirements-dev.txt \
            requirements-minimal.txt requirements-gpu.txt requirements-docker.txt; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
        echo "  ✓ $file" >> "$AUDIT_REPORT"
    else
        echo -e "  ${RED}✗${NC} $file (missing)"
        echo "  ✗ $file (missing)" >> "$AUDIT_REPORT"
    fi
done
echo "" >> "$AUDIT_REPORT"
echo ""

# 2. Package count
echo -e "${BLUE}2. Counting packages...${NC}"
{
    echo "2. PACKAGE COUNTS"
    echo "-----------------"
} >> "$AUDIT_REPORT"

for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        count=$(grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | wc -l)
        echo "  $file: $count packages"
        echo "  $file: $count packages" >> "$AUDIT_REPORT"
    fi
done
echo "" >> "$AUDIT_REPORT"
echo ""

# 3. Duplicate check
echo -e "${BLUE}3. Checking for duplicates...${NC}"
{
    echo "3. DUPLICATES"
    echo "-------------"
} >> "$AUDIT_REPORT"

for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        duplicates=$(grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | \
                    sed 's/[>=<].*//' | sed 's/\[.*\]//' | sort | uniq -d)
        if [ -n "$duplicates" ]; then
            echo -e "  ${YELLOW}⚠${NC} $file has duplicates"
            echo "  ⚠ $file has duplicates:" >> "$AUDIT_REPORT"
            echo "$duplicates" | sed 's/^/    - /' >> "$AUDIT_REPORT"
        else
            echo -e "  ${GREEN}✓${NC} $file (no duplicates)"
            echo "  ✓ $file (no duplicates)" >> "$AUDIT_REPORT"
        fi
    fi
done
echo "" >> "$AUDIT_REPORT"
echo ""

# 4. Security check
echo -e "${BLUE}4. Security check...${NC}"
{
    echo "4. SECURITY"
    echo "-----------"
} >> "$AUDIT_REPORT"

if command -v safety &> /dev/null; then
    if safety check -r requirements.txt --short-report 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} No known vulnerabilities"
        echo "  ✓ No known vulnerabilities" >> "$AUDIT_REPORT"
    else
        echo -e "  ${RED}✗${NC} Vulnerabilities found"
        echo "  ✗ Vulnerabilities found" >> "$AUDIT_REPORT"
        safety check -r requirements.txt --short-report 2>&1 >> "$AUDIT_REPORT" || true
    fi
else
    echo -e "  ${YELLOW}⚠${NC} safety not installed"
    echo "  ⚠ safety not installed" >> "$AUDIT_REPORT"
fi
echo "" >> "$AUDIT_REPORT"
echo ""

# 5. Outdated packages
echo -e "${BLUE}5. Checking outdated packages...${NC}"
{
    echo "5. OUTDATED PACKAGES"
    echo "--------------------"
} >> "$AUDIT_REPORT"

outdated=$(pip list --outdated 2>/dev/null | tail -n +3 | wc -l)
if [ "$outdated" -gt 0 ]; then
    echo -e "  ${YELLOW}⚠${NC} $outdated outdated packages"
    echo "  ⚠ $outdated outdated packages" >> "$AUDIT_REPORT"
    pip list --outdated 2>/dev/null | head -10 >> "$AUDIT_REPORT" || true
else
    echo -e "  ${GREEN}✓${NC} All packages up to date"
    echo "  ✓ All packages up to date" >> "$AUDIT_REPORT"
fi
echo "" >> "$AUDIT_REPORT"
echo ""

# 6. Lock file check
echo -e "${BLUE}6. Checking lock file...${NC}"
{
    echo "6. LOCK FILE"
    echo "------------"
} >> "$AUDIT_REPORT"

if [ -f "requirements-lock.txt" ]; then
    age=$(find requirements-lock.txt -mtime +30 2>/dev/null && echo "old" || echo "recent")
    if [ "$age" = "old" ]; then
        echo -e "  ${YELLOW}⚠${NC} Lock file is older than 30 days"
        echo "  ⚠ Lock file is older than 30 days" >> "$AUDIT_REPORT"
    else
        echo -e "  ${GREEN}✓${NC} Lock file is recent"
        echo "  ✓ Lock file is recent" >> "$AUDIT_REPORT"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} No lock file found"
    echo "  ⚠ No lock file found" >> "$AUDIT_REPORT"
fi
echo "" >> "$AUDIT_REPORT"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✓ Audit completed${NC}"
echo "=========================================="
echo "Report saved to: $AUDIT_REPORT"
echo ""

# Add summary to report
{
    echo "SUMMARY"
    echo "-------"
    echo "Audit completed: $TIMESTAMP"
    echo "Report file: $AUDIT_REPORT"
} >> "$AUDIT_REPORT"



