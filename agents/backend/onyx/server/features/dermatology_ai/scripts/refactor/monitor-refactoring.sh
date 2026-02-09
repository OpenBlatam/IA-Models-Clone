#!/bin/bash
# ============================================================================
# Monitor Refactoring Progress
# Monitors refactoring progress over time
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/.requirements-logs/refactoring-progress.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "=========================================="
echo "Refactoring Progress Monitor"
echo "=========================================="
echo ""

# Count files in each category
count_files() {
    dir=$1
    pattern=$2
    if [ -d "$dir" ]; then
        find "$dir" -name "$pattern" 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# Services
services_total=$(count_files "$PROJECT_ROOT/services" "*.py")
services_organized=0
for category in analysis recommendations tracking products ml notifications integrations reporting social shared; do
    count=$(count_files "$PROJECT_ROOT/services/$category" "*.py")
    services_organized=$((services_organized + count))
done

# Utils
utils_total=$(count_files "$PROJECT_ROOT/utils" "*.py")
utils_organized=0
for category in logging caching validation security performance database async monitoring helpers; do
    count=$(count_files "$PROJECT_ROOT/utils/$category" "*.py")
    utils_organized=$((utils_organized + count))
done

# Docs
docs_total=$(count_files "$PROJECT_ROOT" "*.md")
docs_organized=0
for category in architecture dependencies features guides api; do
    count=$(count_files "$PROJECT_ROOT/docs/$category" "*.md")
    docs_organized=$((docs_organized + count))
done

# Scripts
scripts_total=$(count_files "$PROJECT_ROOT/scripts" "*.sh" "*.py")
scripts_organized=0
for category in analysis validation management utils; do
    count=$(count_files "$PROJECT_ROOT/scripts/requirements/$category" "")
    scripts_organized=$((scripts_organized + count))
done

# Calculate percentages
services_pct=0
if [ "$services_total" -gt 0 ]; then
    services_pct=$((services_organized * 100 / services_total))
fi

utils_pct=0
if [ "$utils_total" -gt 0 ]; then
    utils_pct=$((utils_organized * 100 / utils_total))
fi

docs_pct=0
if [ "$docs_total" -gt 0 ]; then
    docs_pct=$((docs_organized * 100 / docs_total))
fi

scripts_pct=0
if [ "$scripts_total" -gt 0 ]; then
    scripts_pct=$((scripts_organized * 100 / scripts_total))
fi

# Display progress
echo -e "${BLUE}Refactoring Progress:${NC}"
echo ""
echo "Services:"
echo "  Organized: $services_organized / $services_total ($services_pct%)"
echo ""

echo "Utils:"
echo "  Organized: $utils_organized / $utils_total ($utils_pct%)"
echo ""

echo "Documentation:"
echo "  Organized: $docs_organized / $docs_total ($docs_pct%)"
echo ""

echo "Scripts:"
echo "  Organized: $scripts_organized / $scripts_total ($scripts_pct%)"
echo ""

# Overall progress
total_files=$((services_total + utils_total + docs_total + scripts_total))
total_organized=$((services_organized + utils_organized + docs_organized + scripts_organized))
overall_pct=0
if [ "$total_files" -gt 0 ]; then
    overall_pct=$((total_organized * 100 / total_files))
fi

echo "=========================================="
echo "Overall Progress: $total_organized / $total_files ($overall_pct%)"
echo "=========================================="

# Log progress
{
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Services: $services_organized/$services_total ($services_pct%)"
    echo "Utils: $utils_organized/$utils_total ($utils_pct%)"
    echo "Docs: $docs_organized/$docs_total ($docs_pct%)"
    echo "Scripts: $scripts_organized/$scripts_total ($scripts_pct%)"
    echo "Overall: $total_organized/$total_files ($overall_pct%)"
    echo "---"
} >> "$LOG_FILE"

echo ""
echo "Progress logged to: $LOG_FILE"



