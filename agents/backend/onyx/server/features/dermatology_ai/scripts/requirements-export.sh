#!/bin/bash
# ============================================================================
# Requirements Export
# Exports requirements in various formats
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Requirements Export"
echo "=========================================="
echo ""

# Export to JSON
echo -e "${BLUE}Exporting to JSON...${NC}"
python3 << 'EOF'
import json
import re
from pathlib import Path

def parse_requirements(filepath):
    packages = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-r'):
                continue
            match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)\s*(.*)$', line)
            if match:
                package = match.group(1).split('[')[0]
                version = match.group(2).strip()
                packages.append({"package": package, "version": version})
    return packages

data = {}
for file in Path('.').glob('requirements*.txt'):
    if file.name != 'requirements-lock.txt':
        data[file.name] = parse_requirements(file)

with open('requirements-export.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✓ Exported to requirements-export.json")
EOF

# Export to CSV
echo -e "${BLUE}Exporting to CSV...${NC}"
{
    echo "file,package,version"
    for file in requirements*.txt; do
        if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
            grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | while read -r line; do
                package=$(echo "$line" | sed 's/[>=<].*//' | sed 's/\[.*\]//')
                version=$(echo "$line" | sed 's/.*[>=<]//' | awk '{print $1}')
                echo "$file,$package,$version"
            done
        fi
    done
} > requirements-export.csv
echo -e "${GREEN}✓ Exported to requirements-export.csv${NC}"

# Export to Markdown table
echo -e "${BLUE}Exporting to Markdown...${NC}"
{
    echo "# Requirements Export"
    echo ""
    echo "Generated: $(date)"
    echo ""
    for file in requirements*.txt; do
        if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
            echo "## $file"
            echo ""
            echo "| Package | Version |"
            echo "|---------|---------|"
            grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | while read -r line; do
                package=$(echo "$line" | sed 's/[>=<].*//' | sed 's/\[.*\]//')
                version=$(echo "$line" | sed 's/.*[>=<]//' | awk '{print $1}')
                echo "| $package | $version |"
            done
            echo ""
        fi
    done
} > requirements-export.md
echo -e "${GREEN}✓ Exported to requirements-export.md${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Export completed${NC}"
echo "=========================================="
echo "Files created:"
echo "  - requirements-export.json"
echo "  - requirements-export.csv"
echo "  - requirements-export.md"



