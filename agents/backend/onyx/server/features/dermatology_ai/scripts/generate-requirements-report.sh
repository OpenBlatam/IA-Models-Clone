#!/bin/bash
# ============================================================================
# Generate Comprehensive Requirements Report
# Creates a detailed markdown report of all dependencies
# ============================================================================

set -e

REPORT_FILE="REQUIREMENTS_REPORT.md"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

echo "Generating requirements report..."

cat > "$REPORT_FILE" << EOF
# Requirements Report

**Generated:** $TIMESTAMP

## Overview

This report provides a comprehensive analysis of all dependency files in the project.

## Files Summary

EOF

# Count packages in each file
for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        count=$(grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | wc -l)
        size=$(wc -l < "$file")
        echo "### $file" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "- **Packages:** $count" >> "$REPORT_FILE"
        echo "- **Total Lines:** $size" >> "$REPORT_FILE"
        echo "- **Estimated Size:** $(du -h "$file" 2>/dev/null | cut -f1 || echo 'N/A')" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done

# Add package lists
cat >> "$REPORT_FILE" << EOF
## Package Lists

EOF

for file in requirements*.txt; do
    if [ -f "$file" ] && [ "$file" != "requirements-lock.txt" ]; then
        echo "### $file" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        grep -E "^[a-zA-Z]" "$file" | grep -v "^#" | head -50 >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done

# Add security check if available
if command -v safety &> /dev/null; then
    cat >> "$REPORT_FILE" << EOF
## Security Check

\`\`\`
$(safety check -r requirements.txt --short-report 2>&1 || echo "No issues found")
\`\`\`

EOF
fi

# Add outdated packages
cat >> "$REPORT_FILE" << EOF
## Outdated Packages

\`\`\`
$(pip list --outdated 2>/dev/null | head -20 || echo "Unable to check")
\`\`\`

EOF

echo "✅ Report generated: $REPORT_FILE"



