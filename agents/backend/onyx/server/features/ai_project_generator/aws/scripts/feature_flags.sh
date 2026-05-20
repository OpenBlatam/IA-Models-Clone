#!/bin/bash
# Feature Flags Management Script
# Manages feature flags for gradual feature rollouts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_FLAGS_FILE="/opt/ai-project-generator/config/feature_flags.json"
FEATURE_FLAGS_DIR="/opt/ai-project-generator/config"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Initialize feature flags file
init_feature_flags() {
    sudo mkdir -p "$FEATURE_FLAGS_DIR"
    
    if [ ! -f "$FEATURE_FLAGS_FILE" ]; then
        sudo tee "$FEATURE_FLAGS_FILE" > /dev/null <<EOF
{
  "features": {},
  "version": "1.0",
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
        sudo chown ubuntu:ubuntu "$FEATURE_FLAGS_FILE"
        log_info "Feature flags file initialized"
    fi
}

# Enable feature flag
enable_feature() {
    local feature_name="$1"
    local percentage="${2:-100}"
    
    init_feature_flags
    
    log_info "Enabling feature '$feature_name' at $percentage%"
    
    # Use jq if available, otherwise use sed
    if command -v jq > /dev/null 2>&1; then
        sudo jq ".features[\"$feature_name\"] = {
            enabled: true,
            percentage: $percentage,
            updated_at: \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
        }" "$FEATURE_FLAGS_FILE" > "${FEATURE_FLAGS_FILE}.tmp" && \
        sudo mv "${FEATURE_FLAGS_FILE}.tmp" "$FEATURE_FLAGS_FILE"
    else
        log_warn "jq not available, using basic JSON update"
        # Basic JSON update (simplified)
        sudo python3 <<PYTHON
import json
import sys
from datetime import datetime

with open('$FEATURE_FLAGS_FILE', 'r') as f:
    data = json.load(f)

data['features']['$feature_name'] = {
    'enabled': True,
    'percentage': $percentage,
    'updated_at': datetime.utcnow().isoformat() + 'Z'
}
data['last_updated'] = datetime.utcnow().isoformat() + 'Z'

with open('$FEATURE_FLAGS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
PYTHON
    fi
    
    sudo chown ubuntu:ubuntu "$FEATURE_FLAGS_FILE"
    log_info "✅ Feature '$feature_name' enabled"
}

# Disable feature flag
disable_feature() {
    local feature_name="$1"
    
    init_feature_flags
    
    log_info "Disabling feature '$feature_name'"
    
    if command -v jq > /dev/null 2>&1; then
        sudo jq "del(.features[\"$feature_name\"])" "$FEATURE_FLAGS_FILE" > "${FEATURE_FLAGS_FILE}.tmp" && \
        sudo mv "${FEATURE_FLAGS_FILE}.tmp" "$FEATURE_FLAGS_FILE"
    else
        sudo python3 <<PYTHON
import json
from datetime import datetime

with open('$FEATURE_FLAGS_FILE', 'r') as f:
    data = json.load(f)

if '$feature_name' in data['features']:
    del data['features']['$feature_name']
    data['last_updated'] = datetime.utcnow().isoformat() + 'Z'

with open('$FEATURE_FLAGS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
PYTHON
    fi
    
    sudo chown ubuntu:ubuntu "$FEATURE_FLAGS_FILE"
    log_info "✅ Feature '$feature_name' disabled"
}

# List feature flags
list_features() {
    init_feature_flags
    
    if command -v jq > /dev/null 2>&1; then
        echo "Feature Flags:"
        sudo jq -r '.features | to_entries[] | "\(.key): \(if .value.enabled then "enabled" else "disabled" end) (\(.value.percentage // 0)%)"' "$FEATURE_FLAGS_FILE"
    else
        sudo python3 <<PYTHON
import json

with open('$FEATURE_FLAGS_FILE', 'r') as f:
    data = json.load(f)

print("Feature Flags:")
for name, config in data.get('features', {}).items():
    status = "enabled" if config.get('enabled', False) else "disabled"
    percentage = config.get('percentage', 0)
    print(f"{name}: {status} ({percentage}%)")
PYTHON
    fi
}

# Get feature flag status
get_feature_status() {
    local feature_name="$1"
    
    init_feature_flags
    
    if command -v jq > /dev/null 2>&1; then
        sudo jq -r ".features[\"$feature_name\"] // null" "$FEATURE_FLAGS_FILE"
    else
        sudo python3 <<PYTHON
import json
import sys

with open('$FEATURE_FLAGS_FILE', 'r') as f:
    data = json.load(f)

feature = data.get('features', {}).get('$feature_name')
if feature:
    print(json.dumps(feature, indent=2))
else:
    print("null")
PYTHON
    fi
}

# Reload application to pick up feature flags
reload_application() {
    log_info "Reloading application to apply feature flags..."
    
    if sudo systemctl is-active --quiet "ai-project-generator"; then
        sudo systemctl reload "ai-project-generator" || \
        sudo systemctl restart "ai-project-generator"
        log_info "✅ Application reloaded"
    else
        log_warn "Application service not running"
    fi
}

# Main function
main() {
    case "${1:-list}" in
        enable)
            if [ -z "${2:-}" ]; then
                echo "Usage: $0 enable <feature_name> [percentage]"
                exit 1
            fi
            enable_feature "${2}" "${3:-100}"
            reload_application
            ;;
        disable)
            if [ -z "${2:-}" ]; then
                echo "Usage: $0 disable <feature_name>"
                exit 1
            fi
            disable_feature "${2}"
            reload_application
            ;;
        list)
            list_features
            ;;
        status)
            if [ -z "${2:-}" ]; then
                echo "Usage: $0 status <feature_name>"
                exit 1
            fi
            get_feature_status "${2}"
            ;;
        *)
            echo "Usage: $0 {enable|disable|list|status} [feature_name] [percentage]"
            exit 1
            ;;
    esac
}

main "$@"

