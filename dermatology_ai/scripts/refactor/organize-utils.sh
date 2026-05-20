#!/bin/bash
# ============================================================================
# Organize Utils
# Organizes utility files into functional categories
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UTILS_DIR="$PROJECT_ROOT/utils"

echo "=========================================="
echo "Organizing Utils"
echo "=========================================="
echo ""

# Create util categories
echo -e "${BLUE}Creating util categories...${NC}"

categories=(
    "logging"
    "caching"
    "validation"
    "security"
    "performance"
    "database"
    "async"
    "monitoring"
    "helpers"
)

for category in "${categories[@]}"; do
    mkdir -p "$UTILS_DIR/$category"
    echo -e "  ${GREEN}✓${NC} Created $category/"
done
echo ""

# Categorize utils
echo -e "${BLUE}Categorizing utils...${NC}"

# Logging
logging_utils=(
    "logger.py"
    "advanced_logging.py"
)

# Caching
caching_utils=(
    "cache.py"
    "advanced_caching.py"
    "distributed_cache.py"
    "intelligent_cache.py"
)

# Validation
validation_utils=(
    "advanced_validator.py"
)

# Security
security_utils=(
    "oauth2.py"
    "security_headers.py"
)

# Performance
performance_utils=(
    "optimization.py"
    "advanced_optimization.py"
    "performance_profiler.py"
    "profiling.py"
    "model_pruning.py"
)

# Database
database_utils=(
    "database_abstraction.py"
    "connection_pool_manager.py"
)

# Async
async_utils=(
    "async_inference.py"
    "async_workers.py"
    "retry.py"
)

# Monitoring
monitoring_utils=(
    "observability.py"
    "circuit_breaker.py"
    "rate_limiter.py"
    "advanced_rate_limiter.py"
    "endpoint_rate_limiter.py"
)

# Helpers
helper_utils=(
    "exceptions.py"
    "api_gateway.py"
    "backup_recovery.py"
    "elasticsearch_client.py"
    "message_broker.py"
    "service_discovery.py"
    "service_mesh.py"
)

# Function to move utils
move_utils() {
    category=$1
    shift
    utils=("$@")
    
    for util in "${utils[@]}"; do
        if [ -f "$UTILS_DIR/$util" ]; then
            cp "$UTILS_DIR/$util" "$UTILS_DIR/$category/" 2>/dev/null || true
            echo -e "  ${GREEN}✓${NC} $util → $category/"
        fi
    done
}

# Move utils to categories
move_utils "logging" "${logging_utils[@]}"
move_utils "caching" "${caching_utils[@]}"
move_utils "validation" "${validation_utils[@]}"
move_utils "security" "${security_utils[@]}"
move_utils "performance" "${performance_utils[@]}"
move_utils "database" "${database_utils[@]}"
move_utils "async" "${async_utils[@]}"
move_utils "monitoring" "${monitoring_utils[@]}"
move_utils "helpers" "${helper_utils[@]}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Utils organized${NC}"
echo "=========================================="
echo ""
echo "Note: Original files remain in utils/ for compatibility"
echo "New organized structure is in utils/{category}/"



