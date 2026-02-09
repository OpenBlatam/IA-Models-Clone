#!/bin/bash
# ============================================================================
# Organize Configuration Files
# Organizes configuration files into structured directories
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"

echo "=========================================="
echo "Organizing Configuration Files"
echo "=========================================="
echo ""

# Create config structure
mkdir -p "$CONFIG_DIR"/{environments,schemas,models}

echo -e "${BLUE}Creating environment configurations...${NC}"

# Create development config
cat > "$CONFIG_DIR/environments/development.yaml" << 'EOF'
# Development Environment Configuration
environment: development
debug: true
log_level: DEBUG
database:
  host: localhost
  port: 5432
  name: dermatology_dev
cache:
  enabled: true
  ttl: 300
EOF

# Create production config
cat > "$CONFIG_DIR/environments/production.yaml" << 'EOF'
# Production Environment Configuration
environment: production
debug: false
log_level: INFO
database:
  host: ${DB_HOST}
  port: ${DB_PORT}
  name: ${DB_NAME}
cache:
  enabled: true
  ttl: 3600
EOF

# Create testing config
cat > "$CONFIG_DIR/environments/testing.yaml" << 'EOF'
# Testing Environment Configuration
environment: testing
debug: true
log_level: DEBUG
database:
  host: localhost
  port: 5432
  name: dermatology_test
cache:
  enabled: false
EOF

echo -e "  ${GREEN}✓${NC} Created environment configurations"
echo ""

echo "=========================================="
echo -e "${GREEN}✓ Configuration organized${NC}"
echo "=========================================="



