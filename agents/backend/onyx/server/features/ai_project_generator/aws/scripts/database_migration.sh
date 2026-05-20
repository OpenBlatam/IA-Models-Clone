#!/bin/bash
# Database Migration Script
# Handles database migrations safely with rollback support

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS_DIR="/opt/ai-project-generator/migrations"
BACKUP_DIR="/opt/backups/database"
DB_TYPE="${DB_TYPE:-postgresql}"  # postgresql, mysql, sqlite

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Create database backup
backup_database() {
    local backup_file="$BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql"
    
    log_info "Creating database backup..."
    sudo mkdir -p "$BACKUP_DIR"
    
    case "$DB_TYPE" in
        postgresql)
            sudo -u postgres pg_dumpall > "$backup_file" 2>/dev/null || {
                log_error "PostgreSQL backup failed"
                return 1
            }
            ;;
        mysql)
            mysqldump --all-databases > "$backup_file" 2>/dev/null || {
                log_error "MySQL backup failed"
                return 1
            }
            ;;
        sqlite)
            if [ -f "$DB_FILE" ]; then
                cp "$DB_FILE" "$backup_file"
            else
                log_error "SQLite database file not found"
                return 1
            fi
            ;;
        *)
            log_error "Unsupported database type: $DB_TYPE"
            return 1
            ;;
    esac
    
    log_info "✅ Database backup created: $backup_file"
    echo "$backup_file"
}

# Run migrations
run_migrations() {
    local migration_file="${1:-}"
    
    if [ -z "$migration_file" ]; then
        log_info "Running all pending migrations..."
        
        if [ ! -d "$MIGRATIONS_DIR" ]; then
            log_warn "Migrations directory not found: $MIGRATIONS_DIR"
            return 0
        fi
        
        # Find and run migration files
        for migration in "$MIGRATIONS_DIR"/*.sql; do
            if [ -f "$migration" ]; then
                run_single_migration "$migration"
            fi
        done
        
        # Run Alembic migrations if available
        if command -v alembic > /dev/null 2>&1; then
            log_info "Running Alembic migrations..."
            cd /opt/ai-project-generator
            alembic upgrade head || {
                log_error "Alembic migration failed"
                return 1
            }
        fi
    else
        run_single_migration "$migration_file"
    fi
    
    log_info "✅ Migrations completed"
}

# Run single migration
run_single_migration() {
    local migration_file="$1"
    
    log_info "Running migration: $(basename "$migration_file")"
    
    case "$DB_TYPE" in
        postgresql)
            sudo -u postgres psql < "$migration_file" || {
                log_error "Migration failed: $migration_file"
                return 1
            }
            ;;
        mysql)
            mysql < "$migration_file" || {
                log_error "Migration failed: $migration_file"
                return 1
            }
            ;;
        sqlite)
            sqlite3 "$DB_FILE" < "$migration_file" || {
                log_error "Migration failed: $migration_file"
                return 1
            }
            ;;
    esac
    
    log_info "✅ Migration completed: $(basename "$migration_file")"
}

# Rollback migration
rollback_migration() {
    local backup_file="${1:-}"
    
    if [ -z "$backup_file" ]; then
        # Find latest backup
        backup_file=$(ls -t "$BACKUP_DIR"/db_backup_*.sql 2>/dev/null | head -1)
    fi
    
    if [ -z "$backup_file" ] || [ ! -f "$backup_file" ]; then
        log_error "Backup file not found"
        return 1
    fi
    
    log_warn "Rolling back to: $backup_file"
    
    case "$DB_TYPE" in
        postgresql)
            sudo -u postgres psql < "$backup_file" || {
                log_error "Rollback failed"
                return 1
            }
            ;;
        mysql)
            mysql < "$backup_file" || {
                log_error "Rollback failed"
                return 1
            }
            ;;
        sqlite)
            cp "$backup_file" "$DB_FILE" || {
                log_error "Rollback failed"
                return 1
            }
            ;;
    esac
    
    log_info "✅ Database rolled back successfully"
}

# Verify migration
verify_migration() {
    log_info "Verifying migration..."
    
    # Check database connectivity
    case "$DB_TYPE" in
        postgresql)
            sudo -u postgres psql -c "SELECT version();" > /dev/null 2>&1 || {
                log_error "Database connection failed"
                return 1
            }
            ;;
        mysql)
            mysql -e "SELECT VERSION();" > /dev/null 2>&1 || {
                log_error "Database connection failed"
                return 1
            }
            ;;
        sqlite)
            if [ ! -f "$DB_FILE" ]; then
                log_error "Database file not found"
                return 1
            fi
            ;;
    esac
    
    log_info "✅ Migration verification passed"
    return 0
}

# Main function
main() {
    case "${1:-migrate}" in
        migrate)
            local backup_file=$(backup_database)
            if run_migrations "${2:-}"; then
                verify_migration
                log_info "✅ Migration completed successfully"
            else
                log_error "Migration failed, rolling back..."
                rollback_migration "$backup_file"
                exit 1
            fi
            ;;
        rollback)
            rollback_migration "${2:-}"
            ;;
        backup)
            backup_database
            ;;
        verify)
            verify_migration
            ;;
        *)
            echo "Usage: $0 {migrate|rollback|backup|verify} [migration_file|backup_file]"
            exit 1
            ;;
    esac
}

main "$@"

