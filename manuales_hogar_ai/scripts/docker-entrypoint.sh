#!/bin/bash
# Docker entrypoint script

set -e

# Wait for database to be ready
if [ -n "$DB_HOST" ]; then
    echo "Waiting for database to be ready..."
    until pg_isready -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}"; do
        echo "Database is unavailable - sleeping"
        sleep 1
    done
    echo "Database is ready!"
fi

# Run database migrations
if [ "$RUN_MIGRATIONS" != "false" ]; then
    echo "Running database migrations..."
    alembic upgrade head || echo "Migration failed, continuing anyway..."
fi

# Execute the main command
exec "$@"




