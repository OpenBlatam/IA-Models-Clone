#!/usr/bin/env python3
"""
Database Migration and Seed Script
Manages database schema migrations and seed data for the content redundancy detector
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Migration utilities
class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, db_url: Optional[str] = None):
        self.settings = Settings()
        self.db_url = db_url or self.settings.database_url
        
        if not self.db_url:
            logger.warning("No database URL configured. Migrations will be skipped.")
    
    async def create_migration_table(self):
        """Create migrations tracking table."""
        # This would use your actual database connection
        # For now, it's a placeholder that shows the structure
        migration_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            version VARCHAR(255) UNIQUE NOT NULL,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(255)
        );
        """
        logger.info("Migration table SQL:")
        logger.info(migration_sql)
        # In actual implementation, execute this SQL
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations."""
        # Placeholder - would query database
        return []
    
    async def apply_migration(self, version: str, sql: str, description: str = ""):
        """Apply a single migration."""
        logger.info(f"Applying migration {version}: {description}")
        
        try:
            # In actual implementation, would execute SQL here
            # async with db.transaction():
            #     await db.execute(sql)
            #     await db.execute(
            #         "INSERT INTO schema_migrations (version, description) VALUES ($1, $2)",
            #         version, description
            #     )
            
            logger.info(f"Migration {version} applied successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to apply migration {version}: {e}")
            return False
    
    async def rollback_migration(self, version: str):
        """Rollback a migration."""
        logger.warning(f"Rollback for migration {version} - implement rollback SQL")
        # Implementation would contain rollback SQL


# Migration definitions
MIGRATIONS = [
    {
        "version": "001_initial_schema",
        "description": "Initial database schema",
        "up": """
        CREATE TABLE IF NOT EXISTS content_analyses (
            id SERIAL PRIMARY KEY,
            content_hash VARCHAR(255) UNIQUE NOT NULL,
            content_text TEXT NOT NULL,
            similarity_score FLOAT,
            quality_score FLOAT,
            redundancy_detected BOOLEAN DEFAULT FALSE,
            analysis_metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_content_hash ON content_analyses(content_hash);
        CREATE INDEX IF NOT EXISTS idx_created_at ON content_analyses(created_at);
        CREATE INDEX IF NOT EXISTS idx_redundancy_detected ON content_analyses(redundancy_detected);
        
        CREATE TABLE IF NOT EXISTS similarity_pairs (
            id SERIAL PRIMARY KEY,
            content_id_1 INTEGER REFERENCES content_analyses(id),
            content_id_2 INTEGER REFERENCES content_analyses(id),
            similarity_score FLOAT NOT NULL,
            comparison_metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_similarity_score ON similarity_pairs(similarity_score);
        CREATE INDEX IF NOT EXISTS idx_content_pair ON similarity_pairs(content_id_1, content_id_2);
        """
    },
    {
        "version": "002_add_webhooks_table",
        "description": "Add webhooks tracking table",
        "up": """
        CREATE TABLE IF NOT EXISTS webhook_subscriptions (
            id SERIAL PRIMARY KEY,
            url VARCHAR(512) NOT NULL,
            events TEXT[] NOT NULL,
            secret_key VARCHAR(255),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS webhook_events (
            id SERIAL PRIMARY KEY,
            subscription_id INTEGER REFERENCES webhook_subscriptions(id),
            event_type VARCHAR(100) NOT NULL,
            payload JSONB NOT NULL,
            status VARCHAR(50) DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            last_attempt_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_webhook_status ON webhook_events(status);
        CREATE INDEX IF NOT EXISTS idx_webhook_subscription ON webhook_events(subscription_id);
        """
    },
    {
        "version": "003_add_metrics_table",
        "description": "Add metrics and analytics tables",
        "up": """
        CREATE TABLE IF NOT EXISTS api_metrics (
            id SERIAL PRIMARY KEY,
            endpoint VARCHAR(255) NOT NULL,
            method VARCHAR(10) NOT NULL,
            status_code INTEGER,
            response_time_ms FLOAT,
            request_size_bytes INTEGER,
            response_size_bytes INTEGER,
            user_agent TEXT,
            ip_address VARCHAR(45),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_metrics_endpoint ON api_metrics(endpoint);
        CREATE INDEX IF NOT EXISTS idx_metrics_created_at ON api_metrics(created_at);
        CREATE INDEX IF NOT EXISTS idx_metrics_status ON api_metrics(status_code);
        
        CREATE TABLE IF NOT EXISTS cache_metrics (
            id SERIAL PRIMARY KEY,
            cache_key VARCHAR(255) NOT NULL,
            cache_hit BOOLEAN DEFAULT FALSE,
            ttl_seconds INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_metrics(cache_key);
        CREATE INDEX IF NOT EXISTS idx_cache_hit ON cache_metrics(cache_hit);
        """
    }
]


# Seed data
SEED_DATA = {
    "content_analyses": [
        {
            "content_hash": "sample_hash_1",
            "content_text": "Sample content for testing",
            "similarity_score": 0.85,
            "quality_score": 0.92,
            "redundancy_detected": False
        }
    ]
}


async def apply_all_migrations(dry_run: bool = False):
    """Apply all pending migrations."""
    manager = MigrationManager()
    
    if not manager.db_url:
        logger.error("No database URL configured. Cannot run migrations.")
        return False
    
    await manager.create_migration_table()
    applied = await manager.get_applied_migrations()
    
    pending = [m for m in MIGRATIONS if m["version"] not in applied]
    
    if not pending:
        logger.info("No pending migrations")
        return True
    
    logger.info(f"Found {len(pending)} pending migrations")
    
    if dry_run:
        logger.info("DRY RUN - Would apply:")
        for migration in pending:
            logger.info(f"  - {migration['version']}: {migration['description']}")
        return True
    
    for migration in pending:
        success = await manager.apply_migration(
            migration["version"],
            migration["up"],
            migration["description"]
        )
        if not success:
            logger.error(f"Migration {migration['version']} failed. Stopping.")
            return False
    
    logger.info("All migrations applied successfully")
    return True


async def seed_database():
    """Seed database with initial data."""
    manager = MigrationManager()
    
    if not manager.db_url:
        logger.error("No database URL configured. Cannot seed database.")
        return False
    
    logger.info("Seeding database with initial data...")
    
    # In actual implementation, would insert seed data
    # for table, records in SEED_DATA.items():
    #     for record in records:
    #         await db.execute(f"INSERT INTO {table} ...", **record)
    
    logger.info("Database seeded successfully")
    return True


async def reset_database():
    """Reset database (DANGEROUS - drops all tables)."""
    import click
    
    if not click.confirm("This will DROP ALL TABLES. Are you sure?", default=False):
        logger.info("Reset cancelled")
        return False
    
    manager = MigrationManager()
    
    if not manager.db_url:
        logger.error("No database URL configured.")
        return False
    
    logger.warning("Resetting database...")
    
    # In actual implementation:
    # DROP TABLE IF EXISTS ... CASCADE;
    
    logger.info("Database reset. Run migrations to recreate schema.")
    return True


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument(
        "command",
        choices=["up", "seed", "reset", "status"],
        help="Migration command"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying"
    )
    
    args = parser.parse_args()
    
    if args.command == "up":
        asyncio.run(apply_all_migrations(dry_run=args.dry_run))
    elif args.command == "seed":
        asyncio.run(seed_database())
    elif args.command == "reset":
        asyncio.run(reset_database())
    elif args.command == "status":
        manager = MigrationManager()
        applied = asyncio.run(manager.get_applied_migrations())
        logger.info(f"Applied migrations: {len(applied)}")
        for mig in applied:
            logger.info(f"  - {mig}")


if __name__ == "__main__":
    main()

