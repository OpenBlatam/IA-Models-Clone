#!/usr/bin/env python3
"""
Migration Script
================

Script para ejecutar migraciones de base de datos.
"""

import sys
import os
from pathlib import Path

# Agregar ruta del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.migrations import MigrationService, Migration
from datetime import datetime


def create_initial_migration():
    """Crear migración inicial."""
    return Migration(
        version="001_initial",
        description="Initial database schema",
        up_sql="""
        -- Esta migración ya está aplicada por el DatabaseService
        -- Se mantiene para referencia
        """,
        down_sql="""
        -- Rollback no aplicable para migración inicial
        """
    )


def main():
    """Ejecutar migraciones."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument("--db-path", default="artist_manager.db", help="Database path")
    parser.add_argument("--action", choices=["up", "down", "status"], default="status", help="Action to perform")
    parser.add_argument("--version", help="Migration version (for rollback)")
    
    args = parser.parse_args()
    
    migration_service = MigrationService(args.db_path)
    
    if args.action == "status":
        applied = migration_service.get_applied_migrations()
        print(f"Applied migrations: {len(applied)}")
        for version in applied:
            print(f"  - {version}")
    
    elif args.action == "up":
        migration = create_initial_migration()
        if migration_service.apply_migration(migration):
            print(f"✓ Applied migration: {migration.version}")
        else:
            print(f"✗ Failed to apply migration: {migration.version}")
    
    elif args.action == "down":
        if not args.version:
            print("Error: --version required for rollback")
            sys.exit(1)
        
        migration = create_initial_migration()
        if migration.version == args.version:
            if migration_service.rollback_migration(migration):
                print(f"✓ Rolled back migration: {migration.version}")
            else:
                print(f"✗ Failed to rollback migration: {migration.version}")


if __name__ == "__main__":
    main()




