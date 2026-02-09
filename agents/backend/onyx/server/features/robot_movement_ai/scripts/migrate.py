#!/usr/bin/env python3
"""
Script de migraciones de base de datos
"""

import sys
import sqlite3
import argparse
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.migrations.migration_manager import MigrationManager
from db.migrations.migrations import MIGRATIONS


def main():
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument(
        "command",
        choices=["migrate", "rollback", "status"],
        help="Migration command"
    )
    parser.add_argument(
        "--db",
        default="db/robots.db",
        help="Database file path"
    )
    parser.add_argument(
        "--version",
        help="Target version for migrate/rollback"
    )
    
    args = parser.parse_args()
    
    # Crear directorio de BD si no existe
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Conectar a base de datos
    db = sqlite3.connect(str(db_path))
    
    # Crear gestor de migraciones
    manager = MigrationManager(db)
    
    # Registrar migraciones
    for migration in MIGRATIONS:
        manager.register_migration(migration)
    
    # Ejecutar comando
    if args.command == "migrate":
        manager.migrate(target_version=args.version)
    elif args.command == "rollback":
        manager.rollback(target_version=args.version)
    elif args.command == "status":
        manager.status()
    
    db.close()


if __name__ == "__main__":
    main()




