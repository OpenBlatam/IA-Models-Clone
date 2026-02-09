"""
Script para migraciones de base de datos
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.base import get_db_session, engine
from sqlalchemy import text

def create_migration_table():
    """Crea tabla de migraciones si no existe"""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

def get_applied_migrations():
    """Obtiene migraciones aplicadas"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM migrations ORDER BY applied_at"))
            return [row[0] for row in result]
    except:
        return []

def apply_migration(name: str, sql: str):
    """Aplica una migración"""
    applied = get_applied_migrations()
    if name in applied:
        print(f"⏭️  Migración {name} ya aplicada, saltando...")
        return
    
    print(f"🔄 Aplicando migración: {name}")
    
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.execute(text(
                "INSERT INTO migrations (name) VALUES (:name)",
                {"name": name}
            ))
            conn.commit()
        print(f"✅ Migración {name} aplicada correctamente")
    except Exception as e:
        print(f"❌ Error aplicando migración {name}: {e}")
        raise

def main():
    """Ejecuta migraciones pendientes"""
    print("Ejecutando migraciones...")
    
    create_migration_table()
    
    # Ejemplo de migración
    migrations = [
        {
            "name": "001_initial_schema",
            "sql": """
                -- Esta migración ya está aplicada por init_db
                -- Aquí puedes agregar migraciones futuras
            """
        }
    ]
    
    for migration in migrations:
        apply_migration(migration["name"], migration["sql"])
    
    print("\n✅ Todas las migraciones aplicadas")

if __name__ == "__main__":
    main()




