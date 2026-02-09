"""
Script para inicializar la base de datos
=========================================

Crea las tablas iniciales usando Alembic.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alembic import command
from alembic.config import Config


def init_database():
    """Inicializar base de datos con Alembic."""
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    
    print("🚀 Inicializando base de datos...")
    print("=" * 60)
    
    try:
        # Crear migración inicial si no existe
        print("📝 Creando migración inicial...")
        command.revision(
            alembic_cfg,
            autogenerate=True,
            message="Initial migration"
        )
        
        print("✅ Migración creada exitosamente")
        print()
        
        # Aplicar migraciones
        print("📊 Aplicando migraciones...")
        command.upgrade(alembic_cfg, "head")
        
        print("✅ Base de datos inicializada exitosamente")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print()
        print("💡 Asegúrate de que:")
        print("   1. PostgreSQL está corriendo")
        print("   2. Las variables de entorno están configuradas")
        print("   3. La base de datos existe")
        sys.exit(1)


if __name__ == "__main__":
    init_database()




