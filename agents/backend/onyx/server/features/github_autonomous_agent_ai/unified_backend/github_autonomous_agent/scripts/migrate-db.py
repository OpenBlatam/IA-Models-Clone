#!/usr/bin/env python3
"""
Script para ejecutar migraciones de base de datos.
"""

import sys
import subprocess
from pathlib import Path

def run_migration(command: str = "upgrade head"):
    """Ejecuta migraciones de Alembic."""
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print(f"🔄 Ejecutando migración: {command}")
    
    try:
        # Verificar que alembic está instalado
        result = subprocess.run(
            ["python", "-c", "import alembic"],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError:
        print("❌ Alembic no está instalado")
        print("💡 Instala con: pip install alembic")
        sys.exit(1)
    
    # Ejecutar migración
    try:
        if command == "init":
            subprocess.run(["alembic", "init", "alembic"], check=True)
            print("✅ Directorio de migraciones creado")
        elif command == "create":
            migration_name = input("Nombre de la migración: ")
            subprocess.run(["alembic", "revision", "--autogenerate", "-m", migration_name], check=True)
            print(f"✅ Migración '{migration_name}' creada")
        elif command == "upgrade":
            subprocess.run(["alembic", "upgrade", "head"], check=True)
            print("✅ Base de datos actualizada")
        elif command == "downgrade":
            revision = input("Revisión a la que hacer downgrade (o 'head'): ")
            subprocess.run(["alembic", "downgrade", revision], check=True)
            print(f"✅ Base de datos revertida a {revision}")
        elif command == "history":
            subprocess.run(["alembic", "history"], check=True)
        elif command == "current":
            subprocess.run(["alembic", "current"], check=True)
        else:
            print(f"❌ Comando desconocido: {command}")
            print_usage()
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando migración: {e}")
        sys.exit(1)

def print_usage():
    """Muestra uso del script."""
    print("""
Uso: python scripts/migrate-db.py [comando]

Comandos disponibles:
  init          - Inicializar directorio de migraciones
  create        - Crear nueva migración (autogenerate)
  upgrade       - Aplicar todas las migraciones pendientes
  downgrade     - Revertir migraciones
  history       - Ver historial de migraciones
  current       - Ver migración actual

Ejemplos:
  python scripts/migrate-db.py upgrade
  python scripts/migrate-db.py create
  python scripts/migrate-db.py history
    """)

def main():
    """Función principal."""
    import os
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    run_migration(command)

if __name__ == "__main__":
    main()




