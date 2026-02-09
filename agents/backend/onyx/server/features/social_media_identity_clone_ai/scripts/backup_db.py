"""
Script para hacer backup de la base de datos
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings

def main():
    """Crea backup de la base de datos"""
    settings = get_settings()
    
    # Obtener path de la base de datos
    db_url = settings.database_url
    if db_url.startswith("sqlite:///"):
        db_path = Path(db_url.replace("sqlite:///", ""))
    else:
        print("❌ Solo se soportan backups de SQLite")
        sys.exit(1)
    
    if not db_path.exists():
        print(f"❌ Base de datos no encontrada: {db_path}")
        sys.exit(1)
    
    # Crear directorio de backups
    backup_dir = Path(settings.storage_path) / "backups" / "database"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear nombre de backup
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"db_backup_{timestamp}.db"
    
    print(f"📦 Creando backup de {db_path}...")
    
    try:
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup creado: {backup_path}")
        print(f"   Tamaño: {backup_path.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ Error creando backup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()




