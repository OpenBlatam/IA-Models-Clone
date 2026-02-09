"""
Script para inicializar la base de datos
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.base import init_db, engine
from db.models import Base

def main():
    """Inicializa la base de datos"""
    print("Inicializando base de datos...")
    
    try:
        # Crear todas las tablas
        Base.metadata.create_all(bind=engine)
        print("✅ Base de datos inicializada correctamente")
        
        # Mostrar tablas creadas
        print("\nTablas creadas:")
        for table_name in Base.metadata.tables.keys():
            print(f"  - {table_name}")
            
    except Exception as e:
        print(f"❌ Error inicializando base de datos: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()




