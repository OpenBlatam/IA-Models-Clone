"""
Init Database Script - Inicializar Base de Datos
================================================

Script para inicializar la base de datos.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from community_manager_ai.database import init_db

if __name__ == "__main__":
    print("Inicializando base de datos...")
    init_db()
    print("Base de datos inicializada exitosamente!")




