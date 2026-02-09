#!/usr/bin/env python3
"""
Script de inicio simple - Cursor Agent 24/7
============================================

Uso:
    python run.py                    # Iniciar API en puerto 8024
    python run.py --port 8080       # Puerto personalizado
    python run.py --aws              # Habilitar AWS
    python run.py --mode service    # Modo servicio
"""

import sys
from pathlib import Path

# Agregar directorio al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar y ejecutar CLI
from cli import main

if __name__ == "__main__":
    main()




