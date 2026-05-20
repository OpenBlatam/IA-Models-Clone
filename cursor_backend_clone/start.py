"""
Script de inicio rápido para Cursor Agent 24/7
"""

import sys
import asyncio
from pathlib import Path

# Agregar el directorio al path
sys.path.insert(0, str(Path(__file__).parent))

from main import main

if __name__ == "__main__":
    main()


