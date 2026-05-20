#!/usr/bin/env python3
"""
Script para ejecutar el servidor API de Faceless Video AI
"""

import uvicorn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from api.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,  # Desactivar en producción
    )

