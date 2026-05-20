"""
Script para iniciar el servicio como daemon.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import run_service

if __name__ == "__main__":
    asyncio.run(run_service())




