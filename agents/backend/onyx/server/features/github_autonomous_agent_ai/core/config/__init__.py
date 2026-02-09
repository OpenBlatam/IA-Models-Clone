"""
Módulo de configuración
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings, Settings

__all__ = ["settings", "Settings"]

