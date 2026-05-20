"""
Version handler for MCP Server
==============================

Handler para obtener información de versión del servidor.
"""

import logging
from typing import Dict, Any

from .. import __version__, __author__, __license__

logger = logging.getLogger(__name__)


async def get_version() -> Dict[str, Any]:
    """
    Obtener información de versión del servidor MCP.
    
    Returns:
        Diccionario con información de versión
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "protocol": "MCP v1",
        "api_version": "v1",
    }

