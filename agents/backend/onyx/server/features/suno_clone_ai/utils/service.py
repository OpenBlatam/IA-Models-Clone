"""
Util Service - Servicio de utilidades
"""

from typing import Dict, Any, Optional
from .base import BaseUtil


class UtilService:
    """Servicio centralizado de utilidades"""

    def __init__(self):
        """Inicializa el servicio de utilidades"""
        self._utils: Dict[str, BaseUtil] = {}

    def register_util(self, name: str, util: BaseUtil) -> None:
        """Registra una utilidad"""
        self._utils[name] = util

    def get_util(self, name: str) -> Optional[BaseUtil]:
        """Obtiene una utilidad por nombre"""
        return self._utils.get(name)

