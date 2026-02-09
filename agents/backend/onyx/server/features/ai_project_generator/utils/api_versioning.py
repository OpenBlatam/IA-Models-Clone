"""
API Versioning - Gestión de Versiones de API
=============================================

Gestiona versionado de la API.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter
from datetime import datetime

logger = logging.getLogger(__name__)


class APIVersionManager:
    """Gestor de versionado de API"""

    def __init__(self):
        """Inicializa el gestor de versionado"""
        self.current_version = "v1"
        self.supported_versions = ["v1"]
        self.deprecated_versions = []

    def get_version_info(self) -> Dict[str, Any]:
        """
        Obtiene información de versiones.

        Returns:
            Información de versiones
        """
        return {
            "current_version": self.current_version,
            "supported_versions": self.supported_versions,
            "deprecated_versions": self.deprecated_versions,
            "api_base_url": f"/api/{self.current_version}",
        }

    def create_versioned_router(self, version: str) -> APIRouter:
        """
        Crea un router versionado.

        Args:
            version: Versión de la API

        Returns:
            Router versionado
        """
        if version not in self.supported_versions:
            raise ValueError(f"Versión {version} no soportada")

        router = APIRouter(
            prefix=f"/api/{version}",
            tags=[f"API {version}"],
        )

        return router

    def mark_deprecated(self, version: str, deprecation_date: str):
        """
        Marca una versión como deprecada.

        Args:
            version: Versión a deprecar
            deprecation_date: Fecha de deprecación
        """
        if version in self.supported_versions:
            self.supported_versions.remove(version)
        self.deprecated_versions.append({
            "version": version,
            "deprecation_date": deprecation_date,
        })
        logger.info(f"Versión {version} marcada como deprecada")


