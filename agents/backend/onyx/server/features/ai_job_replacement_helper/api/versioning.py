"""
API Versioning
==============

Sistema de versionado de API.
"""

from fastapi import APIRouter, Request
from typing import Callable


class APIVersion:
    """Gestor de versiones de API"""
    
    def __init__(self):
        self.versions = {}
    
    def register_version(self, version: str, router: APIRouter):
        """Registrar versión de API"""
        self.versions[version] = router
    
    def get_router(self, version: str) -> APIRouter:
        """Obtener router de versión específica"""
        return self.versions.get(version)
    
    def get_latest_version(self) -> str:
        """Obtener última versión"""
        if not self.versions:
            return "v1"
        return max(self.versions.keys())


# Instancia global
api_version = APIVersion()


def version_router(version: str):
    """Decorator para crear router versionado"""
    router = APIRouter(prefix=f"/api/{version}")
    api_version.register_version(version, router)
    return router




