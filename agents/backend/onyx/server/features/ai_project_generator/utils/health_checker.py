"""
Health Checker - Verificador de Salud Avanzado
==============================================

Verifica la salud del sistema de forma avanzada.
"""

import logging
import asyncio
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedHealthChecker:
    """Verificador de salud avanzado"""

    def __init__(self):
        """Inicializa el verificador de salud"""
        pass

    async def check_health(self) -> Dict[str, Any]:
        """
        Verifica la salud completa del sistema.

        Returns:
            Estado de salud del sistema
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
        }

        # Check de sistema de archivos
        health_status["checks"]["filesystem"] = await self._check_filesystem()

        # Check de memoria
        health_status["checks"]["memory"] = await self._check_memory()

        # Check de disco
        health_status["checks"]["disk"] = await self._check_disk()

        # Determinar estado general
        all_healthy = all(
            check.get("status") == "healthy"
            for check in health_status["checks"].values()
        )

        health_status["status"] = "healthy" if all_healthy else "degraded"

        return health_status

    async def _check_filesystem(self) -> Dict[str, Any]:
        """Verifica el sistema de archivos"""
        try:
            test_file = Path("/tmp/health_check_test")
            test_file.write_text("test")
            test_file.unlink()

            return {
                "status": "healthy",
                "message": "Sistema de archivos operativo",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Error en sistema de archivos: {e}",
            }

    async def _check_memory(self) -> Dict[str, Any]:
        """Verifica la memoria"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "status": "healthy" if memory.percent < 90 else "warning",
                "message": f"Memoria: {{memory.percent}}% usado",
                "percent_used": memory.percent,
                "available_mb": round(memory.available / (1024 * 1024), 2),
            }
        except ImportError:
            return {
                "status": "unknown",
                "message": "psutil no disponible",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Error verificando memoria: {e}",
            }

    async def _check_disk(self) -> Dict[str, Any]:
        """Verifica el espacio en disco"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            percent_used = (used / total) * 100

            return {
                "status": "healthy" if percent_used < 90 else "warning",
                "message": f"Disco: {{percent_used:.1f}}% usado",
                "percent_used": round(percent_used, 2),
                "free_gb": round(free / (1024 ** 3), 2),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Error verificando disco: {e}",
            }

    async def check_dependencies(self) -> Dict[str, Any]:
        """
        Verifica dependencias del sistema.

        Returns:
            Estado de dependencias
        """
        dependencies = {
            "python": {
                "required": True,
                "installed": True,
                "version": "3.11+",
            },
            "fastapi": {
                "required": True,
                "installed": self._check_module("fastapi"),
            },
            "httpx": {
                "required": True,
                "installed": self._check_module("httpx"),
            },
        }

        all_installed = all(
            dep.get("installed", False) for dep in dependencies.values()
        )

        return {
            "status": "healthy" if all_installed else "unhealthy",
            "dependencies": dependencies,
        }

    def _check_module(self, module_name: str) -> bool:
        """Verifica si un módulo está instalado"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False


