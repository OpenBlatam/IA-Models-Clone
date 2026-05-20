"""
Advanced Testing - Testing Avanzado
===================================

Sistema avanzado de testing para proyectos generados.
"""

import logging
import subprocess
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedTesting:
    """Sistema avanzado de testing"""

    def __init__(self):
        """Inicializa el sistema de testing"""
        self.test_results: Dict[str, Dict[str, Any]] = {}

    async def run_backend_tests(
        self,
        project_path: Path,
    ) -> Dict[str, Any]:
        """
        Ejecuta tests del backend.

        Args:
            project_path: Ruta del proyecto

        Returns:
            Resultados de los tests
        """
        backend_path = project_path / "backend"
        if not backend_path.exists():
            return {
                "success": False,
                "error": "Backend no encontrado",
            }

        try:
            # Ejecutar pytest
            result = subprocess.run(
                ["pytest", "-v", "--json-report", "--json-report-file=test_results.json"],
                cwd=backend_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Leer resultados JSON si existe
            test_results_file = backend_path / "test_results.json"
            test_data = {}
            if test_results_file.exists():
                test_data = json.loads(test_results_file.read_text(encoding="utf-8"))

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "tests_passed": test_data.get("summary", {}).get("passed", 0),
                "tests_failed": test_data.get("summary", {}).get("failed", 0),
                "tests_total": test_data.get("summary", {}).get("total", 0),
                "output": result.stdout,
                "errors": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Tests timeout después de 5 minutos",
            }
        except Exception as e:
            logger.error(f"Error ejecutando tests: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def run_frontend_tests(
        self,
        project_path: Path,
    ) -> Dict[str, Any]:
        """
        Ejecuta tests del frontend.

        Args:
            project_path: Ruta del proyecto

        Returns:
            Resultados de los tests
        """
        frontend_path = project_path / "frontend"
        if not frontend_path.exists():
            return {
                "success": False,
                "error": "Frontend no encontrado",
            }

        try:
            # Ejecutar npm test
            result = subprocess.run(
                ["npm", "test", "--", "--json", "--outputFile=test_results.json"],
                cwd=frontend_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Leer resultados JSON si existe
            test_results_file = frontend_path / "test_results.json"
            test_data = {}
            if test_results_file.exists():
                test_data = json.loads(test_results_file.read_text(encoding="utf-8"))

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "tests_passed": test_data.get("numPassedTests", 0),
                "tests_failed": test_data.get("numFailedTests", 0),
                "tests_total": test_data.get("numTotalTests", 0),
                "output": result.stdout,
                "errors": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Tests timeout después de 5 minutos",
            }
        except Exception as e:
            logger.error(f"Error ejecutando tests: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def run_all_tests(
        self,
        project_path: Path,
    ) -> Dict[str, Any]:
        """
        Ejecuta todos los tests.

        Args:
            project_path: Ruta del proyecto

        Returns:
            Resultados completos
        """
        results = {
            "backend": {},
            "frontend": {},
            "overall_success": False,
            "timestamp": datetime.now().isoformat(),
        }

        # Tests de backend
        backend_path = project_path / "backend"
        if backend_path.exists():
            results["backend"] = await self.run_backend_tests(project_path)

        # Tests de frontend
        frontend_path = project_path / "frontend"
        if frontend_path.exists():
            results["frontend"] = await self.run_frontend_tests(project_path)

        # Determinar éxito general
        backend_success = results["backend"].get("success", True)
        frontend_success = results["frontend"].get("success", True)
        results["overall_success"] = backend_success and frontend_success

        return results


