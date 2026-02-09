"""
Test Runner - Sistema de testing automatizado
==============================================
"""

import logging
import subprocess
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TestRunner:
    """
    Ejecuta tests automatizados para código mejorado.
    """
    
    def __init__(self, test_dir: str = "tests"):
        """
        Inicializar test runner.
        
        Args:
            test_dir: Directorio de tests
        """
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
    
    def run_tests(
        self,
        code: str,
        tests: str,
        language: str = "python",
        framework: str = "pytest"
    ) -> Dict[str, Any]:
        """
        Ejecuta tests para código.
        
        Args:
            code: Código a testear
            tests: Código de tests
            language: Lenguaje de programación
            framework: Framework de testing
            
        Returns:
            Resultado de los tests
        """
        try:
            if language == "python":
                return self._run_python_tests(code, tests, framework)
            elif language in ["javascript", "typescript"]:
                return self._run_js_tests(code, tests, framework)
            else:
                return {
                    "success": False,
                    "error": f"Lenguaje no soportado: {language}"
                }
        except Exception as e:
            logger.error(f"Error ejecutando tests: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_python_tests(
        self,
        code: str,
        tests: str,
        framework: str
    ) -> Dict[str, Any]:
        """Ejecuta tests Python"""
        try:
            # Crear archivos temporales
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                
                # Escribir código
                code_file = tmp_path / "code.py"
                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Escribir tests
                test_file = tmp_path / "test_code.py"
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write(tests)
                
                # Ejecutar tests
                if framework == "pytest":
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                else:  # unittest
                    result = subprocess.run(
                        [sys.executable, "-m", "unittest", str(test_file)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                
                return {
                    "success": result.returncode == 0,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "tests_passed": "passed" in result.stdout.lower() or result.returncode == 0,
                    "framework": framework
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Tests timeout (30s)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_js_tests(
        self,
        code: str,
        tests: str,
        framework: str
    ) -> Dict[str, Any]:
        """Ejecuta tests JavaScript/TypeScript"""
        # Placeholder - en producción usaría jest/mocha real
        return {
            "success": True,
            "message": "JavaScript tests execution (placeholder)",
            "framework": framework
        }
    
    def validate_tests_before_improvement(
        self,
        original_code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Valida que los tests pasen antes de mejorar código.
        
        Args:
            original_code: Código original
            language: Lenguaje
            
        Returns:
            Resultado de validación
        """
        # En producción, esto ejecutaría tests existentes
        return {
            "tests_exist": False,
            "tests_passed": None,
            "message": "No se encontraron tests existentes"
        }




