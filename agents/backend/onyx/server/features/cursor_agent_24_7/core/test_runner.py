"""
Test Runner
===========

Sistema para ejecutar tests y verificaciones antes de reportar cambios,
siguiendo las mejores prácticas de Devin.
"""

import logging
import subprocess
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Resultado de ejecución de test"""
    test_name: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "test_name": self.test_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LintResult:
    """Resultado de linting"""
    success: bool
    output: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "success": self.success,
            "output": self.output,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


class TestRunner:
    """
    Ejecutor de tests y verificaciones.
    
    Ejecuta tests y linting antes de reportar cambios,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar ejecutor de tests.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.test_results: List[TestResult] = []
        self.lint_results: List[LintResult] = []
        logger.info("🧪 Test runner initialized")
    
    async def run_tests(
        self,
        test_path: Optional[str] = None,
        test_command: Optional[str] = None
    ) -> List[TestResult]:
        """
        Ejecutar tests.
        
        Args:
            test_path: Ruta específica de tests (opcional).
            test_command: Comando personalizado de tests (opcional).
        
        Returns:
            Lista de resultados de tests.
        """
        results: List[TestResult] = []
        start_time = datetime.now()
        
        try:
            if test_command:
                command = test_command.split()
            elif test_path:
                command = self._get_test_command(test_path)
            else:
                command = self._detect_test_command()
            
            if not command:
                logger.warning("No test command found")
                return results
            
            logger.info(f"Running tests: {' '.join(command)}")
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root)
            )
            
            stdout, stderr = await process.communicate()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            output = stdout.decode('utf-8', errors='ignore')
            error_output = stderr.decode('utf-8', errors='ignore')
            
            success = process.returncode == 0
            
            result = TestResult(
                test_name=test_path or "all_tests",
                success=success,
                output=output,
                error=error_output if not success else None,
                execution_time=execution_time
            )
            
            results.append(result)
            self.test_results.append(result)
            
            if success:
                logger.info(f"✅ Tests passed in {execution_time:.2f}s")
            else:
                logger.error(f"❌ Tests failed: {error_output[:200]}")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result = TestResult(
                test_name=test_path or "all_tests",
                success=False,
                error=str(e),
                execution_time=execution_time
            )
            results.append(result)
            self.test_results.append(result)
            logger.error(f"Error running tests: {e}", exc_info=True)
        
        return results
    
    async def run_lint(
        self,
        lint_command: Optional[str] = None,
        files: Optional[List[str]] = None
    ) -> LintResult:
        """
        Ejecutar linting.
        
        Args:
            lint_command: Comando personalizado de linting (opcional).
            files: Archivos específicos a verificar (opcional).
        
        Returns:
            Resultado de linting.
        """
        start_time = datetime.now()
        
        try:
            if lint_command:
                command = lint_command.split()
            else:
                command = self._detect_lint_command()
            
            if not command:
                logger.warning("No lint command found")
                return LintResult(
                    success=False,
                    error="No lint command detected"
                )
            
            if files:
                command.extend(files)
            
            logger.info(f"Running lint: {' '.join(command)}")
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root)
            )
            
            stdout, stderr = await process.communicate()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            output = stdout.decode('utf-8', errors='ignore')
            error_output = stderr.decode('utf-8', errors='ignore')
            
            success = process.returncode == 0
            
            errors = []
            warnings = []
            
            if output:
                for line in output.split('\n'):
                    if 'error' in line.lower():
                        errors.append(line)
                    elif 'warning' in line.lower():
                        warnings.append(line)
            
            result = LintResult(
                success=success,
                output=output,
                errors=errors,
                warnings=warnings,
                execution_time=execution_time
            )
            
            self.lint_results.append(result)
            
            if success:
                logger.info(f"✅ Lint passed in {execution_time:.2f}s")
            else:
                logger.warning(f"⚠️ Lint found issues: {len(errors)} errors, {len(warnings)} warnings")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result = LintResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
            self.lint_results.append(result)
            logger.error(f"Error running lint: {e}", exc_info=True)
            return result
    
    def _detect_test_command(self) -> List[str]:
        """Detectar comando de tests"""
        if (self.workspace_root / "pytest.ini").exists() or \
           (self.workspace_root / "pyproject.toml").exists():
            return [sys.executable, "-m", "pytest"]
        elif (self.workspace_root / "setup.py").exists():
            return [sys.executable, "-m", "unittest", "discover"]
        elif (self.workspace_root / "tests").exists():
            return [sys.executable, "-m", "pytest", "tests"]
        else:
            return []
    
    def _get_test_command(self, test_path: str) -> List[str]:
        """Obtener comando para test específico"""
        base_command = self._detect_test_command()
        if base_command:
            base_command.append(test_path)
        return base_command
    
    def _detect_lint_command(self) -> List[str]:
        """Detectar comando de linting"""
        if (self.workspace_root / ".flake8").exists() or \
           (self.workspace_root / "setup.cfg").exists():
            return [sys.executable, "-m", "flake8"]
        elif (self.workspace_root / "pyproject.toml").exists():
            try:
                import tomli
                with open(self.workspace_root / "pyproject.toml", "rb") as f:
                    config = tomli.load(f)
                    if "tool" in config and "ruff" in config.get("tool", {}):
                        return [sys.executable, "-m", "ruff", "check"]
                    elif "tool" in config and "black" in config.get("tool", {}):
                        return [sys.executable, "-m", "black", "--check"]
            except Exception as e:
                # Fallback: if we can't read pyproject.toml, continue to default linter
                logger.debug(f"Could not read pyproject.toml for linter config: {e}")
                pass
        
        if (self.workspace_root / ".ruff.toml").exists():
            return [sys.executable, "-m", "ruff", "check"]
        
        return [sys.executable, "-m", "pylint"]
    
    def get_recent_test_results(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Obtener resultados de tests recientes"""
        return [r.to_dict() for r in self.test_results[-limit:]]
    
    def get_recent_lint_results(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Obtener resultados de linting recientes"""
        return [r.to_dict() for r in self.lint_results[-limit:]]

