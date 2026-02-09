"""
CI Integration
==============

Sistema de integración con CI para testing cuando hay problemas de entorno,
siguiendo las mejores prácticas de Devin de usar CI en lugar del entorno local
cuando hay problemas de entorno.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CIResult:
    """Resultado de ejecución en CI"""
    ci_system: str
    status: str
    build_id: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    logs_url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "ci_system": self.ci_system,
            "status": self.status,
            "build_id": self.build_id,
            "test_results": self.test_results,
            "logs_url": self.logs_url,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CITestRequest:
    """Solicitud de test en CI"""
    task_id: str
    reason: str
    environment_issue: Optional[str] = None
    status: str = "pending"
    ci_result: Optional[CIResult] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "task_id": self.task_id,
            "reason": self.reason,
            "environment_issue": self.environment_issue,
            "status": self.status,
            "ci_result": self.ci_result.to_dict() if self.ci_result else None,
            "timestamp": self.timestamp.isoformat()
        }


class CIIntegration:
    """
    Integración con CI.
    
    Permite usar CI para testing cuando hay problemas de entorno,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar integración con CI.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.test_requests: Dict[str, CITestRequest] = {}
        self.ci_systems: List[str] = []
        self._detect_ci_systems()
        logger.info("🔗 CI integration initialized")
    
    def _detect_ci_systems(self) -> None:
        """Detectar sistemas de CI disponibles"""
        ci_config_files = {
            '.github/workflows': 'github_actions',
            '.gitlab-ci.yml': 'gitlab_ci',
            '.circleci/config.yml': 'circleci',
            '.travis.yml': 'travis_ci',
            'azure-pipelines.yml': 'azure_pipelines',
            'Jenkinsfile': 'jenkins'
        }
        
        for config_path, ci_system in ci_config_files.items():
            config_file = self.workspace_root / config_path
            if config_file.exists():
                self.ci_systems.append(ci_system)
                logger.info(f"Detected CI system: {ci_system}")
    
    async def run_tests_via_ci(
        self,
        task_id: str,
        reason: str = "environment_issue",
        environment_issue: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar tests vía CI.
        
        Según las reglas de Devin:
        - Cuando hay problemas de entorno, usar CI en lugar del entorno local
        - No intentar arreglar problemas de entorno
        - Usar CI para testing
        
        Args:
            task_id: ID de la tarea.
            reason: Razón para usar CI.
            environment_issue: Descripción del problema de entorno (opcional).
        
        Returns:
            Resultado de la ejecución en CI.
        """
        request = CITestRequest(
            task_id=task_id,
            reason=reason,
            environment_issue=environment_issue
        )
        
        self.test_requests[task_id] = request
        
        if not self.ci_systems:
            logger.warning("No CI systems detected")
            request.status = "no_ci_available"
            return {
                "success": False,
                "error": "No CI systems detected",
                "suggestion": "Set up CI/CD pipeline or fix local environment"
            }
        
        try:
            ci_system = self.ci_systems[0]
            logger.info(f"Running tests via {ci_system}")
            
            if ci_system == 'github_actions':
                result = await self._run_github_actions()
            elif ci_system == 'gitlab_ci':
                result = await self._run_gitlab_ci()
            else:
                result = await self._run_generic_ci(ci_system)
            
            request.ci_result = result
            request.status = "completed" if result.status == "success" else "failed"
            
            return {
                "success": result.status == "success",
                "ci_system": ci_system,
                "status": result.status,
                "build_id": result.build_id,
                "test_results": result.test_results,
                "logs_url": result.logs_url
            }
        
        except Exception as e:
            logger.error(f"Error running tests via CI: {e}", exc_info=True)
            request.status = "error"
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_github_actions(self) -> CIResult:
        """Ejecutar tests vía GitHub Actions"""
        try:
            import subprocess
            process = await asyncio.create_subprocess_exec(
                'gh', 'workflow', 'run', 'test.yml',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root)
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode('utf-8', errors='ignore')
                build_id = self._extract_build_id(output)
                return CIResult(
                    ci_system="github_actions",
                    status="running",
                    build_id=build_id
                )
            else:
                return CIResult(
                    ci_system="github_actions",
                    status="error"
                )
        except Exception as e:
            logger.warning(f"Could not run GitHub Actions: {e}")
            return CIResult(
                ci_system="github_actions",
                status="unavailable"
            )
    
    async def _run_gitlab_ci(self) -> CIResult:
        """Ejecutar tests vía GitLab CI"""
        try:
            import subprocess
            process = await asyncio.create_subprocess_exec(
                'gitlab-ci-local', '--run',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root)
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return CIResult(
                    ci_system="gitlab_ci",
                    status="success"
                )
            else:
                return CIResult(
                    ci_system="gitlab_ci",
                    status="failed"
                )
        except Exception as e:
            logger.warning(f"Could not run GitLab CI: {e}")
            return CIResult(
                ci_system="gitlab_ci",
                status="unavailable"
            )
    
    async def _run_generic_ci(self, ci_system: str) -> CIResult:
        """Ejecutar tests vía CI genérico"""
        return CIResult(
            ci_system=ci_system,
            status="unavailable",
            build_id=None
        )
    
    def _extract_build_id(self, output: str) -> Optional[str]:
        """Extraer ID de build del output"""
        import re
        patterns = [
            r'Run ID: (\d+)',
            r'build_id=(\d+)',
            r'#(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1)
        
        return None
    
    def should_use_ci(self, has_environment_issue: bool = False) -> bool:
        """
        Determinar si se debe usar CI.
        
        Según las reglas de Devin:
        - Usar CI cuando hay problemas de entorno
        - No intentar arreglar problemas de entorno
        
        Args:
            has_environment_issue: Si hay problema de entorno.
        
        Returns:
            True si se debe usar CI.
        """
        if has_environment_issue and self.ci_systems:
            return True
        return False
    
    def get_test_request(self, task_id: str) -> Optional[CITestRequest]:
        """Obtener solicitud de test"""
        return self.test_requests.get(task_id)
    
    def get_all_test_requests(self) -> List[Dict[str, Any]]:
        """Obtener todas las solicitudes"""
        return [tr.to_dict() for tr in self.test_requests.values()]

