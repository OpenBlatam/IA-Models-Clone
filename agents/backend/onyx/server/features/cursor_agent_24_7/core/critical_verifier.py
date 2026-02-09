"""
Critical Verifier
=================

Sistema que realiza verificación crítica antes de reportar al usuario,
siguiendo las mejores prácticas de Devin de examinar críticamente el trabajo
antes de reportar completitud.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VerificationCheck:
    """Verificación individual"""
    check_type: str
    description: str
    passed: bool = False
    details: Optional[str] = None
    critical: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "check_type": self.check_type,
            "description": self.description,
            "passed": self.passed,
            "details": self.details,
            "critical": self.critical
        }


@dataclass
class CriticalVerification:
    """Verificación crítica completa"""
    task_id: str
    task_description: str
    checks: List[VerificationCheck] = field(default_factory=list)
    all_critical_passed: bool = False
    overall_passed: bool = False
    issues: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_check(
        self,
        check_type: str,
        description: str,
        passed: bool,
        details: Optional[str] = None,
        critical: bool = True
    ) -> VerificationCheck:
        """Agregar verificación"""
        check = VerificationCheck(
            check_type=check_type,
            description=description,
            passed=passed,
            details=details,
            critical=critical
        )
        self.checks.append(check)
        return check
    
    def evaluate(self) -> None:
        """Evaluar todas las verificaciones"""
        critical_checks = [c for c in self.checks if c.critical]
        self.all_critical_passed = all(c.passed for c in critical_checks)
        self.overall_passed = all(c.passed for c in self.checks)
        
        if not self.all_critical_passed:
            failed_critical = [c for c in critical_checks if not c.passed]
            self.issues = [
                f"Critical check failed: {c.description}" 
                for c in failed_critical
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "checks": [c.to_dict() for c in self.checks],
            "all_critical_passed": self.all_critical_passed,
            "overall_passed": self.overall_passed,
            "issues": self.issues,
            "timestamp": self.timestamp.isoformat()
        }


class CriticalVerifier:
    """
    Verificador crítico.
    
    Realiza verificación crítica antes de reportar al usuario,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar verificador crítico.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.verifications: Dict[str, CriticalVerification] = {}
        logger.info("🔍 Critical verifier initialized")
    
    async def verify_before_reporting(
        self,
        task_id: str,
        task_description: str,
        agent: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Verificar críticamente antes de reportar.
        
        Realiza todas las verificaciones críticas según Devin:
        1. Verificar que se cumplió la intención del usuario
        2. Verificar que todos los archivos fueron modificados
        3. Verificar que todas las referencias fueron actualizadas
        4. Verificar que tests pasan
        5. Verificar que linting pasa
        6. Verificar que no hay secretos expuestos
        7. Verificar que se siguieron las convenciones
        
        Args:
            task_id: ID de la tarea.
            task_description: Descripción de la tarea.
            agent: Instancia del agente (opcional).
        
        Returns:
            Resultado de la verificación.
        """
        verification = CriticalVerification(
            task_id=task_id,
            task_description=task_description
        )
        
        # Verificación 1: Tests pasan
        if agent and hasattr(agent, 'test_runner'):
            try:
                test_results = await agent.test_runner.run_tests()
                tests_passed = all(r.success for r in test_results) if test_results else True
                verification.add_check(
                    "tests",
                    "All tests pass",
                    tests_passed,
                    details=f"{len(test_results)} tests run" if test_results else "No tests found",
                    critical=True
                )
            except Exception as e:
                verification.add_check(
                    "tests",
                    "All tests pass",
                    False,
                    details=f"Error running tests: {e}",
                    critical=True
                )
        else:
            verification.add_check(
                "tests",
                "All tests pass",
                True,
                details="Test runner not available",
                critical=False
            )
        
        # Verificación 2: Linting pasa
        if agent and hasattr(agent, 'test_runner'):
            try:
                lint_result = await agent.test_runner.run_lint()
                lint_passed = lint_result.success if lint_result else True
                verification.add_check(
                    "linting",
                    "Linting passes",
                    lint_passed,
                    details=f"{len(lint_result.errors)} errors" if lint_result and not lint_result.success else "No errors",
                    critical=True
                )
            except Exception as e:
                verification.add_check(
                    "linting",
                    "Linting passes",
                    False,
                    details=f"Error running lint: {e}",
                    critical=True
                )
        else:
            verification.add_check(
                "linting",
                "Linting passes",
                True,
                details="Lint runner not available",
                critical=False
            )
        
        # Verificación 3: Referencias actualizadas
        if agent and hasattr(agent, 'reference_tracker'):
            try:
                tracked = agent.reference_tracker.get_tracked_changes()
                if tracked:
                    all_updated = True
                    pending_refs = []
                    for change_id, refs in tracked.items():
                        for ref in refs:
                            if not ref.get('updated', False):
                                all_updated = False
                                pending_refs.append(ref.get('file_path', 'unknown'))
                    
                    verification.add_check(
                        "references",
                        "All references updated",
                        all_updated,
                        details=f"{len(pending_refs)} pending references" if pending_refs else "All updated",
                        critical=True
                    )
                else:
                    verification.add_check(
                        "references",
                        "All references updated",
                        True,
                        details="No references to track",
                        critical=False
                    )
            except Exception as e:
                verification.add_check(
                    "references",
                    "All references updated",
                    False,
                    details=f"Error checking references: {e}",
                    critical=True
                )
        else:
            verification.add_check(
                "references",
                "All references updated",
                True,
                details="Reference tracker not available",
                critical=False
            )
        
        # Verificación 4: No hay secretos expuestos
        if agent and hasattr(agent, 'security_manager'):
            try:
                detected_secrets = agent.security_manager.secret_detector.get_detected_secrets()
                no_secrets = len(detected_secrets) == 0
                verification.add_check(
                    "secrets",
                    "No secrets exposed",
                    no_secrets,
                    details=f"{len(detected_secrets)} secrets detected" if detected_secrets else "No secrets found",
                    critical=True
                )
            except Exception as e:
                verification.add_check(
                    "secrets",
                    "No secrets exposed",
                    True,
                    details=f"Error checking secrets: {e}",
                    critical=False
                )
        else:
            verification.add_check(
                "secrets",
                "No secrets exposed",
                True,
                details="Security manager not available",
                critical=False
            )
        
        # Verificación 5: Cambios verificados
        if agent and hasattr(agent, 'change_verifier'):
            try:
                change_sets = agent.change_verifier.get_all_change_sets()
                if change_sets:
                    all_verified = True
                    unverified = []
                    for cs in change_sets:
                        if not cs.get('all_verified', False):
                            all_verified = False
                            unverified.append(cs.get('id', 'unknown'))
                    
                    verification.add_check(
                        "changes",
                        "All changes verified",
                        all_verified,
                        details=f"{len(unverified)} unverified change sets" if unverified else "All verified",
                        critical=True
                    )
                else:
                    verification.add_check(
                        "changes",
                        "All changes verified",
                        True,
                        details="No change sets to verify",
                        critical=False
                    )
            except Exception as e:
                verification.add_check(
                    "changes",
                    "All changes verified",
                    False,
                    details=f"Error verifying changes: {e}",
                    critical=True
                )
        else:
            verification.add_check(
                "changes",
                "All changes verified",
                True,
                details="Change verifier not available",
                critical=False
            )
        
        # Verificación 6: Requisitos cumplidos
        if agent and hasattr(agent, 'completion_verifier'):
            try:
                task = agent.completion_verifier.get_task(task_id)
                if task:
                    all_met = task.all_requirements_met
                    verification.add_check(
                        "requirements",
                        "All requirements met",
                        all_met,
                        details=f"{len([r for r in task.requirements if r.verified])}/{len(task.requirements)} requirements met",
                        critical=True
                    )
                else:
                    verification.add_check(
                        "requirements",
                        "All requirements met",
                        True,
                        details="No requirements defined",
                        critical=False
                    )
            except Exception as e:
                verification.add_check(
                    "requirements",
                    "All requirements met",
                    False,
                    details=f"Error checking requirements: {e}",
                    critical=True
                )
        else:
            verification.add_check(
                "requirements",
                "All requirements met",
                True,
                details="Completion verifier not available",
                critical=False
            )
        
        # Evaluar todas las verificaciones
        verification.evaluate()
        self.verifications[task_id] = verification
        
        return {
            "success": verification.all_critical_passed,
            "overall_success": verification.overall_passed,
            "checks": [c.to_dict() for c in verification.checks],
            "issues": verification.issues,
            "can_report": verification.all_critical_passed
        }
    
    def get_verification(self, task_id: str) -> Optional[CriticalVerification]:
        """Obtener verificación"""
        return self.verifications.get(task_id)
    
    def get_all_verifications(self) -> List[Dict[str, Any]]:
        """Obtener todas las verificaciones"""
        return [v.to_dict() for v in self.verifications.values()]

