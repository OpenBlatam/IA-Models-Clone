"""
Change Verifier
===============

Sistema que verifica que todos los cambios están completos antes de reportar,
siguiendo las mejores prácticas de Devin de verificar que se cumplieron
todos los requisitos.
"""

import logging
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChangeItem:
    """Item de cambio"""
    file_path: str
    change_type: str
    description: str
    status: str = "pending"
    verified: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "file_path": self.file_path,
            "change_type": self.change_type,
            "description": self.description,
            "status": self.status,
            "verified": self.verified,
            "error": self.error
        }


@dataclass
class ChangeSet:
    """Conjunto de cambios"""
    id: str
    description: str
    changes: List[ChangeItem] = field(default_factory=list)
    references_to_update: List[str] = field(default_factory=list)
    tests_to_run: List[str] = field(default_factory=list)
    lint_checks: bool = True
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)
    verified_at: Optional[datetime] = None
    
    def add_change(
        self,
        file_path: str,
        change_type: str,
        description: str
    ) -> ChangeItem:
        """Agregar cambio"""
        change = ChangeItem(
            file_path=file_path,
            change_type=change_type,
            description=description
        )
        self.changes.append(change)
        return change
    
    def add_reference(self, reference: str) -> None:
        """Agregar referencia a actualizar"""
        if reference not in self.references_to_update:
            self.references_to_update.append(reference)
    
    def add_test(self, test_path: str) -> None:
        """Agregar test a ejecutar"""
        if test_path not in self.tests_to_run:
            self.tests_to_run.append(test_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "description": self.description,
            "changes": [c.to_dict() for c in self.changes],
            "references_to_update": self.references_to_update,
            "tests_to_run": self.tests_to_run,
            "lint_checks": self.lint_checks,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "verified_at": self.verified_at.isoformat() if self.verified_at else None
        }


class ChangeVerifier:
    """
    Verificador de cambios.
    
    Verifica que todos los cambios están completos antes de reportar,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar verificador.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.change_sets: Dict[str, ChangeSet] = {}
        logger.info("✅ Change verifier initialized")
    
    def create_change_set(self, description: str) -> ChangeSet:
        """
        Crear conjunto de cambios.
        
        Args:
            description: Descripción del conjunto de cambios.
        
        Returns:
            Conjunto de cambios creado.
        """
        change_set_id = f"changeset_{datetime.now().timestamp()}"
        change_set = ChangeSet(
            id=change_set_id,
            description=description
        )
        self.change_sets[change_set_id] = change_set
        return change_set
    
    async def verify_change_set(
        self,
        change_set_id: str,
        run_tests: bool = True,
        run_lint: bool = True
    ) -> Dict[str, Any]:
        """
        Verificar conjunto de cambios.
        
        Args:
            change_set_id: ID del conjunto de cambios.
            run_tests: Si ejecutar tests (default: True).
            run_lint: Si ejecutar linting (default: True).
        
        Returns:
            Resultado de la verificación.
        """
        if change_set_id not in self.change_sets:
            return {
                "success": False,
                "error": f"Change set {change_set_id} not found"
            }
        
        change_set = self.change_sets[change_set_id]
        results = {
            "success": True,
            "changes_verified": 0,
            "changes_total": len(change_set.changes),
            "references_verified": 0,
            "references_total": len(change_set.references_to_update),
            "tests_passed": None,
            "lint_passed": None,
            "issues": []
        }
        
        for change in change_set.changes:
            file_path = Path(change.file_path)
            if not file_path.is_absolute():
                file_path = self.workspace_root / file_path
            
            if file_path.exists():
                change.verified = True
                change.status = "verified"
                results["changes_verified"] += 1
            else:
                change.verified = False
                change.status = "missing"
                change.error = "File not found"
                results["issues"].append({
                    "type": "missing_file",
                    "file": change.file_path,
                    "message": f"File {change.file_path} not found"
                })
        
        for ref in change_set.references_to_update:
            ref_path = Path(ref)
            if not ref_path.is_absolute():
                ref_path = self.workspace_root / ref_path
            
            if ref_path.exists():
                results["references_verified"] += 1
            else:
                results["issues"].append({
                    "type": "missing_reference",
                    "reference": ref,
                    "message": f"Reference {ref} not found"
                })
        
        if run_tests and change_set.tests_to_run:
            try:
                from .test_runner import TestRunner
                test_runner = TestRunner(str(self.workspace_root))
                test_results = await test_runner.run_tests()
                if test_results:
                    results["tests_passed"] = all(r.success for r in test_results)
                    if not results["tests_passed"]:
                        results["issues"].append({
                            "type": "tests_failed",
                            "message": "Some tests failed"
                        })
            except Exception as e:
                logger.warning(f"Could not run tests: {e}")
                results["tests_passed"] = None
        
        if run_lint and change_set.lint_checks:
            try:
                from .test_runner import TestRunner
                test_runner = TestRunner(str(self.workspace_root))
                lint_result = await test_runner.run_lint()
                results["lint_passed"] = lint_result.success
                if not lint_result.success:
                    results["issues"].append({
                        "type": "lint_failed",
                        "message": f"Lint found {len(lint_result.errors)} errors"
                    })
            except Exception as e:
                logger.warning(f"Could not run lint: {e}")
                results["lint_passed"] = None
        
        if results["changes_verified"] == results["changes_total"] and \
           results["references_verified"] == results["references_total"] and \
           len(results["issues"]) == 0 and \
           (results["tests_passed"] is None or results["tests_passed"]) and \
           (results["lint_passed"] is None or results["lint_passed"]):
            change_set.status = "verified"
            change_set.verified_at = datetime.now()
            results["success"] = True
        else:
            change_set.status = "incomplete"
            results["success"] = False
        
        return results
    
    def get_change_set(self, change_set_id: str) -> Optional[ChangeSet]:
        """
        Obtener conjunto de cambios.
        
        Args:
            change_set_id: ID del conjunto de cambios.
        
        Returns:
            Conjunto de cambios o None.
        """
        return self.change_sets.get(change_set_id)
    
    def get_all_change_sets(self) -> List[Dict[str, Any]]:
        """Obtener todos los conjuntos de cambios"""
        return [cs.to_dict() for cs in self.change_sets.values()]

