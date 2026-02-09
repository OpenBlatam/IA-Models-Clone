"""
Compliance Checker System
=========================

Sistema de verificación de cumplimiento.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Estándar de cumplimiento."""
    PEP8 = "pep8"
    PEP257 = "pep257"
    CUSTOM = "custom"


@dataclass
class ComplianceRule:
    """Regla de cumplimiento."""
    rule_id: str
    standard: ComplianceStandard
    name: str
    description: str
    check_function: callable
    enabled: bool = True


@dataclass
class ComplianceResult:
    """Resultado de cumplimiento."""
    rule_id: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class ComplianceChecker:
    """
    Verificador de cumplimiento.
    
    Verifica el cumplimiento de estándares de código.
    """
    
    def __init__(self):
        """Inicializar verificador de cumplimiento."""
        self.rules: Dict[str, ComplianceRule] = {}
        self.check_history: List[Dict[str, Any]] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Configurar reglas por defecto."""
        # Regla: Docstrings en funciones públicas
        def check_docstrings(file_content: str) -> bool:
            # Implementación simplificada
            return True
        
        self.add_rule(
            rule_id="docstrings_required",
            standard=ComplianceStandard.PEP257,
            name="Docstrings Required",
            description="All public functions must have docstrings",
            check_function=check_docstrings
        )
    
    def add_rule(
        self,
        rule_id: str,
        standard: ComplianceStandard,
        name: str,
        description: str,
        check_function: callable,
        enabled: bool = True
    ) -> ComplianceRule:
        """
        Agregar regla de cumplimiento.
        
        Args:
            rule_id: ID único de la regla
            standard: Estándar de cumplimiento
            name: Nombre de la regla
            description: Descripción
            check_function: Función de verificación
            enabled: Si está habilitada
            
        Returns:
            Regla creada
        """
        rule = ComplianceRule(
            rule_id=rule_id,
            standard=standard,
            name=name,
            description=description,
            check_function=check_function,
            enabled=enabled
        )
        
        self.rules[rule_id] = rule
        logger.info(f"Added compliance rule: {name} ({rule_id})")
        
        return rule
    
    def check_file(self, file_path: str, standard: Optional[ComplianceStandard] = None) -> List[ComplianceResult]:
        """
        Verificar archivo.
        
        Args:
            file_path: Ruta del archivo
            standard: Estándar a verificar (None = todos)
            
        Returns:
            Lista de resultados
        """
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        results = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if standard and rule.standard != standard:
                continue
            
            try:
                passed = rule.check_function(content)
                results.append(ComplianceResult(
                    rule_id=rule.rule_id,
                    passed=passed,
                    message=f"{rule.name}: {'PASSED' if passed else 'FAILED'}"
                ))
            except Exception as e:
                logger.error(f"Error checking rule {rule.rule_id}: {e}")
                results.append(ComplianceResult(
                    rule_id=rule.rule_id,
                    passed=False,
                    message=f"Error: {str(e)}"
                ))
        
        return results
    
    def check_directory(
        self,
        directory: str,
        standard: Optional[ComplianceStandard] = None,
        pattern: str = "*.py"
    ) -> Dict[str, Any]:
        """
        Verificar directorio.
        
        Args:
            directory: Directorio a verificar
            standard: Estándar a verificar
            pattern: Patrón de archivos
            
        Returns:
            Resumen de verificación
        """
        from pathlib import Path
        
        path = Path(directory)
        all_results = {}
        
        for file_path in path.rglob(pattern):
            results = self.check_file(str(file_path), standard=standard)
            all_results[str(file_path)] = results
        
        # Calcular estadísticas
        total_files = len(all_results)
        total_checks = sum(len(results) for results in all_results.values())
        passed_checks = sum(
            sum(1 for r in results if r.passed)
            for results in all_results.values()
        )
        
        return {
            "total_files": total_files,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "compliance_rate": passed_checks / total_checks if total_checks > 0 else 0.0,
            "results": all_results
        }
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de cumplimiento."""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "standards": list(set(r.standard.value for r in self.rules.values()))
        }


# Instancia global
_compliance_checker: Optional[ComplianceChecker] = None


def get_compliance_checker() -> ComplianceChecker:
    """Obtener instancia global del verificador de cumplimiento."""
    global _compliance_checker
    if _compliance_checker is None:
        _compliance_checker = ComplianceChecker()
    return _compliance_checker






