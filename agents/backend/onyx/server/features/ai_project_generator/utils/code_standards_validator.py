"""
Code Standards Validator
========================

Sistema de validación de estándares de código.
"""

import ast
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StandardType(Enum):
    """Tipos de estándares."""
    PEP8 = "pep8"
    GOOGLE = "google"
    NAMING = "naming"
    DOCUMENTATION = "documentation"
    TYPE_HINTS = "type_hints"
    ERROR_HANDLING = "error_handling"


@dataclass
class StandardViolation:
    """Violación de estándar."""
    standard_type: StandardType
    rule: str
    severity: str
    location: str
    line_number: int
    description: str
    fix: str


class CodeStandardsValidator:
    """Validador de estándares de código."""
    
    def __init__(self):
        self.standards = {
            StandardType.PEP8: self._validate_pep8,
            StandardType.NAMING: self._validate_naming,
            StandardType.DOCUMENTATION: self._validate_documentation,
            StandardType.TYPE_HINTS: self._validate_type_hints,
            StandardType.ERROR_HANDLING: self._validate_error_handling,
        }
    
    def validate(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Valida código contra estándares."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "violations": []
            }
        
        violations = []
        
        # Validar cada estándar
        for standard_type, validator in self.standards.items():
            violations.extend(validator(code, tree))
        
        # Calcular puntuación
        score = self._calculate_score(violations, code)
        
        return {
            "file_path": file_path,
            "total_violations": len(violations),
            "score": score,
            "by_standard": self._group_by_standard(violations),
            "by_severity": self._group_by_severity(violations),
            "violations": [
                {
                    "standard": v.standard_type.value,
                    "rule": v.rule,
                    "severity": v.severity,
                    "location": v.location,
                    "line": v.line_number,
                    "description": v.description,
                    "fix": v.fix
                }
                for v in violations
            ]
        }
    
    def _validate_pep8(self, code: str, tree: ast.AST) -> List[StandardViolation]:
        """Valida PEP8."""
        violations = []
        lines = code.split('\n')
        
        # Líneas muy largas (>79 caracteres)
        for i, line in enumerate(lines, 1):
            if len(line) > 79:
                violations.append(StandardViolation(
                    standard_type=StandardType.PEP8,
                    rule="E501",
                    severity="low",
                    location=f"Line {i}",
                    line_number=i,
                    description=f"Line too long ({len(line)} > 79 characters)",
                    fix="Break line into multiple lines"
                ))
        
        # Múltiples espacios
        for i, line in enumerate(lines, 1):
            if re.search(r'[^ ]  +', line):
                violations.append(StandardViolation(
                    standard_type=StandardType.PEP8,
                    rule="E271",
                    severity="low",
                    location=f"Line {i}",
                    line_number=i,
                    description="Multiple spaces found",
                    fix="Use single space"
                ))
        
        # Imports no ordenados
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if len(imports) > 1:
            violations.append(StandardViolation(
                standard_type=StandardType.PEP8,
                rule="E401",
                severity="low",
                location="Imports section",
                line_number=1,
                description="Imports should be grouped and sorted",
                fix="Group imports: stdlib, third-party, local"
            ))
        
        return violations
    
    def _validate_naming(self, code: str, tree: ast.AST) -> List[StandardViolation]:
        """Valida convenciones de nombres."""
        violations = []
        
        # Nombres de clases deben ser PascalCase
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    violations.append(StandardViolation(
                        standard_type=StandardType.NAMING,
                        rule="N801",
                        severity="medium",
                        location=f"Class {node.name}",
                        line_number=node.lineno,
                        description=f"Class name '{node.name}' should be PascalCase",
                        fix=f"Rename to PascalCase: {node.name.title()}"
                    ))
        
        # Nombres de funciones deben ser snake_case
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    violations.append(StandardViolation(
                        standard_type=StandardType.NAMING,
                        rule="N802",
                        severity="medium",
                        location=f"Function {node.name}",
                        line_number=node.lineno,
                        description=f"Function name '{node.name}' should be snake_case",
                        fix=f"Rename to snake_case: {node.name.lower().replace('-', '_')}"
                    ))
        
        # Constantes deben ser UPPER_CASE
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper() and '_' not in target.id:
                            # Verificar si es realmente una constante
                            if isinstance(node.value, (ast.Constant, ast.Str, ast.Num)):
                                if not target.id.isupper() or target.id.islower():
                                    violations.append(StandardViolation(
                                        standard_type=StandardType.NAMING,
                                        rule="N806",
                                        severity="low",
                                        location=f"Constant {target.id}",
                                        line_number=node.lineno,
                                        description=f"Constant '{target.id}' should be UPPER_CASE",
                                        fix=f"Rename to UPPER_CASE: {target.id.upper()}"
                                    ))
        
        return violations
    
    def _validate_documentation(self, code: str, tree: ast.AST) -> List[StandardViolation]:
        """Valida documentación."""
        violations = []
        
        # Funciones públicas deben tener docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_') and ast.get_docstring(node) is None:
                    violations.append(StandardViolation(
                        standard_type=StandardType.DOCUMENTATION,
                        rule="D100",
                        severity="medium",
                        location=f"Function {node.name}",
                        line_number=node.lineno,
                        description=f"Public function '{node.name}' missing docstring",
                        fix=f'Add docstring: """Description of {node.name}."""'
                    ))
        
        # Clases deben tener docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if ast.get_docstring(node) is None:
                    violations.append(StandardViolation(
                        standard_type=StandardType.DOCUMENTATION,
                        rule="D101",
                        severity="medium",
                        location=f"Class {node.name}",
                        line_number=node.lineno,
                        description=f"Class '{node.name}' missing docstring",
                        fix=f'Add docstring: """Description of {node.name}."""'
                    ))
        
        return violations
    
    def _validate_type_hints(self, code: str, tree: ast.AST) -> List[StandardViolation]:
        """Valida type hints."""
        violations = []
        
        # Funciones públicas deben tener type hints
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    has_return_hint = node.returns is not None
                    has_param_hints = any(arg.annotation is not None for arg in node.args.args)
                    
                    if not has_param_hints or not has_return_hint:
                        violations.append(StandardViolation(
                            standard_type=StandardType.TYPE_HINTS,
                            rule="TYP001",
                            severity="low",
                            location=f"Function {node.name}",
                            line_number=node.lineno,
                            description=f"Function '{node.name}' missing type hints",
                            fix="Add type hints to parameters and return type"
                        ))
        
        return violations
    
    def _validate_error_handling(self, code: str, tree: ast.AST) -> List[StandardViolation]:
        """Valida manejo de errores."""
        violations = []
        
        # Operaciones riesgosas deben tener try/except
        risky_operations = ['open(', 'requests.', 'db.', 'sql']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(code, node) or ""
                has_try = 'try' in func_code
                
                for op in risky_operations:
                    if op in func_code and not has_try:
                        violations.append(StandardViolation(
                            standard_type=StandardType.ERROR_HANDLING,
                            rule="ERR001",
                            severity="high",
                            location=f"Function {node.name}",
                            line_number=node.lineno,
                            description=f"Function '{node.name}' uses risky operation without error handling",
                            fix="Wrap in try/except block"
                        ))
                        break
        
        return violations
    
    def _calculate_score(self, violations: List[StandardViolation], code: str) -> float:
        """Calcula puntuación de cumplimiento."""
        if not violations:
            return 100.0
        
        total_lines = len(code.split('\n'))
        if total_lines == 0:
            return 0.0
        
        # Penalización por violación
        penalty = 0
        for v in violations:
            if v.severity == "critical":
                penalty += 5
            elif v.severity == "high":
                penalty += 3
            elif v.severity == "medium":
                penalty += 2
            else:
                penalty += 1
        
        score = max(0, 100 - (penalty / total_lines * 100))
        return round(score, 2)
    
    def _group_by_standard(self, violations: List[StandardViolation]) -> Dict[str, int]:
        """Agrupa por estándar."""
        grouped = {}
        for v in violations:
            grouped[v.standard_type.value] = grouped.get(v.standard_type.value, 0) + 1
        return grouped
    
    def _group_by_severity(self, violations: List[StandardViolation]) -> Dict[str, int]:
        """Agrupa por severidad."""
        grouped = {}
        for v in violations:
            grouped[v.severity] = grouped.get(v.severity, 0) + 1
        return grouped


# Factory function
_code_standards_validator = None

def get_code_standards_validator() -> CodeStandardsValidator:
    """Obtiene instancia global del validador."""
    global _code_standards_validator
    if _code_standards_validator is None:
        _code_standards_validator = CodeStandardsValidator()
    return _code_standards_validator

