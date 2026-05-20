"""
Advanced Bug Detector
=====================

Sistema avanzado de detección de bugs.
"""

import ast
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BugType(Enum):
    """Tipos de bugs."""
    NULL_POINTER = "null_pointer"
    INDEX_ERROR = "index_error"
    TYPE_ERROR = "type_error"
    LOGIC_ERROR = "logic_error"
    RACE_CONDITION = "race_condition"
    RESOURCE_LEAK = "resource_leak"
    INFINITE_LOOP = "infinite_loop"
    DIVISION_BY_ZERO = "division_by_zero"
    UNDEFINED_VARIABLE = "undefined_variable"


@dataclass
class BugReport:
    """Reporte de bug."""
    bug_type: BugType
    severity: str  # low, medium, high, critical
    location: str
    line_number: int
    description: str
    code_snippet: str
    fix_suggestion: str
    confidence: float  # 0-1


class AdvancedBugDetector:
    """Detector avanzado de bugs."""
    
    def __init__(self):
        self.bug_patterns = {
            BugType.NULL_POINTER: [
                (r'(\w+)\.(\w+)\(\)', "Potential None attribute access"),
                (r'if\s+(\w+)\s*:\s*\n.*?\1\.', "Unchecked None before access"),
            ],
            BugType.INDEX_ERROR: [
                (r'(\w+)\[(\d+|\w+)\]', "Potential index out of bounds"),
                (r'(\w+)\[-1\]', "Negative index without length check"),
            ],
            BugType.DIVISION_BY_ZERO: [
                (r'/\s*(\w+)', "Division by variable without zero check"),
                (r'/\s*0', "Direct division by zero"),
            ],
            BugType.UNDEFINED_VARIABLE: [
                (r'\b(\w+)\s*=', "Variable assignment"),
            ],
            BugType.INFINITE_LOOP: [
                (r'while\s+True:', "Infinite loop without break condition"),
                (r'while\s+(\w+):\s*\n(?!.*break)', "Loop without break"),
            ],
            BugType.RESOURCE_LEAK: [
                (r'open\([^)]+\)', "File open without context manager"),
                (r'\.connect\(\)', "Connection without proper cleanup"),
            ],
        }
    
    def detect_bugs(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Detecta bugs en el código."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "bugs": []
            }
        
        bugs = []
        
        # Detectar patrones conocidos
        for bug_type, patterns in self.bug_patterns.items():
            for pattern, description in patterns:
                bugs.extend(self._detect_pattern(code, pattern, bug_type, description))
        
        # Análisis AST para bugs más complejos
        bugs.extend(self._analyze_ast(tree, code))
        
        # Análisis de lógica
        bugs.extend(self._analyze_logic(code))
        
        # Ordenar por severidad y confianza
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        bugs.sort(key=lambda x: (severity_order.get(x.severity, 0), x.confidence), reverse=True)
        
        return {
            "file_path": file_path,
            "total_bugs": len(bugs),
            "by_severity": self._group_by_severity(bugs),
            "by_type": self._group_by_type(bugs),
            "bugs": [
                {
                    "type": b.bug_type.value,
                    "severity": b.severity,
                    "location": b.location,
                    "line": b.line_number,
                    "description": b.description,
                    "code_snippet": b.code_snippet[:200],
                    "fix_suggestion": b.fix_suggestion,
                    "confidence": b.confidence
                }
                for b in bugs
            ]
        }
    
    def _detect_pattern(
        self, code: str, pattern: str, bug_type: BugType, description: str
    ) -> List[BugReport]:
        """Detecta un patrón de bug."""
        bugs = []
        
        for match in re.finditer(pattern, code, re.MULTILINE):
            line_num = code[:match.start()].count('\n') + 1
            code_snippet = match.group(0)
            
            severity = self._determine_severity(bug_type)
            confidence = self._calculate_confidence(bug_type, code, match)
            
            bugs.append(BugReport(
                bug_type=bug_type,
                severity=severity,
                location=f"Line {line_num}",
                line_number=line_num,
                description=description,
                code_snippet=code_snippet,
                fix_suggestion=self._get_fix_suggestion(bug_type),
                confidence=confidence
            ))
        
        return bugs
    
    def _analyze_ast(self, tree: ast.AST, code: str) -> List[BugReport]:
        """Analiza AST para bugs complejos."""
        bugs = []
        
        # Detectar acceso a atributos sin verificación
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Verificar si el objeto puede ser None
                if isinstance(node.value, ast.Name):
                    var_name = node.value.id
                    # Buscar si hay verificación de None antes
                    line_num = node.lineno
                    context_before = code[:code.find(node.value.id, code.split('\n')[line_num-1].find(node.value.id) if line_num > 0 else 0)]
                    
                    if f'if {var_name}' not in context_before and f'{var_name} is not None' not in context_before:
                        bugs.append(BugReport(
                            bug_type=BugType.NULL_POINTER,
                            severity="high",
                            location=f"Line {line_num}",
                            line_number=line_num,
                            description=f"Potential None attribute access: {var_name}.{node.attr}",
                            code_snippet=f"{var_name}.{node.attr}",
                            fix_suggestion=f"Add None check: if {var_name} is not None:",
                            confidence=0.6
                        ))
        
        # Detectar división sin verificación de cero
        for node in ast.walk(tree):
            if isinstance(node, ast.Div):
                line_num = node.lineno
                # Verificar si el divisor puede ser cero
                if isinstance(node.right, ast.Name):
                    var_name = node.right.id
                    context = code[:code.find(var_name, code.split('\n')[line_num-1].find(var_name) if line_num > 0 else 0)]
                    
                    if f'if {var_name}' not in context and f'{var_name} != 0' not in context:
                        bugs.append(BugReport(
                            bug_type=BugType.DIVISION_BY_ZERO,
                            severity="critical",
                            location=f"Line {line_num}",
                            line_number=line_num,
                            description=f"Potential division by zero: {var_name}",
                            code_snippet=f"... / {var_name}",
                            fix_suggestion=f"Add zero check: if {var_name} != 0:",
                            confidence=0.7
                        ))
        
        return bugs
    
    def _analyze_logic(self, code: str) -> List[BugReport]:
        """Analiza errores lógicos."""
        bugs = []
        
        # Detectar comparaciones incorrectas
        pattern = r'if\s+(\w+)\s*==\s*None:'
        for match in re.finditer(pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            bugs.append(BugReport(
                bug_type=BugType.LOGIC_ERROR,
                severity="medium",
                location=f"Line {line_num}",
                line_number=line_num,
                description="Use 'is None' instead of '== None'",
                code_snippet=match.group(0),
                fix_suggestion="Change to: if var is None:",
                confidence=0.9
            ))
        
        # Detectar asignación en lugar de comparación
        pattern = r'if\s+(\w+)\s*=\s*'
        for match in re.finditer(pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            bugs.append(BugReport(
                bug_type=BugType.LOGIC_ERROR,
                severity="high",
                location=f"Line {line_num}",
                line_number=line_num,
                description="Assignment in condition (should be ==)",
                code_snippet=match.group(0),
                fix_suggestion="Change = to == for comparison",
                confidence=0.95
            ))
        
        return bugs
    
    def _determine_severity(self, bug_type: BugType) -> str:
        """Determina severidad del bug."""
        critical_bugs = [BugType.DIVISION_BY_ZERO, BugType.INFINITE_LOOP]
        high_bugs = [BugType.NULL_POINTER, BugType.INDEX_ERROR, BugType.RESOURCE_LEAK]
        
        if bug_type in critical_bugs:
            return "critical"
        elif bug_type in high_bugs:
            return "high"
        elif bug_type == BugType.LOGIC_ERROR:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, bug_type: BugType, code: str, match: re.Match) -> float:
        """Calcula confianza en la detección."""
        # Confianza base según tipo
        base_confidence = {
            BugType.DIVISION_BY_ZERO: 0.8,
            BugType.INFINITE_LOOP: 0.7,
            BugType.NULL_POINTER: 0.6,
            BugType.INDEX_ERROR: 0.5,
            BugType.LOGIC_ERROR: 0.9,
        }.get(bug_type, 0.5)
        
        # Ajustar según contexto
        context = code[max(0, match.start()-100):match.end()+100]
        
        # Si hay manejo de errores cerca, reducir confianza
        if 'try' in context or 'except' in context:
            base_confidence *= 0.8
        
        return min(1.0, base_confidence)
    
    def _get_fix_suggestion(self, bug_type: BugType) -> str:
        """Obtiene sugerencia de fix."""
        suggestions = {
            BugType.NULL_POINTER: "Add None check before attribute access",
            BugType.INDEX_ERROR: "Check bounds before accessing index",
            BugType.DIVISION_BY_ZERO: "Check for zero before division",
            BugType.UNDEFINED_VARIABLE: "Ensure variable is defined before use",
            BugType.INFINITE_LOOP: "Add break condition or timeout",
            BugType.RESOURCE_LEAK: "Use context manager (with statement)",
            BugType.LOGIC_ERROR: "Review logic and fix comparison/assignment",
        }
        return suggestions.get(bug_type, "Review and fix the issue")
    
    def _group_by_severity(self, bugs: List[BugReport]) -> Dict[str, int]:
        """Agrupa por severidad."""
        grouped = {}
        for bug in bugs:
            grouped[bug.severity] = grouped.get(bug.severity, 0) + 1
        return grouped
    
    def _group_by_type(self, bugs: List[BugReport]) -> Dict[str, int]:
        """Agrupa por tipo."""
        grouped = {}
        for bug in bugs:
            grouped[bug.bug_type.value] = grouped.get(bug.bug_type.value, 0) + 1
        return grouped


# Factory function
_advanced_bug_detector = None

def get_advanced_bug_detector() -> AdvancedBugDetector:
    """Obtiene instancia global del detector."""
    global _advanced_bug_detector
    if _advanced_bug_detector is None:
        _advanced_bug_detector = AdvancedBugDetector()
    return _advanced_bug_detector

