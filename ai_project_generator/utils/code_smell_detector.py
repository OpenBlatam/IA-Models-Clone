"""
Code Smell Detector
===================

Sistema de detección de code smells.
"""

import ast
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CodeSmellType(Enum):
    """Tipos de code smells."""
    LONG_METHOD = "long_method"
    LONG_PARAMETER_LIST = "long_parameter_list"
    DUPLICATE_CODE = "duplicate_code"
    LARGE_CLASS = "large_class"
    DATA_CLASS = "data_class"
    FEATURE_ENVY = "feature_envy"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    SWITCH_STATEMENTS = "switch_statements"
    SPECULATIVE_GENERALITY = "speculative_generality"
    DEAD_CODE = "dead_code"
    COMMENTS = "too_many_comments"


@dataclass
class CodeSmell:
    """Code smell detectado."""
    smell_type: CodeSmellType
    severity: str
    location: str
    line_number: int
    description: str
    suggestion: str
    refactoring: str


class CodeSmellDetector:
    """Detector de code smells."""
    
    def __init__(self):
        self.thresholds = {
            CodeSmellType.LONG_METHOD: 50,  # líneas
            CodeSmellType.LONG_PARAMETER_LIST: 5,  # parámetros
            CodeSmellType.LARGE_CLASS: 20,  # métodos
        }
    
    def detect_smells(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Detecta code smells."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "smells": []
            }
        
        smells = []
        
        # Detectar diferentes tipos de smells
        smells.extend(self._detect_long_methods(tree, code))
        smells.extend(self._detect_long_parameter_lists(tree))
        smells.extend(self._detect_large_classes(tree))
        smells.extend(self._detect_duplicate_code(code))
        smells.extend(self._detect_data_classes(tree))
        smells.extend(self._detect_primitive_obsession(code))
        smells.extend(self._detect_dead_code(code, tree))
        
        # Ordenar por severidad
        severity_order = {"high": 3, "medium": 2, "low": 1}
        smells.sort(key=lambda x: severity_order.get(x.severity, 0), reverse=True)
        
        return {
            "file_path": file_path,
            "total_smells": len(smells),
            "by_type": self._group_by_type(smells),
            "by_severity": self._group_by_severity(smells),
            "smells": [
                {
                    "type": s.smell_type.value,
                    "severity": s.severity,
                    "location": s.location,
                    "line": s.line_number,
                    "description": s.description,
                    "suggestion": s.suggestion,
                    "refactoring": s.refactoring
                }
                for s in smells
            ]
        }
    
    def _detect_long_methods(self, tree: ast.AST, code: str) -> List[CodeSmell]:
        """Detecta métodos largos."""
        smells = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                
                if lines > self.thresholds[CodeSmellType.LONG_METHOD]:
                    severity = "high" if lines > 100 else "medium"
                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.LONG_METHOD,
                        severity=severity,
                        location=f"Function {node.name}",
                        line_number=node.lineno,
                        description=f"Method {node.name} has {lines} lines (threshold: {self.thresholds[CodeSmellType.LONG_METHOD]})",
                        suggestion="Extract methods to reduce complexity",
                        refactoring="Extract Method"
                    ))
        
        return smells
    
    def _detect_long_parameter_lists(self, tree: ast.AST) -> List[CodeSmell]:
        """Detecta listas largas de parámetros."""
        smells = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                params = len(node.args.args)
                
                if params > self.thresholds[CodeSmellType.LONG_PARAMETER_LIST]:
                    severity = "high" if params > 8 else "medium"
                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.LONG_PARAMETER_LIST,
                        severity=severity,
                        location=f"Function {node.name}",
                        line_number=node.lineno,
                        description=f"Function {node.name} has {params} parameters (threshold: {self.thresholds[CodeSmellType.LONG_PARAMETER_LIST]})",
                        suggestion="Use data class or configuration object",
                        refactoring="Introduce Parameter Object"
                    ))
        
        return smells
    
    def _detect_large_classes(self, tree: ast.AST) -> List[CodeSmell]:
        """Detecta clases grandes."""
        smells = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
                
                if len(methods) > self.thresholds[CodeSmellType.LARGE_CLASS]:
                    severity = "high" if len(methods) > 30 else "medium"
                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.LARGE_CLASS,
                        severity=severity,
                        location=f"Class {node.name}",
                        line_number=node.lineno,
                        description=f"Class {node.name} has {len(methods)} methods (threshold: {self.thresholds[CodeSmellType.LARGE_CLASS]})",
                        suggestion="Split into multiple smaller classes",
                        refactoring="Extract Class"
                    ))
        
        return smells
    
    def _detect_duplicate_code(self, code: str) -> List[CodeSmell]:
        """Detecta código duplicado."""
        smells = []
        
        lines = code.split('\n')
        seen_blocks = {}
        
        # Detectar bloques similares
        for i in range(len(lines) - 5):
            block = '\n'.join(lines[i:i+5])
            block_hash = hash(block.strip())
            
            if block_hash in seen_blocks:
                smells.append(CodeSmell(
                    smell_type=CodeSmellType.DUPLICATE_CODE,
                    severity="medium",
                    location=f"Lines {i+1}-{i+5}",
                    line_number=i+1,
                    description="Duplicate code block detected",
                    suggestion="Extract to common function",
                    refactoring="Extract Method"
                ))
            else:
                seen_blocks[block_hash] = i
        
        return smells[:10]  # Limitar a 10
    
    def _detect_data_classes(self, tree: ast.AST) -> List[CodeSmell]:
        """Detecta data classes (clases que solo almacenan datos)."""
        smells = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
                # Filtrar métodos especiales y getters/setters
                non_trivial_methods = [
                    m for m in methods
                    if not m.name.startswith('__') and m.name not in ['get', 'set']
                ]
                
                # Si solo tiene métodos triviales, es una data class
                if len(non_trivial_methods) == 0 and len(methods) > 0:
                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.DATA_CLASS,
                        severity="low",
                        location=f"Class {node.name}",
                        line_number=node.lineno,
                        description=f"Class {node.name} appears to be a data class",
                        suggestion="Consider using dataclass decorator or add behavior",
                        refactoring="Replace Data Class with Data Class decorator"
                    ))
        
        return smells
    
    def _detect_primitive_obsession(self, code: str) -> List[CodeSmell]:
        """Detecta obsesión por primitivos."""
        smells = []
        
        # Detectar múltiples parámetros primitivos relacionados
        pattern = r'def\s+\w+\([^)]*(?:str|int|float)[^)]*(?:str|int|float)[^)]*(?:str|int|float)'
        
        for match in re.finditer(pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            smells.append(CodeSmell(
                smell_type=CodeSmellType.PRIMITIVE_OBSESSION,
                severity="low",
                location=f"Line {line_num}",
                line_number=line_num,
                description="Multiple primitive parameters that could be grouped",
                suggestion="Create a value object or data class",
                refactoring="Replace Primitive with Object"
            ))
        
        return smells[:5]  # Limitar
    
    def _detect_dead_code(self, code: str, tree: ast.AST) -> List[CodeSmell]:
        """Detecta código muerto."""
        smells = []
        
        # Detectar funciones no usadas (simplificado)
        defined_functions = set()
        called_functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_functions.add(node.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_functions.add(node.func.id)
        
        unused = defined_functions - called_functions - {'__init__', '__str__', '__repr__'}
        
        for func_name in unused:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.DEAD_CODE,
                        severity="low",
                        location=f"Function {func_name}",
                        line_number=node.lineno,
                        description=f"Function {func_name} is defined but never called",
                        suggestion="Remove if unused or make it public if needed",
                        refactoring="Remove Dead Code"
                    ))
                    break
        
        return smells
    
    def _group_by_type(self, smells: List[CodeSmell]) -> Dict[str, int]:
        """Agrupa por tipo."""
        grouped = {}
        for s in smells:
            grouped[s.smell_type.value] = grouped.get(s.smell_type.value, 0) + 1
        return grouped
    
    def _group_by_severity(self, smells: List[CodeSmell]) -> Dict[str, int]:
        """Agrupa por severidad."""
        grouped = {}
        for s in smells:
            grouped[s.severity] = grouped.get(s.severity, 0) + 1
        return grouped


# Factory function
_code_smell_detector = None

def get_code_smell_detector() -> CodeSmellDetector:
    """Obtiene instancia global del detector."""
    global _code_smell_detector
    if _code_smell_detector is None:
        _code_smell_detector = CodeSmellDetector()
    return _code_smell_detector

