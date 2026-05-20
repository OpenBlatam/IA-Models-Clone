"""
Auto Refactor
=============

Sistema de refactoring automático de código.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RefactorType(Enum):
    """Tipos de refactoring."""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_VARIABLE = "extract_variable"
    RENAME = "rename"
    SIMPLIFY = "simplify"
    REMOVE_DUPLICATION = "remove_duplication"
    IMPROVE_NAMING = "improve_naming"
    ADD_TYPE_HINTS = "add_type_hints"
    ADD_DOCSTRINGS = "add_docstrings"


@dataclass
class RefactorSuggestion:
    """Sugerencia de refactoring."""
    type: RefactorType
    description: str
    code_before: str
    code_after: str
    line_number: int
    confidence: float  # 0-1
    impact: str


class AutoRefactor:
    """Sistema de refactoring automático."""
    
    def __init__(self):
        self.refactor_rules = {
            RefactorType.EXTRACT_METHOD: self._suggest_extract_method,
            RefactorType.EXTRACT_VARIABLE: self._suggest_extract_variable,
            RefactorType.SIMPLIFY: self._suggest_simplify,
            RefactorType.IMPROVE_NAMING: self._suggest_improve_naming,
            RefactorType.ADD_TYPE_HINTS: self._suggest_add_type_hints,
            RefactorType.ADD_DOCSTRINGS: self._suggest_add_docstrings,
        }
    
    def analyze_and_refactor(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Analiza código y sugiere refactorings."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "suggestions": []
            }
        
        suggestions = []
        
        # Aplicar todas las reglas de refactoring
        for refactor_type, rule_func in self.refactor_rules.items():
            try:
                rule_suggestions = rule_func(code, tree)
                suggestions.extend(rule_suggestions)
            except Exception as e:
                logger.warning(f"Error in refactor rule {rule_func.__name__}: {e}")
        
        # Ordenar por confianza
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return {
            "file_path": file_path,
            "total_suggestions": len(suggestions),
            "suggestions": [
                {
                    "type": s.type.value,
                    "description": s.description,
                    "line": s.line_number,
                    "confidence": s.confidence,
                    "impact": s.impact,
                    "code_before": s.code_before[:300],
                    "code_after": s.code_after[:300]
                }
                for s in suggestions
            ],
            "by_type": self._group_by_type(suggestions)
        }
    
    def _suggest_extract_method(self, code: str, tree: ast.AST) -> List[RefactorSuggestion]:
        """Sugiere extraer métodos de funciones largas."""
        suggestions = []
        lines = code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                
                # Si la función es muy larga (>50 líneas), sugerir extraer
                if func_lines > 50:
                    # Buscar bloques que puedan extraerse
                    for child in node.body:
                        if isinstance(child, (ast.If, ast.For, ast.While)):
                            block_lines = child.end_lineno - child.lineno if hasattr(child, 'end_lineno') else 0
                            if block_lines > 10:
                                line_num = child.lineno
                                block_code = '\n'.join(lines[child.lineno-1:child.end_lineno if hasattr(child, 'end_lineno') else child.lineno])
                                
                                suggestions.append(RefactorSuggestion(
                                    type=RefactorType.EXTRACT_METHOD,
                                    description=f"Extract {type(child).__name__} block into separate method",
                                    code_before=block_code[:200],
                                    code_after=f"def extracted_method():\n    {block_code[:100]}",
                                    line_number=line_num,
                                    confidence=0.7,
                                    impact="Improves readability and testability"
                                ))
        
        return suggestions
    
    def _suggest_extract_variable(self, code: str, tree: ast.AST) -> List[RefactorSuggestion]:
        """Sugiere extraer variables de expresiones complejas."""
        suggestions = []
        
        # Buscar expresiones complejas anidadas
        pattern = r'(\w+)\s*=\s*([^=]+\([^)]+\([^)]+\)[^)]+\))'
        
        for match in re.finditer(pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            expr = match.group(2)
            
            if len(expr) > 80:  # Expresión muy larga
                var_name = match.group(1)
                suggestions.append(RefactorSuggestion(
                    type=RefactorType.EXTRACT_VARIABLE,
                    description=f"Extract complex expression into variable",
                    code_before=match.group(0)[:200],
                    code_after=f"# Extract intermediate values\n{var_name} = simplified_expression",
                    line_number=line_num,
                    confidence=0.6,
                    impact="Improves readability"
                ))
        
        return suggestions
    
    def _suggest_simplify(self, code: str, tree: ast.AST) -> List[RefactorSuggestion]:
        """Sugiere simplificaciones."""
        suggestions = []
        
        # Simplificar condiciones complejas
        pattern = r'if\s+(.+?)\s+and\s+(.+?)\s+and\s+(.+?):'
        
        for match in re.finditer(pattern, code, re.MULTILINE):
            line_num = code[:match.start()].count('\n') + 1
            simplified = f"if all([{match.group(1)}, {match.group(2)}, {match.group(3)}]):"
            
            suggestions.append(RefactorSuggestion(
                type=RefactorType.SIMPLIFY,
                description="Simplify complex condition",
                code_before=match.group(0),
                code_after=simplified,
                line_number=line_num,
                confidence=0.5,
                impact="Improves readability"
            ))
        
        return suggestions
    
    def _suggest_improve_naming(self, code: str, tree: ast.AST) -> List[RefactorSuggestion]:
        """Sugiere mejorar nombres de variables/funciones."""
        suggestions = []
        
        # Buscar nombres de una letra o muy cortos
        pattern = r'\b([a-z])\s*='
        
        for match in re.finditer(pattern, code):
            var_name = match.group(1)
            line_num = code[:match.start()].count('\n') + 1
            
            # Solo sugerir si no es una variable común (i, j, k en loops)
            context = code[max(0, match.start()-20):match.start()]
            if 'for' not in context and 'while' not in context:
                suggestions.append(RefactorSuggestion(
                    type=RefactorType.IMPROVE_NAMING,
                    description=f"Improve variable name '{var_name}' to be more descriptive",
                    code_before=f"{var_name} = ...",
                    code_after=f"descriptive_name = ...",
                    line_number=line_num,
                    confidence=0.8,
                    impact="Improves code clarity"
                ))
        
        return suggestions
    
    def _suggest_add_type_hints(self, code: str, tree: ast.AST) -> List[RefactorSuggestion]:
        """Sugiere agregar type hints."""
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_return_hint = node.returns is not None
                has_param_hints = any(arg.annotation is not None for arg in node.args.args)
                
                if not has_return_hint or not has_param_hints:
                    line_num = node.lineno
                    params = ', '.join([arg.arg for arg in node.args.args])
                    
                    suggestions.append(RefactorSuggestion(
                        type=RefactorType.ADD_TYPE_HINTS,
                        description=f"Add type hints to function {node.name}",
                        code_before=f"def {node.name}({params}):",
                        code_after=f"def {node.name}({params}: str) -> None:",
                        line_number=line_num,
                        confidence=0.9,
                        impact="Improves type safety and IDE support"
                    ))
        
        return suggestions
    
    def _suggest_add_docstrings(self, code: str, tree: ast.AST) -> List[RefactorSuggestion]:
        """Sugiere agregar docstrings."""
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if ast.get_docstring(node) is None:
                    line_num = node.lineno
                    name = node.name
                    node_type = "function" if isinstance(node, ast.FunctionDef) else "class"
                    
                    suggestions.append(RefactorSuggestion(
                        type=RefactorType.ADD_DOCSTRINGS,
                        description=f"Add docstring to {node_type} {name}",
                        code_before=f"{'def' if isinstance(node, ast.FunctionDef) else 'class'} {name}(...):",
                        code_after=f"{'def' if isinstance(node, ast.FunctionDef) else 'class'} {name}(...):\n    \"\"\"Description of {name}.\"\"\"",
                        line_number=line_num,
                        confidence=0.95,
                        impact="Improves documentation"
                    ))
        
        return suggestions
    
    def _group_by_type(self, suggestions: List[RefactorSuggestion]) -> Dict[str, int]:
        """Agrupa sugerencias por tipo."""
        grouped = {}
        for s in suggestions:
            grouped[s.type.value] = grouped.get(s.type.value, 0) + 1
        return grouped
    
    def apply_refactor(self, code: str, suggestion: RefactorSuggestion) -> str:
        """Aplica un refactoring sugerido (simplificado)."""
        # Esta es una implementación simplificada
        # En producción, se necesitaría un sistema más robusto
        lines = code.split('\n')
        
        if suggestion.line_number <= len(lines):
            # Aquí se aplicaría el refactoring real
            # Por ahora, solo retornamos el código original
            return code
        
        return code


# Factory function
_auto_refactor = None

def get_auto_refactor() -> AutoRefactor:
    """Obtiene instancia global del refactor."""
    global _auto_refactor
    if _auto_refactor is None:
        _auto_refactor = AutoRefactor()
    return _auto_refactor

