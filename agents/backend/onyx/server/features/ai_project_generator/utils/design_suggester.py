"""
Design Suggester
================

Sistema de sugerencias de diseño de código.
"""

import ast
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DesignSuggestionType(Enum):
    """Tipos de sugerencias de diseño."""
    PATTERN_APPLICATION = "pattern_application"
    CODE_ORGANIZATION = "code_organization"
    INTERFACE_DESIGN = "interface_design"
    DATA_STRUCTURE = "data_structure"
    ALGORITHM_IMPROVEMENT = "algorithm_improvement"


@dataclass
class DesignSuggestion:
    """Sugerencia de diseño."""
    type: DesignSuggestionType
    title: str
    description: str
    current_approach: str
    suggested_approach: str
    benefits: List[str]
    implementation_hint: str
    priority: int  # 1-10


class DesignSuggester:
    """Sugeridor de diseño."""
    
    def __init__(self):
        self.suggestion_generators = {
            DesignSuggestionType.PATTERN_APPLICATION: self._suggest_patterns,
            DesignSuggestionType.CODE_ORGANIZATION: self._suggest_organization,
            DesignSuggestionType.INTERFACE_DESIGN: self._suggest_interfaces,
            DesignSuggestionType.DATA_STRUCTURE: self._suggest_data_structures,
            DesignSuggestionType.ALGORITHM_IMPROVEMENT: self._suggest_algorithm_improvements,
        }
    
    def suggest_improvements(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Sugiere mejoras de diseño."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "suggestions": []
            }
        
        suggestions = []
        
        # Generar sugerencias de cada tipo
        for suggestion_type, generator in self.suggestion_generators.items():
            suggestions.extend(generator(code, tree))
        
        # Ordenar por prioridad
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        return {
            "file_path": file_path,
            "total_suggestions": len(suggestions),
            "suggestions": [
                {
                    "type": s.type.value,
                    "title": s.title,
                    "description": s.description,
                    "current_approach": s.current_approach,
                    "suggested_approach": s.suggested_approach,
                    "benefits": s.benefits,
                    "implementation_hint": s.implementation_hint,
                    "priority": s.priority
                }
                for s in suggestions
            ],
            "by_type": self._group_by_type(suggestions)
        }
    
    def _suggest_patterns(self, code: str, tree: ast.AST) -> List[DesignSuggestion]:
        """Sugiere patrones de diseño."""
        suggestions = []
        
        # Detectar si se necesita Factory
        if 'if' in code and 'elif' in code and code.count('if') > 3:
            suggestions.append(DesignSuggestion(
                type=DesignSuggestionType.PATTERN_APPLICATION,
                title="Consider Factory Pattern",
                description="Multiple conditional object creation detected",
                current_approach="Multiple if/elif statements for object creation",
                suggested_approach="Use Factory pattern to encapsulate object creation",
                benefits=[
                    "Reduces coupling",
                    "Easier to extend",
                    "Centralizes creation logic"
                ],
                implementation_hint="Create a Factory class with a create() method",
                priority=7
            ))
        
        # Detectar si se necesita Strategy
        if code.count('if') > 5 and 'algorithm' in code.lower():
            suggestions.append(DesignSuggestion(
                type=DesignSuggestionType.PATTERN_APPLICATION,
                title="Consider Strategy Pattern",
                description="Multiple conditional algorithms detected",
                current_approach="Multiple if/elif for different algorithms",
                suggested_approach="Use Strategy pattern to encapsulate algorithms",
                benefits=[
                    "Easier to add new algorithms",
                    "Reduces complexity",
                    "Better testability"
                ],
                implementation_hint="Create Strategy interface and concrete implementations",
                priority=8
            ))
        
        return suggestions
    
    def _suggest_organization(self, code: str, tree: ast.AST) -> List[DesignSuggestion]:
        """Sugiere mejoras de organización."""
        suggestions = []
        
        # Detectar archivo muy largo
        lines = code.split('\n')
        if len(lines) > 500:
            suggestions.append(DesignSuggestion(
                type=DesignSuggestionType.CODE_ORGANIZATION,
                title="Split Large File",
                description=f"File has {len(lines)} lines, consider splitting",
                current_approach="All code in single file",
                suggested_approach="Split into multiple modules by responsibility",
                benefits=[
                    "Better maintainability",
                    "Easier navigation",
                    "Reduced cognitive load"
                ],
                implementation_hint="Group related classes/functions into separate modules",
                priority=9
            ))
        
        # Detectar clases muy grandes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
                if len(methods) > 20:
                    suggestions.append(DesignSuggestion(
                        type=DesignSuggestionType.CODE_ORGANIZATION,
                        title=f"Refactor Large Class: {node.name}",
                        description=f"Class {node.name} has {len(methods)} methods",
                        current_approach=f"Single class with {len(methods)} methods",
                        suggested_approach="Split into multiple smaller classes",
                        benefits=[
                            "Single Responsibility Principle",
                            "Better testability",
                            "Reduced complexity"
                        ],
                        implementation_hint="Identify responsibilities and extract into separate classes",
                        priority=8
                    ))
        
        return suggestions
    
    def _suggest_interfaces(self, code: str, tree: ast.AST) -> List[DesignSuggestion]:
        """Sugiere mejoras de interfaces."""
        suggestions = []
        
        # Detectar funciones con muchos parámetros
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                params = len(node.args.args)
                if params > 5:
                    suggestions.append(DesignSuggestion(
                        type=DesignSuggestionType.INTERFACE_DESIGN,
                        title=f"Simplify Function Interface: {node.name}",
                        description=f"Function {node.name} has {params} parameters",
                        current_approach=f"Function with {params} parameters",
                        suggested_approach="Use data class or configuration object",
                        benefits=[
                            "Easier to use",
                            "More maintainable",
                            "Better extensibility"
                        ],
                        implementation_hint="Create a dataclass for parameters",
                        priority=6
                    ))
        
        return suggestions
    
    def _suggest_data_structures(self, code: str, tree: ast.AST) -> List[DesignSuggestion]:
        """Sugiere mejoras de estructuras de datos."""
        suggestions = []
        
        # Detectar uso de listas donde dict sería mejor
        if 'in [' in code and code.count('in [') > 3:
            suggestions.append(DesignSuggestion(
                type=DesignSuggestionType.DATA_STRUCTURE,
                title="Consider Using Dictionary",
                description="Multiple 'in list' operations detected",
                current_approach="Using list for lookups",
                suggested_approach="Use dictionary for O(1) lookups",
                benefits=[
                    "Faster lookups",
                    "Better performance",
                    "More semantic"
                ],
                implementation_hint="Replace list with dict for key-value lookups",
                priority=7
            ))
        
        return suggestions
    
    def _suggest_algorithm_improvements(self, code: str, tree: ast.AST) -> List[DesignSuggestion]:
        """Sugiere mejoras de algoritmos."""
        suggestions = []
        
        # Detectar loops anidados
        nested_loops = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.For):
                        nested_loops += 1
                        break
        
        if nested_loops > 2:
            suggestions.append(DesignSuggestion(
                type=DesignSuggestionType.ALGORITHM_IMPROVEMENT,
                title="Optimize Nested Loops",
                description="Multiple nested loops detected",
                current_approach="Nested loops with O(n²) or worse complexity",
                suggested_approach="Use more efficient algorithm or data structure",
                benefits=[
                    "Better performance",
                    "Reduced time complexity",
                    "Scalability"
                ],
                implementation_hint="Consider using sets, dicts, or optimized algorithms",
                priority=9
            ))
        
        return suggestions
    
    def _group_by_type(self, suggestions: List[DesignSuggestion]) -> Dict[str, int]:
        """Agrupa por tipo."""
        grouped = {}
        for s in suggestions:
            grouped[s.type.value] = grouped.get(s.type.value, 0) + 1
        return grouped


# Factory function
_design_suggester = None

def get_design_suggester() -> DesignSuggester:
    """Obtiene instancia global del sugeridor."""
    global _design_suggester
    if _design_suggester is None:
        _design_suggester = DesignSuggester()
    return _design_suggester

