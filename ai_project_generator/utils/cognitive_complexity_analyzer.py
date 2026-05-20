"""
Cognitive Complexity Analyzer
==============================

Sistema de análisis de complejidad cognitiva.
"""

import ast
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetric:
    """Métrica de complejidad."""
    function_name: str
    cognitive_complexity: int
    cyclomatic_complexity: int
    nesting_level: int
    line_number: int
    recommendation: str


class CognitiveComplexityAnalyzer:
    """Analizador de complejidad cognitiva."""
    
    def __init__(self):
        self.complexity_threshold = 15
        self.nesting_threshold = 4
    
    def analyze_complexity(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Analiza complejidad cognitiva."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "metrics": []
            }
        
        metrics = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                cognitive = self._calculate_cognitive_complexity(node)
                cyclomatic = self._calculate_cyclomatic_complexity(node)
                nesting = self._calculate_max_nesting(node)
                
                recommendation = self._get_recommendation(cognitive, cyclomatic, nesting)
                
                metrics.append(ComplexityMetric(
                    function_name=node.name,
                    cognitive_complexity=cognitive,
                    cyclomatic_complexity=cyclomatic,
                    nesting_level=nesting,
                    line_number=node.lineno,
                    recommendation=recommendation
                ))
        
        # Ordenar por complejidad cognitiva
        metrics.sort(key=lambda x: x.cognitive_complexity, reverse=True)
        
        return {
            "file_path": file_path,
            "total_functions": len(metrics),
            "average_cognitive_complexity": sum(m.cognitive_complexity for m in metrics) / len(metrics) if metrics else 0,
            "average_cyclomatic_complexity": sum(m.cyclomatic_complexity for m in metrics) / len(metrics) if metrics else 0,
            "high_complexity_functions": [
                m.function_name for m in metrics
                if m.cognitive_complexity > self.complexity_threshold
            ],
            "metrics": [
                {
                    "function": m.function_name,
                    "cognitive_complexity": m.cognitive_complexity,
                    "cyclomatic_complexity": m.cyclomatic_complexity,
                    "nesting_level": m.nesting_level,
                    "line": m.line_number,
                    "recommendation": m.recommendation
                }
                for m in metrics
            ]
        }
    
    def _calculate_cognitive_complexity(self, node: ast.FunctionDef) -> int:
        """Calcula complejidad cognitiva."""
        complexity = 0
        
        for child in ast.walk(node):
            # Incrementar por estructuras de control
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
                
                # Bonificación por anidamiento
                nesting = self._get_nesting_level(child, node)
                complexity += nesting
            
            # Incrementar por operadores lógicos
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calcula complejidad ciclomática."""
        complexity = 1  # Base
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_max_nesting(self, node: ast.FunctionDef) -> int:
        """Calcula nivel máximo de anidamiento."""
        max_nesting = 0
        
        def get_nesting(n, current=0):
            nonlocal max_nesting
            max_nesting = max(max_nesting, current)
            
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                    get_nesting(child, current + 1)
                else:
                    get_nesting(child, current)
        
        get_nesting(node)
        return max_nesting
    
    def _get_nesting_level(self, node: ast.AST, root: ast.FunctionDef) -> int:
        """Obtiene nivel de anidamiento de un nodo."""
        level = 0
        current = node
        
        while current != root:
            parent = None
            for child in ast.walk(root):
                if child == current:
                    # Encontrar padre (simplificado)
                    break
            level += 1
            if level > 10:  # Prevenir loops infinitos
                break
        
        return level
    
    def _get_recommendation(self, cognitive: int, cyclomatic: int, nesting: int) -> str:
        """Obtiene recomendación basada en complejidad."""
        if cognitive > self.complexity_threshold:
            return "High cognitive complexity - consider refactoring into smaller functions"
        elif nesting > self.nesting_threshold:
            return "High nesting level - consider extracting nested logic"
        elif cyclomatic > 10:
            return "High cyclomatic complexity - consider simplifying control flow"
        else:
            return "Complexity is acceptable"


# Factory function
_cognitive_complexity_analyzer = None

def get_cognitive_complexity_analyzer() -> CognitiveComplexityAnalyzer:
    """Obtiene instancia global del analizador."""
    global _cognitive_complexity_analyzer
    if _cognitive_complexity_analyzer is None:
        _cognitive_complexity_analyzer = CognitiveComplexityAnalyzer()
    return _cognitive_complexity_analyzer

