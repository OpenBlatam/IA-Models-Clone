"""
Code Quality Analyzer
=====================

Sistema de análisis de calidad de código con métricas avanzadas.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Métricas de calidad."""
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"
    CODE_DUPLICATION = "code_duplication"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class QualityScore:
    """Puntuación de calidad."""
    metric: QualityMetric
    score: float  # 0-100
    grade: str  # A, B, C, D, F
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class FunctionMetrics:
    """Métricas de una función."""
    name: str
    complexity: int
    lines: int
    parameters: int
    has_docstring: bool
    has_type_hints: bool
    has_error_handling: bool


class CodeQualityAnalyzer:
    """Analizador de calidad de código."""
    
    def __init__(self):
        self.complexity_threshold = 10
        self.maintainability_threshold = 70
    
    def analyze_file(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Analiza calidad de un archivo completo."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "valid": False
            }
        
        functions = self._extract_functions(tree)
        classes = self._extract_classes(tree)
        
        metrics = {
            "file_path": file_path,
            "valid": True,
            "total_lines": len(code.split('\n')),
            "total_functions": len(functions),
            "total_classes": len(classes),
            "complexity": self._calculate_complexity(tree, functions),
            "maintainability_index": self._calculate_maintainability_index(
                code, functions, classes
            ),
            "documentation_score": self._calculate_documentation_score(
                functions, classes
            ),
            "code_duplication": self._detect_duplication(code),
            "security_issues": self._detect_security_issues(code),
            "performance_issues": self._detect_performance_issues(code),
            "function_metrics": [
                {
                    "name": f.name,
                    "complexity": f.complexity,
                    "lines": f.lines,
                    "parameters": f.parameters,
                    "has_docstring": f.has_docstring,
                    "has_type_hints": f.has_type_hints,
                    "has_error_handling": f.has_error_handling
                }
                for f in functions
            ],
            "overall_score": 0,
            "grades": {}
        }
        
        # Calcular puntuación general
        scores = self._calculate_scores(metrics)
        metrics["overall_score"] = sum(s.score for s in scores) / len(scores)
        metrics["grades"] = {s.metric.value: s.grade for s in scores}
        metrics["detailed_scores"] = [
            {
                "metric": s.metric.value,
                "score": s.score,
                "grade": s.grade,
                "details": s.details,
                "recommendations": s.recommendations
            }
            for s in scores
        ]
        
        return metrics
    
    def _extract_functions(self, tree: ast.AST) -> List[FunctionMetrics]:
        """Extrae métricas de funciones."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_function_complexity(node)
                lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                params = len(node.args.args)
                has_docstring = ast.get_docstring(node) is not None
                has_type_hints = any(
                    arg.annotation is not None for arg in node.args.args
                ) or node.returns is not None
                has_error_handling = self._has_error_handling(node)
                
                functions.append(FunctionMetrics(
                    name=node.name,
                    complexity=complexity,
                    lines=lines,
                    parameters=params,
                    has_docstring=has_docstring,
                    has_type_hints=has_type_hints,
                    has_error_handling=has_error_handling
                ))
        
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extrae información de clases."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                has_docstring = ast.get_docstring(node) is not None
                
                classes.append({
                    "name": node.name,
                    "methods": len(methods),
                    "has_docstring": has_docstring
                })
        
        return classes
    
    def _calculate_complexity(self, tree: ast.AST, functions: List[FunctionMetrics]) -> Dict[str, Any]:
        """Calcula complejidad ciclomática."""
        total_complexity = sum(f.complexity for f in functions)
        avg_complexity = total_complexity / len(functions) if functions else 0
        max_complexity = max((f.complexity for f in functions), default=0)
        
        high_complexity_functions = [
            f.name for f in functions if f.complexity > self.complexity_threshold
        ]
        
        return {
            "total": total_complexity,
            "average": avg_complexity,
            "max": max_complexity,
            "high_complexity_functions": high_complexity_functions
        }
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calcula complejidad ciclomática de una función."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_maintainability_index(
        self, code: str, functions: List[FunctionMetrics], classes: List[Dict]
    ) -> float:
        """Calcula índice de mantenibilidad (0-100)."""
        # Fórmula simplificada basada en Halstead Volume y Complejidad Ciclomática
        lines = len(code.split('\n'))
        total_complexity = sum(f.complexity for f in functions)
        
        if lines == 0:
            return 100.0
        
        # Calcular volumen de Halstead simplificado
        volume = self._calculate_halstead_volume(code)
        
        # Fórmula de mantenibilidad (simplificada)
        maintainability = 171 - 5.2 * math.log(volume) - 0.23 * total_complexity - 16.2 * math.log(lines)
        
        # Normalizar a 0-100
        maintainability = max(0, min(100, maintainability))
        
        return maintainability
    
    def _calculate_halstead_volume(self, code: str) -> float:
        """Calcula volumen de Halstead simplificado."""
        # Contar operadores y operandos únicos
        operators = set()
        operands = set()
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.operator):
                    operators.add(type(node).__name__)
                elif isinstance(node, ast.Name):
                    operands.add(node.id)
        except:
            pass
        
        n1 = len(operators)  # Operadores únicos
        n2 = len(operands)   # Operandos únicos
        N1 = code.count('+') + code.count('-') + code.count('*') + code.count('/')
        N2 = len(re.findall(r'\b\w+\b', code))
        
        if n1 == 0 or n2 == 0:
            return 1.0
        
        volume = (N1 + N2) * math.log2(n1 + n2)
        return max(1.0, volume)
    
    def _calculate_documentation_score(
        self, functions: List[FunctionMetrics], classes: List[Dict]
    ) -> float:
        """Calcula puntuación de documentación."""
        if not functions and not classes:
            return 100.0
        
        documented_functions = sum(1 for f in functions if f.has_docstring)
        documented_classes = sum(1 for c in classes if c.get("has_docstring", False))
        
        total_items = len(functions) + len(classes)
        documented_items = documented_functions + documented_classes
        
        if total_items == 0:
            return 100.0
        
        score = (documented_items / total_items) * 100
        return score
    
    def _detect_duplication(self, code: str) -> Dict[str, Any]:
        """Detecta duplicación de código."""
        lines = code.split('\n')
        line_hashes = {}
        duplicates = []
        
        # Detectar líneas duplicadas
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 10:  # Ignorar líneas muy cortas
                if stripped in line_hashes:
                    duplicates.append({
                        "line": i + 1,
                        "duplicate_of": line_hashes[stripped],
                        "content": stripped[:50]
                    })
                else:
                    line_hashes[stripped] = i + 1
        
        # Detectar bloques similares (simplificado)
        blocks = self._extract_code_blocks(code)
        similar_blocks = []
        
        for i, block1 in enumerate(blocks):
            for j, block2 in enumerate(blocks[i+1:], i+1):
                similarity = self._calculate_similarity(block1, block2)
                if similarity > 0.8:
                    similar_blocks.append({
                        "block1": i,
                        "block2": j,
                        "similarity": similarity
                    })
        
        return {
            "duplicate_lines": len(duplicates),
            "similar_blocks": len(similar_blocks),
            "duplication_percentage": (len(duplicates) / len(lines) * 100) if lines else 0,
            "details": {
                "duplicate_lines": duplicates[:10],  # Limitar a 10
                "similar_blocks": similar_blocks[:5]  # Limitar a 5
            }
        }
    
    def _extract_code_blocks(self, code: str) -> List[str]:
        """Extrae bloques de código."""
        blocks = []
        lines = code.split('\n')
        current_block = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                current_block.append(stripped)
            else:
                if len(current_block) > 3:
                    blocks.append('\n'.join(current_block))
                current_block = []
        
        if len(current_block) > 3:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    def _calculate_similarity(self, block1: str, block2: str) -> float:
        """Calcula similitud entre dos bloques."""
        words1 = set(block1.split())
        words2 = set(block2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _detect_security_issues(self, code: str) -> List[Dict[str, Any]]:
        """Detecta problemas de seguridad."""
        issues = []
        
        patterns = [
            (r'eval\s*\(', "Use of eval() - security risk"),
            (r'exec\s*\(', "Use of exec() - security risk"),
            (r'__import__\s*\(', "Use of __import__() - security risk"),
            (r'pickle\.loads\s*\(', "Use of pickle.loads() - security risk"),
            (r'shell\s*=\s*True', "shell=True in subprocess - security risk"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
        ]
        
        for pattern, message in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "line": line_num,
                    "severity": "high",
                    "message": message,
                    "code": match.group(0)[:50]
                })
        
        return issues
    
    def _detect_performance_issues(self, code: str) -> List[Dict[str, Any]]:
        """Detecta problemas de rendimiento."""
        issues = []
        
        patterns = [
            (r'for\s+.*?:\s*\n.*?SELECT', "Database query in loop"),
            (r'\.append\s*\(.*?\)\s*\n\s*for', "List append in loop - use comprehension"),
            (r'\+=\s*["\']', "String concatenation in loop"),
        ]
        
        for pattern, message in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE | re.DOTALL):
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "line": line_num,
                    "severity": "medium",
                    "message": message,
                    "code": match.group(0)[:50]
                })
        
        return issues
    
    def _has_error_handling(self, node: ast.FunctionDef) -> bool:
        """Verifica si una función tiene manejo de errores."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Try, ast.ExceptHandler)):
                return True
        return False
    
    def _calculate_scores(self, metrics: Dict[str, Any]) -> List[QualityScore]:
        """Calcula puntuaciones de calidad."""
        scores = []
        
        # Complexity Score
        complexity = metrics["complexity"]
        avg_complexity = complexity["average"]
        if avg_complexity <= 5:
            complexity_score = 100
        elif avg_complexity <= 10:
            complexity_score = 80
        elif avg_complexity <= 15:
            complexity_score = 60
        else:
            complexity_score = 40
        
        scores.append(QualityScore(
            metric=QualityMetric.COMPLEXITY,
            score=complexity_score,
            grade=self._score_to_grade(complexity_score),
            details=complexity,
            recommendations=self._get_complexity_recommendations(complexity)
        ))
        
        # Maintainability Score
        maintainability = metrics["maintainability_index"]
        scores.append(QualityScore(
            metric=QualityMetric.MAINTAINABILITY,
            score=maintainability,
            grade=self._score_to_grade(maintainability),
            details={"index": maintainability},
            recommendations=self._get_maintainability_recommendations(maintainability)
        ))
        
        # Documentation Score
        doc_score = metrics["documentation_score"]
        scores.append(QualityScore(
            metric=QualityMetric.DOCUMENTATION,
            score=doc_score,
            grade=self._score_to_grade(doc_score),
            details={"score": doc_score},
            recommendations=self._get_documentation_recommendations(doc_score)
        ))
        
        # Security Score
        security_issues = metrics["security_issues"]
        security_score = max(0, 100 - len(security_issues) * 10)
        scores.append(QualityScore(
            metric=QualityMetric.SECURITY,
            score=security_score,
            grade=self._score_to_grade(security_score),
            details={"issues": security_issues},
            recommendations=["Fix security issues"] if security_issues else []
        ))
        
        # Performance Score
        perf_issues = metrics["performance_issues"]
        perf_score = max(0, 100 - len(perf_issues) * 5)
        scores.append(QualityScore(
            metric=QualityMetric.PERFORMANCE,
            score=perf_score,
            grade=self._score_to_grade(perf_score),
            details={"issues": perf_issues},
            recommendations=["Fix performance issues"] if perf_issues else []
        ))
        
        return scores
    
    def _score_to_grade(self, score: float) -> str:
        """Convierte puntuación a letra."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _get_complexity_recommendations(self, complexity: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones de complejidad."""
        recommendations = []
        
        if complexity["average"] > 10:
            recommendations.append("Reduce average complexity by breaking down functions")
        
        if complexity["high_complexity_functions"]:
            recommendations.append(
                f"Refactor high complexity functions: {', '.join(complexity['high_complexity_functions'][:3])}"
            )
        
        return recommendations
    
    def _get_maintainability_recommendations(self, index: float) -> List[str]:
        """Genera recomendaciones de mantenibilidad."""
        recommendations = []
        
        if index < 70:
            recommendations.append("Improve code organization and structure")
            recommendations.append("Reduce code duplication")
            recommendations.append("Add more documentation")
        
        return recommendations
    
    def _get_documentation_recommendations(self, score: float) -> List[str]:
        """Genera recomendaciones de documentación."""
        recommendations = []
        
        if score < 80:
            recommendations.append("Add docstrings to functions and classes")
            recommendations.append("Document complex logic")
        
        return recommendations


# Factory function
_code_quality_analyzer = None

def get_code_quality_analyzer() -> CodeQualityAnalyzer:
    """Obtiene instancia global del analizador."""
    global _code_quality_analyzer
    if _code_quality_analyzer is None:
        _code_quality_analyzer = CodeQualityAnalyzer()
    return _code_quality_analyzer
