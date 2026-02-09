"""
Code Quality System
===================

Sistema de análisis de calidad de código.
"""

import ast
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodeMetric:
    """Métrica de código."""
    name: str
    value: float
    threshold: float
    unit: str = ""
    description: str = ""


@dataclass
class CodeQualityReport:
    """Reporte de calidad de código."""
    file_path: str
    lines_of_code: int
    complexity: float
    functions_count: int
    classes_count: int
    imports_count: int
    docstring_coverage: float
    metrics: List[CodeMetric] = field(default_factory=list)


class CodeQualityAnalyzer:
    """
    Analizador de calidad de código.
    
    Analiza código Python y genera métricas de calidad.
    """
    
    def __init__(self):
        """Inicializar analizador."""
        self.reports: List[CodeQualityReport] = []
    
    def analyze_file(self, file_path: str) -> CodeQualityReport:
        """
        Analizar archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Reporte de calidad
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Contar líneas
        lines_of_code = len([line for line in source.split('\n') if line.strip()])
        
        # Analizar AST
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        # Calcular complejidad (simplificada)
        complexity = self._calculate_complexity(tree)
        
        # Calcular cobertura de docstrings
        docstring_coverage = self._calculate_docstring_coverage(functions, classes)
        
        report = CodeQualityReport(
            file_path=file_path,
            lines_of_code=lines_of_code,
            complexity=complexity,
            functions_count=len(functions),
            classes_count=len(classes),
            imports_count=len(imports),
            docstring_coverage=docstring_coverage
        )
        
        # Agregar métricas
        report.metrics = [
            CodeMetric(
                name="lines_of_code",
                value=lines_of_code,
                threshold=1000,
                unit="lines",
                description="Total lines of code"
            ),
            CodeMetric(
                name="complexity",
                value=complexity,
                threshold=10.0,
                unit="",
                description="Cyclomatic complexity"
            ),
            CodeMetric(
                name="docstring_coverage",
                value=docstring_coverage,
                threshold=0.8,
                unit="ratio",
                description="Docstring coverage"
            )
        ]
        
        self.reports.append(report)
        return report
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calcular complejidad ciclomática."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_docstring_coverage(
        self,
        functions: List[ast.FunctionDef],
        classes: List[ast.ClassDef]
    ) -> float:
        """Calcular cobertura de docstrings."""
        total = len(functions) + len(classes)
        if total == 0:
            return 1.0
        
        with_docstrings = 0
        
        for func in functions:
            if ast.get_docstring(func):
                with_docstrings += 1
        
        for cls in classes:
            if ast.get_docstring(cls):
                with_docstrings += 1
        
        return with_docstrings / total if total > 0 else 0.0
    
    def analyze_directory(self, directory: str, pattern: str = "*.py") -> List[CodeQualityReport]:
        """
        Analizar directorio.
        
        Args:
            directory: Directorio a analizar
            pattern: Patrón de archivos
            
        Returns:
            Lista de reportes
        """
        path = Path(directory)
        reports = []
        
        for file_path in path.rglob(pattern):
            try:
                report = self.analyze_file(str(file_path))
                reports.append(report)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        return reports
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis."""
        if not self.reports:
            return {
                "total_files": 0,
                "total_lines": 0,
                "average_complexity": 0.0,
                "average_docstring_coverage": 0.0
            }
        
        total_files = len(self.reports)
        total_lines = sum(r.lines_of_code for r in self.reports)
        avg_complexity = sum(r.complexity for r in self.reports) / total_files
        avg_docstring = sum(r.docstring_coverage for r in self.reports) / total_files
        
        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "total_functions": sum(r.functions_count for r in self.reports),
            "total_classes": sum(r.classes_count for r in self.reports),
            "average_complexity": avg_complexity,
            "average_docstring_coverage": avg_docstring
        }


# Instancia global
_code_quality_analyzer: Optional[CodeQualityAnalyzer] = None


def get_code_quality_analyzer() -> CodeQualityAnalyzer:
    """Obtener instancia global del analizador de calidad."""
    global _code_quality_analyzer
    if _code_quality_analyzer is None:
        _code_quality_analyzer = CodeQualityAnalyzer()
    return _code_quality_analyzer






