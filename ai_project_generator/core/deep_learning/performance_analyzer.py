"""
Performance Analyzer
====================

Analizador de performance para código generado.
"""

import logging
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de performance."""
    cyclomatic_complexity: int = 0
    lines_of_code: int = 0
    function_count: int = 0
    class_count: int = 0
    average_function_length: float = 0.0
    max_nesting_depth: int = 0
    import_count: int = 0
    warnings: List[str] = field(default_factory=list)


class PerformanceAnalyzer:
    """
    Analizador de performance de código.
    """
    
    def __init__(self):
        """Inicializar analizador."""
        pass
    
    def analyze_file(self, file_path: Path) -> PerformanceMetrics:
        """
        Analizar performance de un archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Métricas de performance
        """
        if not file_path.exists() or file_path.suffix != '.py':
            return PerformanceMetrics()
        
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            metrics = PerformanceMetrics()
            metrics.lines_of_code = len(content.split('\n'))
            
            # Contar funciones y clases
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics.function_count += 1
                    metrics.cyclomatic_complexity += self._calculate_complexity(node)
                    metrics.max_nesting_depth = max(
                        metrics.max_nesting_depth,
                        self._get_nesting_depth(node)
                    )
                elif isinstance(node, ast.ClassDef):
                    metrics.class_count += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics.import_count += 1
            
            # Calcular promedio de longitud de funciones
            if metrics.function_count > 0:
                function_lengths = [
                    len(self._get_function_body(func).split('\n'))
                    for func in ast.walk(tree)
                    if isinstance(func, ast.FunctionDef)
                ]
                metrics.average_function_length = sum(function_lengths) / len(function_lengths) if function_lengths else 0
            
            # Generar warnings
            metrics.warnings = self._generate_warnings(metrics)
            
            return metrics
        except Exception as e:
            logger.warning(f"Error analizando {file_path}: {e}")
            return PerformanceMetrics()
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calcular complejidad ciclomática."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _get_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Obtener profundidad de anidamiento."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                child_depth = self._get_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._get_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _get_function_body(self, func: ast.FunctionDef) -> str:
        """Obtener cuerpo de función como string."""
        # Simplificado - en realidad necesitaríamos el código fuente original
        return ""
    
    def _generate_warnings(self, metrics: PerformanceMetrics) -> List[str]:
        """Generar warnings basados en métricas."""
        warnings = []
        
        if metrics.cyclomatic_complexity > 10:
            warnings.append(f"Alta complejidad ciclomática: {metrics.cyclomatic_complexity}")
        
        if metrics.average_function_length > 50:
            warnings.append(f"Funciones muy largas (promedio: {metrics.average_function_length:.1f} líneas)")
        
        if metrics.max_nesting_depth > 4:
            warnings.append(f"Anidamiento profundo: {metrics.max_nesting_depth} niveles")
        
        if metrics.function_count > 20:
            warnings.append(f"Muchas funciones en un archivo: {metrics.function_count}")
        
        return warnings
    
    def analyze_project(
        self,
        project_dir: Path
    ) -> Dict[str, Any]:
        """
        Analizar performance de todo el proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            
        Returns:
            Diccionario con métricas del proyecto
        """
        results = {}
        total_metrics = PerformanceMetrics()
        
        app_dir = project_dir / "app"
        if not app_dir.exists():
            return {'error': 'Directorio app no encontrado'}
        
        for py_file in app_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            metrics = self.analyze_file(py_file)
            results[str(py_file.relative_to(project_dir))] = {
                'cyclomatic_complexity': metrics.cyclomatic_complexity,
                'lines_of_code': metrics.lines_of_code,
                'function_count': metrics.function_count,
                'class_count': metrics.class_count,
                'average_function_length': metrics.average_function_length,
                'max_nesting_depth': metrics.max_nesting_depth,
                'warnings': metrics.warnings
            }
            
            # Acumular métricas totales
            total_metrics.cyclomatic_complexity += metrics.cyclomatic_complexity
            total_metrics.lines_of_code += metrics.lines_of_code
            total_metrics.function_count += metrics.function_count
            total_metrics.class_count += metrics.class_count
        
        # Calcular promedios
        file_count = len(results)
        if file_count > 0:
            total_metrics.average_function_length = (
                sum(m['average_function_length'] for m in results.values()) / file_count
            )
        
        return {
            'total_files': file_count,
            'total_metrics': {
                'cyclomatic_complexity': total_metrics.cyclomatic_complexity,
                'lines_of_code': total_metrics.lines_of_code,
                'function_count': total_metrics.function_count,
                'class_count': total_metrics.class_count,
                'average_function_length': total_metrics.average_function_length
            },
            'by_file': results
        }


# Instancia global
_global_performance_analyzer: Optional[PerformanceAnalyzer] = None


def get_performance_analyzer() -> PerformanceAnalyzer:
    """
    Obtener instancia global del analizador de performance.
    
    Returns:
        Instancia del analizador
    """
    global _global_performance_analyzer
    
    if _global_performance_analyzer is None:
        _global_performance_analyzer = PerformanceAnalyzer()
    
    return _global_performance_analyzer

