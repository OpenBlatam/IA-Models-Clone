"""
Performance Analyzer
===================

Sistema de análisis de rendimiento de código.
"""

import ast
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PerformanceIssue(Enum):
    """Tipos de problemas de rendimiento."""
    SLOW_LOOP = "slow_loop"
    N_PLUS_ONE = "n_plus_one"
    UNNECESSARY_COMPUTATION = "unnecessary_computation"
    MEMORY_LEAK = "memory_leak"
    INEFFICIENT_ALGORITHM = "inefficient_algorithm"
    LARGE_DATA_STRUCTURE = "large_data_structure"


@dataclass
class PerformanceIssueReport:
    """Reporte de problema de rendimiento."""
    issue_type: PerformanceIssue
    severity: str  # low, medium, high, critical
    location: str
    line_number: int
    description: str
    suggestion: str
    estimated_impact: str


class PerformanceAnalyzer:
    """Analizador de rendimiento."""
    
    def __init__(self):
        self.issue_patterns = {
            PerformanceIssue.SLOW_LOOP: [
                (r'for\s+.*?:\s*\n.*?SELECT', "Database query in loop"),
                (r'for\s+.*?:\s*\n.*?requests\.', "HTTP request in loop"),
                (r'for\s+.*?:\s*\n.*?open\(', "File open in loop"),
            ],
            PerformanceIssue.N_PLUS_ONE: [
                (r'for\s+.*?:\s*\n.*?\.get\(', "N+1 query pattern"),
            ],
            PerformanceIssue.UNNECESSARY_COMPUTATION: [
                (r'(\w+)\s*=\s*([^=]+)\s*\n.*?\1\s*=\s*\2', "Repeated computation"),
            ],
            PerformanceIssue.INEFFICIENT_ALGORITHM: [
                (r'\.sort\(\)\s*\n.*?\.sort\(\)', "Multiple sorts"),
                (r'for\s+.*?in\s+range\(len\(', "Inefficient iteration"),
            ],
        }
    
    def analyze_code(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Analiza rendimiento del código."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "issues": []
            }
        
        issues = []
        
        # Detectar problemas conocidos
        for issue_type, patterns in self.issue_patterns.items():
            for pattern, description in patterns:
                issues.extend(self._detect_pattern(code, pattern, issue_type, description))
        
        # Análisis de complejidad algorítmica
        issues.extend(self._analyze_complexity(tree, code))
        
        # Análisis de uso de memoria
        issues.extend(self._analyze_memory_usage(code))
        
        # Ordenar por severidad
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        issues.sort(key=lambda x: severity_order.get(x.severity, 0), reverse=True)
        
        return {
            "file_path": file_path,
            "total_issues": len(issues),
            "by_severity": self._group_by_severity(issues),
            "by_type": self._group_by_type(issues),
            "issues": [
                {
                    "type": i.issue_type.value,
                    "severity": i.severity,
                    "location": i.location,
                    "line": i.line_number,
                    "description": i.description,
                    "suggestion": i.suggestion,
                    "impact": i.estimated_impact
                }
                for i in issues
            ]
        }
    
    def _detect_pattern(
        self, code: str, pattern: str, issue_type: PerformanceIssue, description: str
    ) -> List[PerformanceIssueReport]:
        """Detecta un patrón de problema."""
        issues = []
        
        for match in re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE | re.DOTALL):
            line_num = code[:match.start()].count('\n') + 1
            context = code[max(0, match.start()-50):match.end()+50]
            
            severity = self._determine_severity(issue_type, context)
            
            issues.append(PerformanceIssueReport(
                issue_type=issue_type,
                severity=severity,
                location=f"Line {line_num}",
                line_number=line_num,
                description=description,
                suggestion=self._get_suggestion(issue_type),
                estimated_impact=self._get_impact(issue_type, severity)
            ))
        
        return issues
    
    def _analyze_complexity(self, tree: ast.AST, code: str) -> List[PerformanceIssueReport]:
        """Analiza complejidad algorítmica."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Detectar loops anidados
                nested_loops = sum(1 for child in ast.walk(node) if isinstance(child, ast.For))
                if nested_loops > 2:
                    line_num = node.lineno
                    issues.append(PerformanceIssueReport(
                        issue_type=PerformanceIssue.INEFFICIENT_ALGORITHM,
                        severity="high",
                        location=f"Line {line_num}",
                        line_number=line_num,
                        description=f"Nested loops detected (O(n^{nested_loops+1}) complexity)",
                        suggestion="Consider using more efficient algorithms or data structures",
                        estimated_impact="High - can cause significant slowdown with large datasets"
                    ))
        
        return issues
    
    def _analyze_memory_usage(self, code: str) -> List[PerformanceIssueReport]:
        """Analiza uso de memoria."""
        issues = []
        
        # Detectar listas grandes en memoria
        pattern = r'(\w+)\s*=\s*\[.*?\]\s*#.*?large|(\w+)\s*=\s*list\(range\(\d{5,}\)\)'
        
        for match in re.finditer(pattern, code, re.IGNORECASE):
            line_num = code[:match.start()].count('\n') + 1
            var_name = match.group(1) or match.group(2)
            
            issues.append(PerformanceIssueReport(
                issue_type=PerformanceIssue.LARGE_DATA_STRUCTURE,
                severity="medium",
                location=f"Line {line_num}",
                line_number=line_num,
                description=f"Large data structure '{var_name}' may consume significant memory",
                suggestion="Consider using generators or streaming for large datasets",
                estimated_impact="Medium - may cause memory issues with large datasets"
            ))
        
        return issues
    
    def _determine_severity(self, issue_type: PerformanceIssue, context: str) -> str:
        """Determina severidad del problema."""
        if issue_type in [PerformanceIssue.N_PLUS_ONE, PerformanceIssue.SLOW_LOOP]:
            # Verificar si está en un loop anidado
            if 'for' in context and context.count('for') > 1:
                return "critical"
            return "high"
        
        if issue_type == PerformanceIssue.MEMORY_LEAK:
            return "high"
        
        return "medium"
    
    def _get_suggestion(self, issue_type: PerformanceIssue) -> str:
        """Obtiene sugerencia para el tipo de problema."""
        suggestions = {
            PerformanceIssue.SLOW_LOOP: "Move operation outside loop or use batch processing",
            PerformanceIssue.N_PLUS_ONE: "Use eager loading or batch queries",
            PerformanceIssue.UNNECESSARY_COMPUTATION: "Cache result or compute once",
            PerformanceIssue.INEFFICIENT_ALGORITHM: "Use more efficient algorithm or data structure",
            PerformanceIssue.LARGE_DATA_STRUCTURE: "Use generators or streaming",
            PerformanceIssue.MEMORY_LEAK: "Ensure proper cleanup and resource management",
        }
        return suggestions.get(issue_type, "Review and optimize")
    
    def _get_impact(self, issue_type: PerformanceIssue, severity: str) -> str:
        """Obtiene impacto estimado."""
        if severity == "critical":
            return "Critical - can cause severe performance degradation"
        elif severity == "high":
            return "High - significant performance impact expected"
        elif severity == "medium":
            return "Medium - moderate performance impact"
        else:
            return "Low - minor performance impact"
    
    def _group_by_severity(self, issues: List[PerformanceIssueReport]) -> Dict[str, int]:
        """Agrupa por severidad."""
        grouped = {}
        for issue in issues:
            grouped[issue.severity] = grouped.get(issue.severity, 0) + 1
        return grouped
    
    def _group_by_type(self, issues: List[PerformanceIssueReport]) -> Dict[str, int]:
        """Agrupa por tipo."""
        grouped = {}
        for issue in issues:
            grouped[issue.issue_type.value] = grouped.get(issue.issue_type.value, 0) + 1
        return grouped


# Factory function
_performance_analyzer = None

def get_performance_analyzer_code() -> PerformanceAnalyzer:
    """Obtiene instancia global del analizador."""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer()
    return _performance_analyzer
