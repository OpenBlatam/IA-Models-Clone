"""
Code Optimizer
==============

Sistema de optimización automática de código generado.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Tipos de optimización."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    READABILITY = "readability"
    SECURITY = "security"
    BEST_PRACTICES = "best_practices"


@dataclass
class OptimizationSuggestion:
    """Sugerencia de optimización."""
    type: OptimizationType
    file_path: str
    line_number: int
    suggestion: str
    priority: int  # 1-10, mayor = más importante
    code_before: str
    code_after: str
    impact: str


class CodeOptimizer:
    """Optimizador de código automático."""
    
    def __init__(self):
        self.optimization_rules: Dict[OptimizationType, List[callable]] = {}
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Registra reglas de optimización por defecto."""
        # Performance optimizations
        self.optimization_rules[OptimizationType.PERFORMANCE] = [
            self._optimize_list_comprehensions,
            self._optimize_string_concatenation,
            self._optimize_database_queries,
        ]
        
        # Memory optimizations
        self.optimization_rules[OptimizationType.MEMORY] = [
            self._optimize_generators,
            self._optimize_imports,
        ]
        
        # Security optimizations
        self.optimization_rules[OptimizationType.SECURITY] = [
            self._check_sql_injection,
            self._check_xss_vulnerabilities,
            self._check_hardcoded_secrets,
        ]
        
        # Best practices
        self.optimization_rules[OptimizationType.BEST_PRACTICES] = [
            self._check_error_handling,
            self._check_type_hints,
            self._check_docstrings,
        ]
    
    def analyze_code(self, code: str, file_path: str = "unknown") -> List[OptimizationSuggestion]:
        """Analiza código y genera sugerencias."""
        suggestions = []
        lines = code.split('\n')
        
        for opt_type, rules in self.optimization_rules.items():
            for rule in rules:
                try:
                    rule_suggestions = rule(code, lines, file_path)
                    suggestions.extend(rule_suggestions)
                except Exception as e:
                    logger.warning(f"Error in optimization rule {rule.__name__}: {e}")
        
        # Ordenar por prioridad
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        return suggestions
    
    def _optimize_list_comprehensions(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Optimiza list comprehensions."""
        suggestions = []
        
        # Buscar patrones como: result = []
        #                     for item in items:
        #                         result.append(...)
        pattern = r'(\w+)\s*=\s*\[\]\s*\n\s*for\s+(\w+)\s+in\s+(\w+):\s*\n\s*(\w+)\.append\((.+)\)'
        
        for match in re.finditer(pattern, code, re.MULTILINE):
            var_name = match.group(1)
            item_var = match.group(2)
            iterable = match.group(3)
            append_expr = match.group(5)
            
            line_num = code[:match.start()].count('\n') + 1
            optimized = f"{var_name} = [{append_expr} for {item_var} in {iterable}]"
            
            suggestions.append(OptimizationSuggestion(
                type=OptimizationType.PERFORMANCE,
                file_path=file_path,
                line_number=line_num,
                suggestion=f"Use list comprehension instead of loop",
                priority=7,
                code_before=match.group(0),
                code_after=optimized,
                impact="Faster execution, more Pythonic"
            ))
        
        return suggestions
    
    def _optimize_string_concatenation(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Optimiza concatenación de strings."""
        suggestions = []
        
        # Buscar múltiples concatenaciones con +
        pattern = r'(\w+)\s*=\s*(\w+)\s*\+\s*(\w+)\s*\+\s*(\w+)'
        
        for match in re.finditer(pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            optimized = f"{match.group(1)} = ''.join([{match.group(2)}, {match.group(3)}, {match.group(4)}])"
            
            suggestions.append(OptimizationSuggestion(
                type=OptimizationType.PERFORMANCE,
                file_path=file_path,
                line_number=line_num,
                suggestion="Use join() for multiple string concatenations",
                priority=6,
                code_before=match.group(0),
                code_after=optimized,
                impact="Better performance for multiple concatenations"
            ))
        
        return suggestions
    
    def _optimize_generators(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Optimiza uso de generadores."""
        suggestions = []
        
        # Buscar list() alrededor de generadores
        pattern = r'list\(\((.+?)\)\s+for\s+(.+?)\s+in\s+(.+?)\)'
        
        for match in re.finditer(pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            optimized = f"({match.group(1)} for {match.group(2)} in {match.group(3)})"
            
            suggestions.append(OptimizationSuggestion(
                type=OptimizationType.MEMORY,
                file_path=file_path,
                line_number=line_num,
                suggestion="Use generator expression instead of list",
                priority=5,
                code_before=match.group(0),
                code_after=optimized,
                impact="Lower memory usage"
            ))
        
        return suggestions
    
    def _check_sql_injection(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Detecta posibles vulnerabilidades SQL injection."""
        suggestions = []
        
        # Buscar f-strings en queries SQL
        pattern = r'f["\']\s*SELECT.*\{.*\}'
        
        for match in re.finditer(pattern, code, re.IGNORECASE | re.DOTALL):
            line_num = code[:match.start()].count('\n') + 1
            
            suggestions.append(OptimizationSuggestion(
                type=OptimizationType.SECURITY,
                file_path=file_path,
                line_number=line_num,
                suggestion="Potential SQL injection vulnerability - use parameterized queries",
                priority=10,
                code_before=match.group(0),
                code_after="# Use parameterized queries: cursor.execute('SELECT * FROM table WHERE id = ?', (id,))",
                impact="Critical security improvement"
            ))
        
        return suggestions
    
    def _check_hardcoded_secrets(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Detecta secretos hardcodeados."""
        suggestions = []
        
        patterns = [
            (r'password\s*=\s*["\']([^"\']+)["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\']([^"\']+)["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\']([^"\']+)["\']', "Hardcoded secret"),
        ]
        
        for pattern, message in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_num = code[:match.start()].count('\n') + 1
                
                suggestions.append(OptimizationSuggestion(
                    type=OptimizationType.SECURITY,
                    file_path=file_path,
                    line_number=line_num,
                    suggestion=f"{message} - use environment variables",
                    priority=9,
                    code_before=match.group(0),
                    code_after="# Use: password = os.getenv('PASSWORD')",
                    impact="Critical security improvement"
                ))
        
        return suggestions
    
    def _check_error_handling(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Verifica manejo de errores."""
        suggestions = []
        
        # Buscar funciones sin try/except
        # Simplificado: buscar funciones que hacen operaciones riesgosas
        pattern = r'def\s+(\w+).*?:\s*\n(?!.*try)'
        
        for match in re.finditer(pattern, code, re.MULTILINE | re.DOTALL):
            if 'open(' in match.group(0) or 'requests.' in match.group(0):
                line_num = code[:match.start()].count('\n') + 1
                
                suggestions.append(OptimizationSuggestion(
                    type=OptimizationType.BEST_PRACTICES,
                    file_path=file_path,
                    line_number=line_num,
                    suggestion="Add error handling for risky operations",
                    priority=6,
                    code_before=match.group(0)[:100],
                    code_after="# Wrap in try/except block",
                    impact="Better error handling and user experience"
                ))
        
        return suggestions
    
    def _optimize_database_queries(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Optimiza queries de base de datos."""
        suggestions = []
        
        # Buscar queries en loops
        pattern = r'for\s+.*?:\s*\n.*?SELECT.*?\n'
        
        for match in re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            line_num = code[:match.start()].count('\n') + 1
            
            suggestions.append(OptimizationSuggestion(
                type=OptimizationType.PERFORMANCE,
                file_path=file_path,
                line_number=line_num,
                suggestion="Database query in loop - consider batch query or JOIN",
                priority=8,
                code_before=match.group(0)[:200],
                code_after="# Use batch query or JOIN instead",
                impact="Significant performance improvement"
            ))
        
        return suggestions
    
    def _optimize_imports(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Optimiza imports."""
        suggestions = []
        
        # Buscar import *
        if 'from' in code and 'import *' in code:
            line_num = code.find('import *')
            if line_num > 0:
                line_num = code[:line_num].count('\n') + 1
                
                suggestions.append(OptimizationSuggestion(
                    type=OptimizationType.MEMORY,
                    file_path=file_path,
                    line_number=line_num,
                    suggestion="Avoid 'import *' - import specific functions",
                    priority=4,
                    code_before="from module import *",
                    code_after="from module import specific_function",
                    impact="Lower memory usage, clearer code"
                ))
        
        return suggestions
    
    def _check_type_hints(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Verifica type hints."""
        suggestions = []
        
        # Buscar funciones sin type hints
        pattern = r'def\s+(\w+)\(([^)]+)\)\s*:'
        
        for match in re.finditer(pattern, code):
            params = match.group(2)
            if not any(':' in p for p in params.split(',')):
                line_num = code[:match.start()].count('\n') + 1
                
                suggestions.append(OptimizationSuggestion(
                    type=OptimizationType.BEST_PRACTICES,
                    file_path=file_path,
                    line_number=line_num,
                    suggestion="Add type hints to function parameters",
                    priority=3,
                    code_before=match.group(0),
                    code_after=f"def {match.group(1)}(param: str) -> None:",
                    impact="Better code documentation and IDE support"
                ))
        
        return suggestions
    
    def _check_docstrings(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Verifica docstrings."""
        suggestions = []
        
        # Buscar funciones sin docstrings
        pattern = r'def\s+(\w+).*?:\s*\n(?!\s*""")'
        
        for match in re.finditer(pattern, code, re.MULTILINE):
            func_name = match.group(1)
            if not func_name.startswith('_') or func_name == '__init__':  # Solo funciones públicas
                line_num = code[:match.start()].count('\n') + 1
                
                suggestions.append(OptimizationSuggestion(
                    type=OptimizationType.BEST_PRACTICES,
                    file_path=file_path,
                    line_number=line_num,
                    suggestion=f"Add docstring to function {func_name}",
                    priority=2,
                    code_before=match.group(0)[:100],
                    code_after=f'def {func_name}(...):\n    """Description of function."""',
                    impact="Better code documentation"
                ))
        
        return suggestions
    
    def _check_xss_vulnerabilities(self, code: str, lines: List[str], file_path: str) -> List[OptimizationSuggestion]:
        """Detecta posibles vulnerabilidades XSS."""
        suggestions = []
        
        # Buscar renderizado directo de variables de usuario
        if 'render_template' in code or 'render' in code:
            pattern = r'\{(\w+)\}'
            
            for match in re.finditer(pattern, code):
                var_name = match.group(1)
                if 'user' in var_name.lower() or 'input' in var_name.lower():
                    line_num = code[:match.start()].count('\n') + 1
                    
                    suggestions.append(OptimizationSuggestion(
                        type=OptimizationType.SECURITY,
                        file_path=file_path,
                        line_number=line_num,
                        suggestion="Potential XSS vulnerability - escape user input",
                        priority=8,
                        code_before=match.group(0),
                        code_after=f"{{{{ {var_name}|escape }}}}",
                        impact="Security improvement"
                    ))
        
        return suggestions
    
    def optimize_file(self, file_path: str, code: str) -> Dict[str, Any]:
        """Optimiza un archivo completo."""
        suggestions = self.analyze_code(code, file_path)
        
        return {
            "file_path": file_path,
            "total_suggestions": len(suggestions),
            "by_type": self._group_by_type(suggestions),
            "by_priority": self._group_by_priority(suggestions),
            "suggestions": [
                {
                    "type": s.type.value,
                    "line": s.line_number,
                    "suggestion": s.suggestion,
                    "priority": s.priority,
                    "impact": s.impact,
                    "code_before": s.code_before[:200],
                    "code_after": s.code_after[:200]
                }
                for s in suggestions
            ]
        }
    
    def _group_by_type(self, suggestions: List[OptimizationSuggestion]) -> Dict[str, int]:
        """Agrupa sugerencias por tipo."""
        grouped = {}
        for s in suggestions:
            grouped[s.type.value] = grouped.get(s.type.value, 0) + 1
        return grouped
    
    def _group_by_priority(self, suggestions: List[OptimizationSuggestion]) -> Dict[str, int]:
        """Agrupa sugerencias por prioridad."""
        grouped = {
            "high": 0,  # 8-10
            "medium": 0,  # 5-7
            "low": 0  # 1-4
        }
        
        for s in suggestions:
            if s.priority >= 8:
                grouped["high"] += 1
            elif s.priority >= 5:
                grouped["medium"] += 1
            else:
                grouped["low"] += 1
        
        return grouped


# Factory function
_code_optimizer = None

def get_code_optimizer() -> CodeOptimizer:
    """Obtiene instancia global del optimizador."""
    global _code_optimizer
    if _code_optimizer is None:
        _code_optimizer = CodeOptimizer()
    return _code_optimizer


