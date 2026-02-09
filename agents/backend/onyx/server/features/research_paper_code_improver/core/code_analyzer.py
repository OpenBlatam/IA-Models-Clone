"""
Code Analyzer - Análisis de código usando AST
=============================================
"""

import logging
import ast
from typing import Dict, Any, List, Optional, Set
import re

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Analiza código usando AST para entender estructura y detectar mejoras.
    """
    
    def __init__(self):
        """Inicializar analizador de código"""
        self.supported_languages = ["python", "javascript", "typescript"]
    
    def analyze_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Analiza código y extrae métricas y estructura.
        
        Args:
            code: Código a analizar
            language: Lenguaje de programación
            
        Returns:
            Análisis del código
        """
        try:
            if language == "python":
                return self._analyze_python(code)
            elif language in ["javascript", "typescript"]:
                return self._analyze_javascript(code)
            else:
                return self._analyze_generic(code, language)
        except Exception as e:
            logger.error(f"Error analizando código: {e}")
            return self._analyze_generic(code, language)
    
    def _analyze_python(self, code: str) -> Dict[str, Any]:
        """Analiza código Python usando AST"""
        try:
            tree = ast.parse(code)
            
            analysis = {
                "language": "python",
                "metrics": {
                    "lines_of_code": len(code.split("\n")),
                    "functions": 0,
                    "classes": 0,
                    "imports": 0,
                    "complexity": 0,
                },
                "structure": {
                    "functions": [],
                    "classes": [],
                    "imports": [],
                },
                "issues": [],
                "suggestions": []
            }
            
            # Analizar nodos
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["metrics"]["functions"] += 1
                    func_info = {
                        "name": node.name,
                        "args": len(node.args.args),
                        "lineno": node.lineno,
                        "complexity": self._calculate_complexity(node)
                    }
                    analysis["structure"]["functions"].append(func_info)
                    
                    # Detectar problemas
                    if len(node.args.args) > 5:
                        analysis["issues"].append({
                            "type": "too_many_parameters",
                            "function": node.name,
                            "line": node.lineno,
                            "severity": "medium"
                        })
                    
                    if func_info["complexity"] > 10:
                        analysis["issues"].append({
                            "type": "high_complexity",
                            "function": node.name,
                            "line": node.lineno,
                            "severity": "high"
                        })
                
                elif isinstance(node, ast.ClassDef):
                    analysis["metrics"]["classes"] += 1
                    class_info = {
                        "name": node.name,
                        "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                        "lineno": node.lineno
                    }
                    analysis["structure"]["classes"].append(class_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis["metrics"]["imports"] += 1
                    import_names = []
                    if isinstance(node, ast.Import):
                        import_names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        import_names = [f"{node.module}.{alias.name}" if node.module else alias.name 
                                      for alias in node.names]
                    analysis["structure"]["imports"].extend(import_names)
            
            # Calcular complejidad total
            analysis["metrics"]["complexity"] = sum(
                f.get("complexity", 0) for f in analysis["structure"]["functions"]
            )
            
            # Generar sugerencias
            analysis["suggestions"] = self._generate_suggestions(analysis)
            
            return analysis
            
        except SyntaxError as e:
            logger.warning(f"Error de sintaxis en código Python: {e}")
            return {
                "language": "python",
                "error": f"Syntax error: {str(e)}",
                "metrics": {},
                "structure": {}
            }
        except Exception as e:
            logger.error(f"Error analizando Python: {e}")
            return self._analyze_generic(code, "python")
    
    def _analyze_javascript(self, code: str) -> Dict[str, Any]:
        """Analiza código JavaScript/TypeScript (análisis básico)"""
        analysis = {
            "language": "javascript",
            "metrics": {
                "lines_of_code": len(code.split("\n")),
                "functions": len(re.findall(r'\bfunction\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>', code)),
                "classes": len(re.findall(r'\bclass\s+\w+', code)),
                "imports": len(re.findall(r'import\s+.*from', code)),
            },
            "structure": {},
            "issues": [],
            "suggestions": []
        }
        
        # Detectar problemas comunes
        if "var " in code:
            analysis["issues"].append({
                "type": "use_let_or_const",
                "description": "Consider using let or const instead of var",
                "severity": "low"
            })
        
        return analysis
    
    def _analyze_generic(self, code: str, language: str) -> Dict[str, Any]:
        """Análisis genérico para cualquier lenguaje"""
        lines = code.split("\n")
        
        return {
            "language": language,
            "metrics": {
                "lines_of_code": len(lines),
                "non_empty_lines": len([l for l in lines if l.strip()]),
                "comments": len([l for l in lines if l.strip().startswith("#") or "//" in l]),
            },
            "structure": {},
            "issues": [],
            "suggestions": []
        }
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calcula complejidad ciclomática de una función"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera sugerencias basadas en el análisis"""
        suggestions = []
        metrics = analysis.get("metrics", {})
        
        # Sugerencias basadas en métricas
        if metrics.get("complexity", 0) > 20:
            suggestions.append({
                "type": "reduce_complexity",
                "description": "Consider refactoring to reduce code complexity",
                "priority": "high"
            })
        
        if metrics.get("functions", 0) == 0:
            suggestions.append({
                "type": "add_functions",
                "description": "Consider breaking code into functions for better organization",
                "priority": "medium"
            })
        
        if len(analysis.get("structure", {}).get("imports", [])) > 20:
            suggestions.append({
                "type": "organize_imports",
                "description": "Consider organizing imports and removing unused ones",
                "priority": "low"
            })
        
        # Sugerencias basadas en issues
        high_issues = [i for i in analysis.get("issues", []) if i.get("severity") == "high"]
        if high_issues:
            suggestions.append({
                "type": "fix_high_priority_issues",
                "description": f"Address {len(high_issues)} high priority issues",
                "priority": "high",
                "issues_count": len(high_issues)
            })
        
        return suggestions
    
    def compare_code(self, original: str, improved: str, language: str = "python") -> Dict[str, Any]:
        """
        Compara código original y mejorado.
        
        Args:
            original: Código original
            improved: Código mejorado
            language: Lenguaje de programación
            
        Returns:
            Comparación detallada
        """
        original_analysis = self.analyze_code(original, language)
        improved_analysis = self.analyze_code(improved, language)
        
        original_metrics = original_analysis.get("metrics", {})
        improved_metrics = improved_analysis.get("metrics", {})
        
        comparison = {
            "original": original_analysis,
            "improved": improved_analysis,
            "improvements": {
                "complexity_change": improved_metrics.get("complexity", 0) - original_metrics.get("complexity", 0),
                "functions_change": improved_metrics.get("functions", 0) - original_metrics.get("functions", 0),
                "lines_change": improved_metrics.get("lines_of_code", 0) - original_metrics.get("lines_of_code", 0),
                "issues_fixed": len(original_analysis.get("issues", [])) - len(improved_analysis.get("issues", [])),
            },
            "summary": {
                "better": [],
                "worse": [],
                "unchanged": []
            }
        }
        
        # Determinar qué mejoró
        if comparison["improvements"]["complexity_change"] < 0:
            comparison["summary"]["better"].append("Reduced complexity")
        elif comparison["improvements"]["complexity_change"] > 0:
            comparison["summary"]["worse"].append("Increased complexity")
        
        if comparison["improvements"]["issues_fixed"] > 0:
            comparison["summary"]["better"].append(f"Fixed {comparison['improvements']['issues_fixed']} issues")
        
        return comparison




