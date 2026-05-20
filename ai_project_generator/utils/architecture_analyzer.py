"""
Architecture Analyzer
=====================

Sistema de análisis de arquitectura de código.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ArchitecturePattern(Enum):
    """Patrones arquitectónicos."""
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    LAYERED = "layered"
    MICROSERVICES = "microservices"
    REPOSITORY = "repository"
    FACTORY = "factory"
    SINGLETON = "singleton"
    OBSERVER = "observer"
    STRATEGY = "strategy"


@dataclass
class ArchitectureIssue:
    """Problema arquitectónico."""
    pattern: ArchitecturePattern
    severity: str
    location: str
    description: str
    suggestion: str
    line_number: int


class ArchitectureAnalyzer:
    """Analizador de arquitectura."""
    
    def __init__(self):
        self.pattern_detectors = {
            ArchitecturePattern.MVC: self._detect_mvc,
            ArchitecturePattern.REPOSITORY: self._detect_repository,
            ArchitecturePattern.FACTORY: self._detect_factory,
            ArchitecturePattern.SINGLETON: self._detect_singleton,
            ArchitecturePattern.OBSERVER: self._detect_observer,
            ArchitecturePattern.STRATEGY: self._detect_strategy,
        }
    
    def analyze_architecture(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Analiza arquitectura del código."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "patterns": [],
                "issues": []
            }
        
        detected_patterns = []
        issues = []
        
        # Detectar patrones
        for pattern, detector in self.pattern_detectors.items():
            if detector(code, tree):
                detected_patterns.append(pattern.value)
        
        # Analizar problemas arquitectónicos
        issues.extend(self._analyze_coupling(code, tree))
        issues.extend(self._analyze_cohesion(code, tree))
        issues.extend(self._analyze_separation_of_concerns(code, tree))
        
        return {
            "file_path": file_path,
            "detected_patterns": detected_patterns,
            "architecture_score": self._calculate_architecture_score(detected_patterns, issues),
            "issues": [
                {
                    "pattern": i.pattern.value,
                    "severity": i.severity,
                    "location": i.location,
                    "line": i.line_number,
                    "description": i.description,
                    "suggestion": i.suggestion
                }
                for i in issues
            ],
            "recommendations": self._generate_recommendations(detected_patterns, issues)
        }
    
    def _detect_mvc(self, code: str, tree: ast.AST) -> bool:
        """Detecta patrón MVC."""
        has_model = any('Model' in node.name or 'model' in node.name.lower() 
                       for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        has_view = any('View' in node.name or 'view' in node.name.lower() 
                      for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        has_controller = any('Controller' in node.name or 'controller' in node.name.lower() 
                            for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        return has_model and has_view and has_controller
    
    def _detect_repository(self, code: str, tree: ast.AST) -> bool:
        """Detecta patrón Repository."""
        return any('Repository' in node.name or 'repository' in node.name.lower() 
                  for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    
    def _detect_factory(self, code: str, tree: ast.AST) -> bool:
        """Detecta patrón Factory."""
        return any('Factory' in node.name or 'factory' in node.name.lower() 
                  for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    
    def _detect_singleton(self, code: str, tree: ast.AST) -> bool:
        """Detecta patrón Singleton."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Buscar métodos get_instance o _instance
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        if 'get_instance' in child.name or '_instance' in code:
                            return True
        return False
    
    def _detect_observer(self, code: str, tree: ast.AST) -> bool:
        """Detecta patrón Observer."""
        return 'subscribe' in code.lower() or 'notify' in code.lower() or 'observer' in code.lower()
    
    def _detect_strategy(self, code: str, tree: ast.AST) -> bool:
        """Detecta patrón Strategy."""
        return 'Strategy' in code or 'strategy' in code.lower()
    
    def _analyze_coupling(self, code: str, tree: ast.AST) -> List[ArchitectureIssue]:
        """Analiza acoplamiento."""
        issues = []
        
        # Detectar imports excesivos
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if len(imports) > 20:
            issues.append(ArchitectureIssue(
                pattern=ArchitecturePattern.LAYERED,
                severity="medium",
                location="File level",
                description="High number of imports suggests tight coupling",
                suggestion="Consider using dependency injection or reducing dependencies",
                line_number=1
            ))
        
        return issues
    
    def _analyze_cohesion(self, code: str, tree: ast.AST) -> List[ArchitectureIssue]:
        """Analiza cohesión."""
        issues = []
        
        # Detectar clases con demasiadas responsabilidades
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
                if len(methods) > 15:
                    issues.append(ArchitectureIssue(
                        pattern=ArchitecturePattern.LAYERED,
                        severity="high",
                        location=f"Class {node.name}",
                        description=f"Class {node.name} has too many methods (low cohesion)",
                        suggestion="Split into multiple classes with single responsibilities",
                        line_number=node.lineno
                    ))
        
        return issues
    
    def _analyze_separation_of_concerns(self, code: str, tree: ast.AST) -> List[ArchitectureIssue]:
        """Analiza separación de responsabilidades."""
        issues = []
        
        # Detectar mezcla de lógica de negocio y presentación
        has_db_ops = any('db.' in code or 'database' in code.lower() or 'sql' in code.lower())
        has_ui_ops = any('render' in code.lower() or 'template' in code.lower() or 'html' in code.lower())
        
        if has_db_ops and has_ui_ops:
            issues.append(ArchitectureIssue(
                pattern=ArchitecturePattern.MVC,
                severity="high",
                location="File level",
                description="Mixing database operations with UI logic",
                suggestion="Separate into different layers (Model, View, Controller)",
                line_number=1
            ))
        
        return issues
    
    def _calculate_architecture_score(self, patterns: List[str], issues: List[ArchitectureIssue]) -> float:
        """Calcula puntuación arquitectónica."""
        score = 50.0  # Base score
        
        # Bonificación por patrones detectados
        score += len(patterns) * 10
        
        # Penalización por problemas
        for issue in issues:
            if issue.severity == "critical":
                score -= 20
            elif issue.severity == "high":
                score -= 10
            elif issue.severity == "medium":
                score -= 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(
        self, patterns: List[str], issues: List[ArchitectureIssue]
    ) -> List[str]:
        """Genera recomendaciones."""
        recommendations = []
        
        if not patterns:
            recommendations.append("Consider implementing common design patterns (Repository, Factory, etc.)")
        
        if any(i.severity == "high" for i in issues):
            recommendations.append("Address high-severity architecture issues")
        
        if ArchitecturePattern.MVC.value not in patterns:
            recommendations.append("Consider implementing MVC pattern for better separation of concerns")
        
        return recommendations


# Factory function
_architecture_analyzer = None

def get_architecture_analyzer() -> ArchitectureAnalyzer:
    """Obtiene instancia global del analizador."""
    global _architecture_analyzer
    if _architecture_analyzer is None:
        _architecture_analyzer = ArchitectureAnalyzer()
    return _architecture_analyzer

