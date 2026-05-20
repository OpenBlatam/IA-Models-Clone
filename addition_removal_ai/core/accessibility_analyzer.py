"""
Accessibility Analyzer - Sistema de análisis de accesibilidad
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AccessibilityIssue:
    """Problema de accesibilidad"""
    type: str
    severity: str
    description: str
    suggestion: str
    location: Optional[int] = None


class AccessibilityAnalyzer:
    """Analizador de accesibilidad"""

    def __init__(self):
        """Inicializar analizador"""
        pass

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar accesibilidad del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de accesibilidad
        """
        issues = []
        
        # Verificar headers
        header_issues = self._check_headers(content)
        issues.extend(header_issues)
        
        # Verificar imágenes sin alt text
        image_issues = self._check_images(content)
        issues.extend(image_issues)
        
        # Verificar contraste de texto (simulado)
        contrast_issues = self._check_contrast(content)
        issues.extend(contrast_issues)
        
        # Verificar estructura
        structure_issues = self._check_structure(content)
        issues.extend(structure_issues)
        
        # Verificar links
        link_issues = self._check_links(content)
        issues.extend(link_issues)
        
        # Calcular score
        total_issues = len(issues)
        critical_issues = len([i for i in issues if i.severity == "critical"])
        high_issues = len([i for i in issues if i.severity == "high"])
        
        # Score: 100 - (critical * 20 + high * 10 + medium * 5 + low * 1)
        score = 100
        for issue in issues:
            if issue.severity == "critical":
                score -= 20
            elif issue.severity == "high":
                score -= 10
            elif issue.severity == "medium":
                score -= 5
            else:
                score -= 1
        
        score = max(0, min(100, score))
        
        return {
            "accessibility_score": score,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "medium_issues": len([i for i in issues if i.severity == "medium"]),
            "low_issues": len([i for i in issues if i.severity == "low"]),
            "issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "description": issue.description,
                    "suggestion": issue.suggestion,
                    "location": issue.location
                }
                for issue in issues
            ],
            "is_accessible": score >= 70
        }

    def _check_headers(self, content: str) -> List[AccessibilityIssue]:
        """Verificar headers"""
        issues = []
        lines = content.split('\n')
        
        # Verificar si hay headers
        has_headers = any(line.startswith('#') for line in lines)
        if not has_headers and len(content) > 500:
            issues.append(AccessibilityIssue(
                type="missing_headers",
                severity="high",
                description="El documento no tiene headers para estructurar el contenido",
                suggestion="Agrega headers usando # para mejorar la navegación"
            ))
        
        # Verificar jerarquía de headers
        header_levels = []
        for i, line in enumerate(lines):
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                header_levels.append((level, i + 1))
        
        # Verificar saltos de nivel
        for i in range(len(header_levels) - 1):
            current_level = header_levels[i][0]
            next_level = header_levels[i + 1][0]
            if next_level > current_level + 1:
                issues.append(AccessibilityIssue(
                    type="header_hierarchy",
                    severity="medium",
                    description=f"Salto de nivel en headers (nivel {current_level} a {next_level})",
                    suggestion="Mantén una jerarquía consistente de headers (h1 -> h2 -> h3)",
                    location=header_levels[i + 1][1]
                ))
        
        return issues

    def _check_images(self, content: str) -> List[AccessibilityIssue]:
        """Verificar imágenes"""
        issues = []
        
        # Buscar imágenes Markdown
        image_pattern = re.compile(r'!\[([^\]]*)\]\(([^\)]+)\)')
        matches = image_pattern.finditer(content)
        
        for match in matches:
            alt_text = match.group(1)
            if not alt_text or alt_text.strip() == "":
                issues.append(AccessibilityIssue(
                    type="missing_alt_text",
                    severity="critical",
                    description="Imagen sin texto alternativo",
                    suggestion="Agrega texto alternativo descriptivo: ![descripción](url)",
                    location=match.start()
                ))
        
        return issues

    def _check_contrast(self, content: str) -> List[AccessibilityIssue]:
        """Verificar contraste (simulado)"""
        issues = []
        
        # En un sistema real, esto verificaría colores reales
        # Por ahora, solo verificamos si hay menciones de colores que podrían ser problemáticos
        low_contrast_colors = ['yellow', 'light', 'pale', 'amarillo', 'claro', 'pálido']
        content_lower = content.lower()
        
        for color in low_contrast_colors:
            if color in content_lower:
                issues.append(AccessibilityIssue(
                    type="potential_contrast_issue",
                    severity="medium",
                    description=f"Posible problema de contraste con color '{color}'",
                    suggestion="Asegúrate de que el contraste entre texto y fondo sea suficiente (ratio 4.5:1 mínimo)"
                ))
        
        return issues

    def _check_structure(self, content: str) -> List[AccessibilityIssue]:
        """Verificar estructura"""
        issues = []
        
        # Verificar si tiene título
        lines = content.split('\n')
        first_line = lines[0].strip() if lines else ""
        if not first_line.startswith('#') and len(content) > 200:
            issues.append(AccessibilityIssue(
                type="missing_title",
                severity="high",
                description="El documento no tiene un título claro",
                suggestion="Agrega un título usando # al inicio del documento"
            ))
        
        # Verificar listas
        has_lists = any(re.match(r'^[\*\-\+]\s', line) or re.match(r'^\d+\.\s', line) for line in lines)
        if not has_lists and len(content) > 1000:
            issues.append(AccessibilityIssue(
                type="no_lists",
                severity="low",
                description="El documento largo no usa listas",
                suggestion="Considera usar listas para mejorar la legibilidad"
            ))
        
        return issues

    def _check_links(self, content: str) -> List[AccessibilityIssue]:
        """Verificar links"""
        issues = []
        
        # Buscar links Markdown
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        matches = link_pattern.finditer(content)
        
        for match in matches:
            link_text = match.group(1)
            url = match.group(2)
            
            # Verificar si el texto del link es descriptivo
            if link_text.lower() in ['click here', 'here', 'más', 'aquí', 'leer más', 'read more']:
                issues.append(AccessibilityIssue(
                    type="non_descriptive_link",
                    severity="medium",
                    description=f"Link con texto no descriptivo: '{link_text}'",
                    suggestion="Usa texto descriptivo para los links en lugar de 'aquí' o 'click aquí'",
                    location=match.start()
                ))
        
        return issues






