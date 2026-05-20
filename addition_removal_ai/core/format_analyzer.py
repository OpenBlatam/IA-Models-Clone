"""
Format Analyzer - Sistema de análisis de formato
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FormatIssue:
    """Problema de formato"""
    type: str
    severity: str
    description: str
    suggestion: str
    location: Optional[int] = None


class FormatAnalyzer:
    """Analizador de formato"""

    def __init__(self):
        """Inicializar analizador"""
        pass

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar formato del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de formato
        """
        issues = []
        
        # Verificar espacios
        space_issues = self._check_spaces(content)
        issues.extend(space_issues)
        
        # Verificar puntuación
        punctuation_issues = self._check_punctuation(content)
        issues.extend(punctuation_issues)
        
        # Verificar mayúsculas
        capitalization_issues = self._check_capitalization(content)
        issues.extend(capitalization_issues)
        
        # Verificar líneas largas
        line_length_issues = self._check_line_length(content)
        issues.extend(line_length_issues)
        
        # Verificar formato consistente
        consistency_issues = self._check_consistency(content)
        issues.extend(consistency_issues)
        
        # Calcular score
        total_issues = len(issues)
        critical_issues = len([i for i in issues if i.severity == "critical"])
        high_issues = len([i for i in issues if i.severity == "high"])
        
        # Score: 100 - (critical * 15 + high * 10 + medium * 5 + low * 1)
        score = 100
        for issue in issues:
            if issue.severity == "critical":
                score -= 15
            elif issue.severity == "high":
                score -= 10
            elif issue.severity == "medium":
                score -= 5
            else:
                score -= 1
        
        score = max(0, min(100, score))
        
        return {
            "format_score": score,
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
            "is_well_formatted": score >= 80
        }

    def _check_spaces(self, content: str) -> List[FormatIssue]:
        """Verificar espacios"""
        issues = []
        
        # Espacios dobles
        double_space_pattern = re.compile(r' {2,}')
        for match in double_space_pattern.finditer(content):
            issues.append(FormatIssue(
                type="double_spaces",
                severity="medium",
                description="Espacios dobles detectados",
                suggestion="Elimina espacios duplicados",
                location=match.start()
            ))
        
        # Espacios antes de puntuación
        space_before_punct = re.compile(r'\s+[.,;:!?]')
        for match in space_before_punct.finditer(content):
            issues.append(FormatIssue(
                type="space_before_punctuation",
                severity="low",
                description="Espacio antes de puntuación",
                suggestion="Elimina espacios antes de signos de puntuación",
                location=match.start()
            ))
        
        # Falta espacio después de puntuación
        no_space_after_punct = re.compile(r'[.,;:!?][A-Za-z]')
        for match in no_space_after_punct.finditer(content):
            issues.append(FormatIssue(
                type="missing_space_after_punctuation",
                severity="high",
                description="Falta espacio después de puntuación",
                suggestion="Agrega un espacio después de signos de puntuación",
                location=match.start()
            ))
        
        return issues

    def _check_punctuation(self, content: str) -> List[FormatIssue]:
        """Verificar puntuación"""
        issues = []
        
        # Falta puntuación al final
        if content and content[-1] not in '.!?':
            issues.append(FormatIssue(
                type="missing_final_punctuation",
                severity="medium",
                description="Falta puntuación al final del contenido",
                suggestion="Agrega un punto, signo de exclamación o interrogación al final"
            ))
        
        # Múltiples signos de exclamación/interrogación
        multiple_excl = re.compile(r'[!?]{3,}')
        for match in multiple_excl.finditer(content):
            issues.append(FormatIssue(
                type="multiple_punctuation",
                severity="low",
                description="Múltiples signos de puntuación",
                suggestion="Usa máximo dos signos de exclamación o interrogación",
                location=match.start()
            ))
        
        return issues

    def _check_capitalization(self, content: str) -> List[FormatIssue]:
        """Verificar mayúsculas"""
        issues = []
        sentences = re.split(r'[.!?]+', content)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                # Ignorar si empieza con número o símbolo
                if not sentence[0].isdigit() and sentence[0].isalnum():
                    issues.append(FormatIssue(
                        type="missing_capitalization",
                        severity="high",
                        description=f"Falta mayúscula al inicio de oración",
                        suggestion="Capitaliza la primera letra de cada oración"
                    ))
        
        return issues

    def _check_line_length(self, content: str) -> List[FormatIssue]:
        """Verificar longitud de líneas"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if len(line) > 100:  # Línea muy larga
                issues.append(FormatIssue(
                    type="long_line",
                    severity="low",
                    description=f"Línea {i+1} es muy larga ({len(line)} caracteres)",
                    suggestion="Divide líneas largas para mejorar legibilidad"
                ))
        
        return issues

    def _check_consistency(self, content: str) -> List[FormatIssue]:
        """Verificar consistencia"""
        issues = []
        
        # Mezcla de comillas
        has_single_quotes = "'" in content
        has_double_quotes = '"' in content
        if has_single_quotes and has_double_quotes:
            issues.append(FormatIssue(
                type="mixed_quotes",
                severity="low",
                description="Mezcla de comillas simples y dobles",
                suggestion="Usa un tipo de comillas consistentemente"
            ))
        
        # Mezcla de guiones
        has_hyphens = '-' in content
        has_em_dashes = '—' in content or '–' in content
        if has_hyphens and has_em_dashes:
            issues.append(FormatIssue(
                type="mixed_dashes",
                severity="low",
                description="Mezcla de guiones y rayas",
                suggestion="Usa un tipo de guión consistentemente"
            ))
        
        return issues






