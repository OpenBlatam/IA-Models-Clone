"""
Quality Analyzer - Sistema de análisis de calidad de contenido
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Niveles de calidad"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class QualityScore:
    """Puntuación de calidad"""
    overall: float
    readability: float
    structure: float
    grammar: float
    coherence: float
    completeness: float
    level: QualityLevel
    issues: List[str]
    suggestions: List[str]


class QualityAnalyzer:
    """Analizador de calidad de contenido"""

    def __init__(self):
        """Inicializar analizador"""
        pass

    def analyze_quality(
        self,
        content: str,
        content_type: Optional[str] = None
    ) -> QualityScore:
        """
        Analizar calidad del contenido.

        Args:
            content: Contenido a analizar
            content_type: Tipo de contenido (opcional)

        Returns:
            Puntuación de calidad
        """
        issues = []
        suggestions = []
        
        # Análisis de legibilidad
        readability_score = self._analyze_readability(content)
        
        # Análisis de estructura
        structure_score = self._analyze_structure(content)
        
        # Análisis de gramática básico
        grammar_score = self._analyze_grammar(content, issues)
        
        # Análisis de coherencia
        coherence_score = self._analyze_coherence(content, issues)
        
        # Análisis de completitud
        completeness_score = self._analyze_completeness(content, issues, suggestions)
        
        # Calcular puntuación general
        overall = (
            readability_score * 0.25 +
            structure_score * 0.20 +
            grammar_score * 0.20 +
            coherence_score * 0.20 +
            completeness_score * 0.15
        )
        
        # Determinar nivel
        if overall >= 0.85:
            level = QualityLevel.EXCELLENT
        elif overall >= 0.70:
            level = QualityLevel.GOOD
        elif overall >= 0.50:
            level = QualityLevel.FAIR
        else:
            level = QualityLevel.POOR
        
        return QualityScore(
            overall=overall,
            readability=readability_score,
            structure=structure_score,
            grammar=grammar_score,
            coherence=coherence_score,
            completeness=completeness_score,
            level=level,
            issues=issues,
            suggestions=suggestions
        )

    def _analyze_readability(self, content: str) -> float:
        """Analizar legibilidad"""
        if not content:
            return 0.0
        
        # Contar palabras y oraciones
        words = len(content.split())
        sentences = len(re.findall(r'[.!?]+', content))
        
        if sentences == 0:
            return 0.5
        
        # Longitud promedio de oraciones
        avg_sentence_length = words / sentences
        
        # Longitud promedio de palabras
        word_lengths = [len(word) for word in content.split()]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # Calcular score (0-1)
        # Ideal: 15-20 palabras por oración, 4-5 caracteres por palabra
        sentence_score = 1.0 - min(abs(avg_sentence_length - 17.5) / 17.5, 1.0)
        word_score = 1.0 - min(abs(avg_word_length - 4.5) / 4.5, 1.0)
        
        return (sentence_score + word_score) / 2

    def _analyze_structure(self, content: str) -> float:
        """Analizar estructura"""
        if not content:
            return 0.0
        
        score = 1.0
        
        # Verificar párrafos
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            score -= 0.2
        
        # Verificar headers (Markdown)
        headers = len(re.findall(r'^#+\s', content, re.MULTILINE))
        if headers == 0 and len(content) > 500:
            score -= 0.2
        
        # Verificar listas
        lists = len(re.findall(r'^[\*\-\+]\s', content, re.MULTILINE))
        if lists == 0 and len(paragraphs) > 3:
            score -= 0.1
        
        # Verificar longitud de líneas
        lines = content.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines > len(lines) * 0.3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))

    def _analyze_grammar(self, content: str, issues: List[str]) -> float:
        """Analizar gramática básica"""
        if not content:
            return 0.0
        
        score = 1.0
        
        # Verificar mayúsculas al inicio de oraciones
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                issues.append("Falta mayúscula al inicio de oración")
                score -= 0.05
        
        # Verificar espacios dobles
        if '  ' in content:
            issues.append("Espacios dobles detectados")
            score -= 0.1
        
        # Verificar puntuación al final
        if content and content[-1] not in '.!?':
            issues.append("Falta puntuación al final")
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _analyze_coherence(self, content: str, issues: List[str]) -> float:
        """Analizar coherencia"""
        if not content:
            return 0.0
        
        score = 1.0
        
        # Verificar repetición de palabras
        words = content.lower().split()
        word_counts = {}
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 3:  # Ignorar palabras cortas
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Detectar repeticiones excesivas
        for word, count in word_counts.items():
            if count > len(words) * 0.1:  # Más del 10% del contenido
                issues.append(f"Palabra '{word}' repetida excesivamente")
                score -= 0.1
        
        # Verificar transiciones
        transitions = ['sin embargo', 'además', 'por lo tanto', 'en consecuencia', 
                      'por otro lado', 'finalmente', 'en resumen']
        found_transitions = sum(1 for t in transitions if t in content.lower())
        if len(content.split('.')) > 3 and found_transitions == 0:
            issues.append("Faltan palabras de transición")
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _analyze_completeness(
        self,
        content: str,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Analizar completitud"""
        if not content:
            return 0.0
        
        score = 1.0
        
        # Verificar longitud mínima
        if len(content) < 100:
            issues.append("Contenido muy corto")
            suggestions.append("Considera expandir el contenido")
            score -= 0.3
        
        # Verificar si tiene introducción y conclusión
        first_paragraph = content.split('\n\n')[0] if '\n\n' in content else content[:200]
        last_paragraph = content.split('\n\n')[-1] if '\n\n' in content else content[-200:]
        
        intro_keywords = ['introducción', 'resumen', 'objetivo', 'propósito']
        concl_keywords = ['conclusión', 'resumen', 'finalmente', 'en resumen']
        
        has_intro = any(kw in first_paragraph.lower() for kw in intro_keywords)
        has_concl = any(kw in last_paragraph.lower() for kw in concl_keywords)
        
        if not has_intro and len(content) > 500:
            suggestions.append("Considera agregar una introducción")
            score -= 0.1
        
        if not has_concl and len(content) > 500:
            suggestions.append("Considera agregar una conclusión")
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def get_quality_report(self, content: str) -> Dict[str, Any]:
        """
        Generar reporte de calidad.

        Args:
            content: Contenido

        Returns:
            Reporte de calidad
        """
        score = self.analyze_quality(content)
        
        return {
            "overall_score": score.overall,
            "level": score.level.value,
            "scores": {
                "readability": score.readability,
                "structure": score.structure,
                "grammar": score.grammar,
                "coherence": score.coherence,
                "completeness": score.completeness
            },
            "issues": score.issues,
            "suggestions": score.suggestions,
            "metrics": {
                "word_count": len(content.split()),
                "sentence_count": len(re.findall(r'[.!?]+', content)),
                "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
                "character_count": len(content)
            }
        }






