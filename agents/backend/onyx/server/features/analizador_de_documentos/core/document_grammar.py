"""
Document Grammar Analyzer - Análisis de Gramática y Redacción
=============================================================

Análisis avanzado de gramática, ortografía y calidad de redacción.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class GrammarIssue:
    """Problema de gramática detectado."""
    issue_type: str  # 'spelling', 'grammar', 'punctuation', 'style'
    message: str
    position: Optional[int] = None
    suggestion: Optional[str] = None
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'


@dataclass
class GrammarAnalysis:
    """Resultado de análisis gramatical."""
    overall_score: float
    spelling_score: float
    grammar_score: float
    punctuation_score: float
    style_score: float
    issues: List[GrammarIssue]
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_words_per_sentence: float
    avg_sentence_length: float
    readability_index: float
    recommendations: List[str]


class GrammarAnalyzer:
    """Analizador de gramática y redacción."""
    
    def __init__(self, analyzer):
        """Inicializar analizador de gramática."""
        self.analyzer = analyzer
        self.common_mistakes = {
            'spelling': ['teh', 'adn', 'taht', 'recieve', 'seperate'],
            'grammar': ['its vs it\'s', 'your vs you\'re', 'there vs their'],
            'punctuation': ['missing comma', 'missing period', 'double space']
        }
    
    async def analyze_grammar(
        self,
        content: str,
        language: str = 'es'
    ) -> GrammarAnalysis:
        """
        Analizar gramática y redacción.
        
        Args:
            content: Contenido del documento
            language: Idioma del documento
        
        Returns:
            GrammarAnalysis con resultados
        """
        # Contar elementos básicos
        word_count = len(content.split())
        sentences = self._split_sentences(content)
        sentence_count = len(sentences)
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Calcular promedios
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        avg_sentence_length = sum(len(s) for s in sentences) / sentence_count if sentence_count > 0 else 0
        
        # Detectar issues
        issues = []
        issues.extend(self._detect_spelling_errors(content))
        issues.extend(self._detect_grammar_errors(content))
        issues.extend(self._detect_punctuation_errors(content))
        issues.extend(self._detect_style_issues(content))
        
        # Calcular scores
        total_issues = len(issues)
        base_score = 100.0
        
        spelling_score = base_score - (len([i for i in issues if i.issue_type == 'spelling']) * 5)
        grammar_score = base_score - (len([i for i in issues if i.issue_type == 'grammar']) * 10)
        punctuation_score = base_score - (len([i for i in issues if i.issue_type == 'punctuation']) * 3)
        style_score = base_score - (len([i for i in issues if i.issue_type == 'style']) * 2)
        
        # Normalizar scores
        spelling_score = max(0, min(100, spelling_score))
        grammar_score = max(0, min(100, grammar_score))
        punctuation_score = max(0, min(100, punctuation_score))
        style_score = max(0, min(100, style_score))
        
        overall_score = (spelling_score + grammar_score + punctuation_score + style_score) / 4
        
        # Calcular índice de legibilidad (Flesch Reading Ease simplificado)
        readability_index = self._calculate_readability_index(content, word_count, sentence_count)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(issues, overall_score)
        
        return GrammarAnalysis(
            overall_score=overall_score,
            spelling_score=spelling_score,
            grammar_score=grammar_score,
            punctuation_score=punctuation_score,
            style_score=style_score,
            issues=issues,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_words_per_sentence=avg_words_per_sentence,
            avg_sentence_length=avg_sentence_length,
            readability_index=readability_index,
            recommendations=recommendations
        )
    
    def _split_sentences(self, content: str) -> List[str]:
        """Dividir en oraciones."""
        # Patrón simple para dividir oraciones
        sentences = re.split(r'[.!?]+\s+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_spelling_errors(self, content: str) -> List[GrammarIssue]:
        """Detectar errores de ortografía."""
        issues = []
        
        # Detectar errores comunes
        words = content.lower().split()
        for mistake in self.common_mistakes['spelling']:
            if mistake in words:
                issues.append(GrammarIssue(
                    issue_type='spelling',
                    message=f"Posible error de ortografía: '{mistake}'",
                    suggestion=self._get_suggestion(mistake),
                    severity='medium'
                ))
        
        # Detectar palabras muy largas o muy cortas (posibles errores)
        for word in words:
            if len(word) > 20:
                issues.append(GrammarIssue(
                    issue_type='spelling',
                    message=f"Palabra muy larga: '{word}'",
                    severity='low'
                ))
        
        return issues
    
    def _detect_grammar_errors(self, content: str) -> List[GrammarIssue]:
        """Detectar errores de gramática."""
        issues = []
        
        # Detectar errores comunes
        if " its " in content.lower() and " it's " not in content.lower():
            issues.append(GrammarIssue(
                issue_type='grammar',
                message="Posible error: 'its' vs 'it's'",
                suggestion="Verificar si debería ser 'it's' (it is)",
                severity='medium'
            ))
        
        # Detectar dobles espacios
        if '  ' in content:
            issues.append(GrammarIssue(
                issue_type='grammar',
                message="Espacios dobles detectados",
                suggestion="Usar un solo espacio entre palabras",
                severity='low'
            ))
        
        # Detectar oraciones muy largas
        sentences = self._split_sentences(content)
        for sentence in sentences:
            if len(sentence.split()) > 30:
                issues.append(GrammarIssue(
                    issue_type='grammar',
                    message="Oración muy larga (más de 30 palabras)",
                    suggestion="Dividir en múltiples oraciones",
                    severity='medium'
                ))
        
        return issues
    
    def _detect_punctuation_errors(self, content: str) -> List[GrammarIssue]:
        """Detectar errores de puntuación."""
        issues = []
        
        # Detectar falta de puntuación al final
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.strip()[-1] in '.!?':
                # Verificar si no es título
                if not line.strip().isupper() and len(line.split()) > 5:
                    issues.append(GrammarIssue(
                        issue_type='punctuation',
                        message=f"Falta puntuación al final (línea {i+1})",
                        severity='low'
                    ))
        
        # Detectar múltiples signos de exclamación/interrogación
        if re.search(r'[!?]{2,}', content):
            issues.append(GrammarIssue(
                issue_type='punctuation',
                message="Múltiples signos de exclamación/interrogación",
                suggestion="Usar un solo signo",
                severity='low'
            ))
        
        return issues
    
    def _detect_style_issues(self, content: str) -> List[GrammarIssue]:
        """Detectar problemas de estilo."""
        issues = []
        
        # Detectar repetición de palabras
        words = content.lower().split()
        word_counts = Counter(words)
        
        for word, count in word_counts.items():
            if count > 10 and len(word) > 3:
                issues.append(GrammarIssue(
                    issue_type='style',
                    message=f"Palabra '{word}' repetida {count} veces",
                    suggestion="Usar sinónimos",
                    severity='low'
                ))
        
        # Detectar uso excesivo de mayúsculas
        if sum(1 for c in content if c.isupper()) / max(len(content), 1) > 0.3:
            issues.append(GrammarIssue(
                issue_type='style',
                message="Uso excesivo de mayúsculas",
                suggestion="Usar mayúsculas solo donde corresponde",
                severity='medium'
            ))
        
        return issues
    
    def _get_suggestion(self, word: str) -> str:
        """Obtener sugerencia para corrección."""
        suggestions = {
            'teh': 'the',
            'adn': 'and',
            'taht': 'that',
            'recieve': 'receive',
            'seperate': 'separate'
        }
        return suggestions.get(word, '')
    
    def _calculate_readability_index(
        self,
        content: str,
        word_count: int,
        sentence_count: int
    ) -> float:
        """Calcular índice de legibilidad (simplificado)."""
        if word_count == 0 or sentence_count == 0:
            return 0.0
        
        # Flesch Reading Ease simplificado
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = self._estimate_syllables(content) / word_count
        
        # Fórmula simplificada
        index = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return max(0, min(100, index))
    
    def _estimate_syllables(self, content: str) -> float:
        """Estimar número de sílabas (simplificado)."""
        words = content.lower().split()
        total_syllables = 0
        
        for word in words:
            # Estimación simple: 1 sílaba por cada 2-3 caracteres
            syllables = max(1, len(word) // 2.5)
            total_syllables += syllables
        
        return total_syllables
    
    def _generate_recommendations(
        self,
        issues: List[GrammarIssue],
        overall_score: float
    ) -> List[str]:
        """Generar recomendaciones."""
        recommendations = []
        
        if overall_score < 60:
            recommendations.append("El documento necesita revisión gramatical significativa")
        
        spelling_issues = [i for i in issues if i.issue_type == 'spelling']
        if len(spelling_issues) > 5:
            recommendations.append("Corregir errores de ortografía")
        
        grammar_issues = [i for i in issues if i.issue_type == 'grammar']
        if len(grammar_issues) > 3:
            recommendations.append("Revisar gramática")
        
        style_issues = [i for i in issues if i.issue_type == 'style']
        if len(style_issues) > 5:
            recommendations.append("Mejorar estilo y evitar repeticiones")
        
        if not recommendations:
            recommendations.append("El documento tiene buena calidad gramatical")
        
        return recommendations
    
    async def suggest_corrections(
        self,
        content: str,
        language: str = 'es'
    ) -> List[Dict[str, Any]]:
        """
        Sugerir correcciones específicas.
        
        Args:
            content: Contenido del documento
            language: Idioma
        
        Returns:
            Lista de sugerencias de corrección
        """
        analysis = await self.analyze_grammar(content, language)
        
        suggestions = []
        for issue in analysis.issues:
            if issue.suggestion:
                suggestions.append({
                    "issue": issue.message,
                    "suggestion": issue.suggestion,
                    "severity": issue.severity,
                    "type": issue.issue_type
                })
        
        return suggestions


__all__ = [
    "GrammarAnalyzer",
    "GrammarAnalysis",
    "GrammarIssue"
]
















