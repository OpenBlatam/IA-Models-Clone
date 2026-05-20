"""
Readability Advanced - Sistema avanzado de análisis de legibilidad
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ReadabilityLevel(Enum):
    """Niveles de legibilidad"""
    VERY_EASY = "very_easy"
    EASY = "easy"
    FAIRLY_EASY = "fairly_easy"
    STANDARD = "standard"
    FAIRLY_DIFFICULT = "fairly_difficult"
    DIFFICULT = "difficult"
    VERY_DIFFICULT = "very_difficult"


@dataclass
class ReadabilityScores:
    """Puntuaciones de legibilidad"""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    smog_index: float
    coleman_liau: float
    automated_readability: float
    average_grade_level: float
    readability_level: ReadabilityLevel


class AdvancedReadabilityAnalyzer:
    """Analizador avanzado de legibilidad"""

    def __init__(self):
        """Inicializar analizador"""
        pass

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar legibilidad del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de legibilidad
        """
        if not content:
            return {"error": "Contenido vacío"}
        
        # Métricas básicas
        sentences = self._split_sentences(content)
        words = self._split_words(content)
        syllables = self._count_syllables(words)
        
        # Calcular diferentes índices de legibilidad
        flesch_reading_ease = self._flesch_reading_ease(sentences, words, syllables)
        flesch_kincaid_grade = self._flesch_kincaid_grade(sentences, words, syllables)
        gunning_fog = self._gunning_fog(sentences, words)
        smog_index = self._smog_index(sentences, words)
        coleman_liau = self._coleman_liau(content, sentences, words)
        automated_readability = self._automated_readability(content, sentences, words)
        
        # Promedio de nivel de grado
        average_grade_level = (
            flesch_kincaid_grade + gunning_fog + coleman_liau + automated_readability
        ) / 4
        
        # Determinar nivel de legibilidad
        readability_level = self._determine_readability_level(flesch_reading_ease)
        
        return {
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "gunning_fog": gunning_fog,
            "smog_index": smog_index,
            "coleman_liau": coleman_liau,
            "automated_readability": automated_readability,
            "average_grade_level": average_grade_level,
            "readability_level": readability_level.value,
            "metrics": {
                "sentence_count": len(sentences),
                "word_count": len(words),
                "syllable_count": syllables,
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
                "avg_syllables_per_word": syllables / len(words) if words else 0
            },
            "interpretation": self._interpret_readability(flesch_reading_ease, readability_level)
        }

    def _split_sentences(self, content: str) -> List[str]:
        """Dividir en oraciones"""
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]

    def _split_words(self, content: str) -> List[str]:
        """Dividir en palabras"""
        words = re.findall(r'\b\w+\b', content)
        return [w.lower() for w in words]

    def _count_syllables(self, words: List[str]) -> int:
        """Contar sílabas"""
        total = 0
        for word in words:
            total += self._count_word_syllables(word)
        return total

    def _count_word_syllables(self, word: str) -> int:
        """Contar sílabas de una palabra"""
        word = word.lower()
        if not word:
            return 0
        
        # Remover 'e' silenciosa al final
        if word.endswith('e'):
            word = word[:-1]
        
        # Contar grupos de vocales
        vowels = re.findall(r'[aeiouáéíóúü]+', word)
        syllable_count = len(vowels)
        
        # Mínimo 1 sílaba
        return max(1, syllable_count)

    def _flesch_reading_ease(self, sentences: List[str], words: List[str], syllables: int) -> float:
        """Flesch Reading Ease Score"""
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words) if words else 0
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))

    def _flesch_kincaid_grade(self, sentences: List[str], words: List[str], syllables: int) -> float:
        """Flesch-Kincaid Grade Level"""
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words) if words else 0
        
        score = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        return max(0, score)

    def _gunning_fog(self, sentences: List[str], words: List[str]) -> float:
        """Gunning Fog Index"""
        if not sentences or not words:
            return 0.0
        
        # Contar palabras complejas (3+ sílabas)
        complex_words = 0
        for word in words:
            if self._count_word_syllables(word) >= 3:
                complex_words += 1
        
        avg_sentence_length = len(words) / len(sentences)
        percentage_complex = (complex_words / len(words)) * 100 if words else 0
        
        score = 0.4 * (avg_sentence_length + percentage_complex)
        return max(0, score)

    def _smog_index(self, sentences: List[str], words: List[str]) -> float:
        """SMOG Index"""
        if len(sentences) < 30:
            return 0.0
        
        # Contar palabras de 3+ sílabas en las primeras 30 oraciones
        sample_sentences = sentences[:30]
        sample_words = []
        for sentence in sample_sentences:
            sample_words.extend(re.findall(r'\b\w+\b', sentence.lower()))
        
        polysyllables = sum(1 for word in sample_words if self._count_word_syllables(word) >= 3)
        
        score = 1.043 * (polysyllables ** 0.5) + 3.1291
        return max(0, score)

    def _coleman_liau(self, content: str, sentences: List[str], words: List[str]) -> float:
        """Coleman-Liau Index"""
        if not sentences or not words:
            return 0.0
        
        characters = len(re.sub(r'\s', '', content))
        letters = len(re.findall(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', content))
        
        avg_sentence_length = (len(words) / len(sentences)) * 100
        avg_letters_per_word = (letters / len(words)) * 100 if words else 0
        
        score = (0.0588 * avg_letters_per_word) - (0.296 * avg_sentence_length) - 15.8
        return max(0, score)

    def _automated_readability(self, content: str, sentences: List[str], words: List[str]) -> float:
        """Automated Readability Index"""
        if not sentences or not words:
            return 0.0
        
        characters = len(re.sub(r'\s', '', content))
        
        avg_sentence_length = len(words) / len(sentences)
        avg_characters_per_word = characters / len(words) if words else 0
        
        score = (4.71 * avg_characters_per_word) + (0.5 * avg_sentence_length) - 21.43
        return max(0, score)

    def _determine_readability_level(self, flesch_score: float) -> ReadabilityLevel:
        """Determinar nivel de legibilidad"""
        if flesch_score >= 90:
            return ReadabilityLevel.VERY_EASY
        elif flesch_score >= 80:
            return ReadabilityLevel.EASY
        elif flesch_score >= 70:
            return ReadabilityLevel.FAIRLY_EASY
        elif flesch_score >= 60:
            return ReadabilityLevel.STANDARD
        elif flesch_score >= 50:
            return ReadabilityLevel.FAIRLY_DIFFICULT
        elif flesch_score >= 30:
            return ReadabilityLevel.DIFFICULT
        else:
            return ReadabilityLevel.VERY_DIFFICULT

    def _interpret_readability(
        self,
        flesch_score: float,
        level: ReadabilityLevel
    ) -> Dict[str, Any]:
        """Interpretar legibilidad"""
        interpretations = {
            ReadabilityLevel.VERY_EASY: {
                "description": "Muy fácil de leer",
                "audience": "Niños de 5-6 años",
                "suggestion": "El contenido es muy simple. Considera agregar más complejidad si es necesario."
            },
            ReadabilityLevel.EASY: {
                "description": "Fácil de leer",
                "audience": "Niños de 7-8 años",
                "suggestion": "El contenido es fácil de entender."
            },
            ReadabilityLevel.FAIRLY_EASY: {
                "description": "Bastante fácil de leer",
                "audience": "Estudiantes de primaria",
                "suggestion": "El contenido es accesible para la mayoría de lectores."
            },
            ReadabilityLevel.STANDARD: {
                "description": "Legibilidad estándar",
                "audience": "Estudiantes de secundaria",
                "suggestion": "El contenido tiene un nivel de legibilidad apropiado."
            },
            ReadabilityLevel.FAIRLY_DIFFICULT: {
                "description": "Bastante difícil de leer",
                "audience": "Estudiantes universitarios",
                "suggestion": "Considera simplificar algunas oraciones."
            },
            ReadabilityLevel.DIFFICULT: {
                "description": "Difícil de leer",
                "audience": "Graduados universitarios",
                "suggestion": "El contenido es complejo. Considera simplificar."
            },
            ReadabilityLevel.VERY_DIFFICULT: {
                "description": "Muy difícil de leer",
                "audience": "Profesionales con educación avanzada",
                "suggestion": "El contenido es muy complejo. Simplifica oraciones y vocabulario."
            }
        }
        
        return interpretations.get(level, {
            "description": "Nivel desconocido",
            "audience": "General",
            "suggestion": "Revisa el contenido."
        })






