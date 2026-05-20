"""
Spell Checker - Sistema de corrección ortográfica
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
import string

logger = logging.getLogger(__name__)


class SpellChecker:
    """Corrector ortográfico"""

    def __init__(self):
        """Inicializar corrector"""
        # Diccionario básico de palabras comunes
        self.dictionary = set()
        self._load_basic_dictionary()

    def _load_basic_dictionary(self):
        """Cargar diccionario básico"""
        # Palabras comunes en español
        spanish_words = [
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'en', 'a', 'y', 'o', 'pero', 'si', 'no',
            'que', 'es', 'son', 'fue', 'ser', 'estar', 'tener',
            'hacer', 'decir', 'ir', 'ver', 'dar', 'saber', 'poder',
            'querer', 'llegar', 'pasar', 'deber', 'poner', 'parecer',
            'quedar', 'hablar', 'llevar', 'dejar', 'seguir', 'encontrar',
            'llamar', 'venir', 'pensar', 'salir', 'volver', 'tomar',
            'conocer', 'vivir', 'sentir', 'tratar', 'mirar', 'contar',
            'empezar', 'esperar', 'buscar', 'existir', 'entrar', 'trabajar',
            'escribir', 'perder', 'producir', 'ocurrir', 'entender', 'pedir',
            'recibir', 'recordar', 'terminar', 'permitir', 'aparecer',
            'conseguir', 'comenzar', 'servir', 'sacar', 'necesitar',
            'mantener', 'resultar', 'leer', 'caer', 'cambiar', 'presentar',
            'crear', 'abrir', 'considerar', 'oír', 'acabar', 'ganar',
            'formar', 'traer', 'partir', 'morir', 'aceptar', 'realizar',
            'suponer', 'comprender', 'lograr', 'explicar', 'preguntar',
            'tocar', 'reconocer', 'estudiar', 'alcanzar', 'nacer', 'dirigir',
            'correr', 'utilizar', 'pagar', 'ayudar', 'gustar', 'jugar',
            'escuchar', 'contar', 'contar', 'contar', 'contar', 'contar'
        ]
        
        # Palabras comunes en inglés
        english_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was',
            'were', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'shall', 'can', 'cannot'
        ]
        
        self.dictionary.update(spanish_words)
        self.dictionary.update(english_words)

    def check(
        self,
        text: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verificar ortografía.

        Args:
            text: Texto a verificar
            language: Idioma (opcional)

        Returns:
            Resultado de verificación
        """
        words = re.findall(r'\b\w+\b', text.lower())
        misspelled = []
        suggestions_map = {}
        
        for word in words:
            if word not in self.dictionary and len(word) > 2:
                misspelled.append(word)
                suggestions = self._suggest_corrections(word)
                if suggestions:
                    suggestions_map[word] = suggestions
        
        return {
            "text": text,
            "total_words": len(words),
            "misspelled_count": len(misspelled),
            "misspelled_words": misspelled,
            "suggestions": suggestions_map,
            "accuracy": 1.0 - (len(misspelled) / len(words)) if words else 1.0
        }

    def correct(
        self,
        text: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Corregir texto.

        Args:
            text: Texto a corregir
            language: Idioma (opcional)

        Returns:
            Texto corregido
        """
        check_result = self.check(text, language)
        
        corrected_text = text
        corrections = []
        
        for word in check_result["misspelled_words"]:
            suggestions = check_result["suggestions"].get(word, [])
            if suggestions:
                # Usar primera sugerencia
                correction = suggestions[0]
                # Reemplazar palabra (case-insensitive)
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                corrected_text = pattern.sub(correction, corrected_text)
                corrections.append({
                    "original": word,
                    "corrected": correction,
                    "position": text.lower().find(word)
                })
        
        return {
            "original": text,
            "corrected": corrected_text,
            "corrections": corrections,
            "corrections_count": len(corrections)
        }

    def _suggest_corrections(self, word: str, max_suggestions: int = 5) -> List[str]:
        """
        Sugerir correcciones para una palabra.

        Args:
            word: Palabra incorrecta
            max_suggestions: Número máximo de sugerencias

        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        # Generar candidatos con ediciones simples
        candidates = self._generate_candidates(word)
        
        # Filtrar candidatos que están en el diccionario
        valid_candidates = [c for c in candidates if c in self.dictionary]
        
        # Ordenar por similitud
        valid_candidates.sort(key=lambda c: self._edit_distance(word, c))
        
        return valid_candidates[:max_suggestions]

    def _generate_candidates(self, word: str) -> List[str]:
        """
        Generar candidatos de corrección.

        Args:
            word: Palabra

        Returns:
            Lista de candidatos
        """
        candidates = []
        
        # Deletions (eliminar un carácter)
        for i in range(len(word)):
            candidate = word[:i] + word[i+1:]
            if candidate:
                candidates.append(candidate)
        
        # Insertions (insertar un carácter)
        for i in range(len(word) + 1):
            for char in string.ascii_lowercase:
                candidate = word[:i] + char + word[i:]
                candidates.append(candidate)
        
        # Substitutions (sustituir un carácter)
        for i in range(len(word)):
            for char in string.ascii_lowercase:
                if word[i] != char:
                    candidate = word[:i] + char + word[i+1:]
                    candidates.append(candidate)
        
        # Transpositions (intercambiar caracteres adyacentes)
        for i in range(len(word) - 1):
            candidate = word[:i] + word[i+1] + word[i] + word[i+2:]
            candidates.append(candidate)
        
        return list(set(candidates))  # Eliminar duplicados

    def _edit_distance(self, word1: str, word2: str) -> int:
        """
        Calcular distancia de edición (Levenshtein).

        Args:
            word1: Palabra 1
            word2: Palabra 2

        Returns:
            Distancia
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # deletion
                        dp[i][j-1],      # insertion
                        dp[i-1][j-1]     # substitution
                    )
        
        return dp[m][n]

    def add_to_dictionary(self, word: str):
        """
        Agregar palabra al diccionario.

        Args:
            word: Palabra a agregar
        """
        self.dictionary.add(word.lower())
        logger.debug(f"Palabra agregada al diccionario: {word}")






