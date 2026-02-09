"""
Topic Modeler - Sistema de modelado de temas
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class TopicModeler:
    """Modelador de temas"""

    def __init__(self):
        """Inicializar modelador"""
        self.stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'en', 'a', 'y', 'o', 'pero', 'si', 'no',
            'que', 'es', 'son', 'fue', 'ser', 'estar', 'tener',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should'
        }

    def extract_topics(
        self,
        content: str,
        num_topics: int = 5,
        min_word_length: int = 4
    ) -> Dict[str, Any]:
        """
        Extraer temas del contenido.

        Args:
            content: Contenido
            num_topics: Número de temas
            min_word_length: Longitud mínima de palabra

        Returns:
            Temas extraídos
        """
        # Tokenizar y limpiar
        words = self._tokenize(content, min_word_length)
        
        if not words:
            return {"topics": [], "error": "No se encontraron palabras válidas"}
        
        # Contar frecuencias
        word_freq = Counter(words)
        
        # Filtrar palabras muy comunes o muy raras
        total_words = len(words)
        filtered_words = {
            word: count
            for word, count in word_freq.items()
            if count >= 2 and count <= total_words * 0.5
        }
        
        # Obtener temas (palabras más frecuentes)
        top_words = sorted(
            filtered_words.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_topics]
        
        topics = [
            {
                "topic": word,
                "frequency": count,
                "percentage": (count / total_words) * 100
            }
            for word, count in top_words
        ]
        
        return {
            "topics": topics,
            "total_words": total_words,
            "unique_words": len(word_freq)
        }

    def extract_topics_by_sections(
        self,
        content: str,
        num_topics_per_section: int = 3
    ) -> Dict[str, Any]:
        """
        Extraer temas por secciones.

        Args:
            content: Contenido con secciones
            num_topics_per_section: Temas por sección

        Returns:
            Temas por sección
        """
        # Dividir en secciones (por headers Markdown o párrafos)
        sections = self._split_into_sections(content)
        
        section_topics = {}
        for section_name, section_content in sections.items():
            topics = self.extract_topics(
                section_content,
                num_topics=num_topics_per_section
            )
            section_topics[section_name] = topics.get("topics", [])
        
        return {
            "sections": section_topics,
            "total_sections": len(sections)
        }

    def _tokenize(self, text: str, min_length: int = 4) -> List[str]:
        """
        Tokenizar texto.

        Args:
            text: Texto
            min_length: Longitud mínima

        Returns:
            Lista de tokens
        """
        # Normalizar
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Filtrar stop words y palabras cortas
        words = [
            w for w in words
            if w not in self.stop_words and len(w) >= min_length
        ]
        
        return words

    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """
        Dividir contenido en secciones.

        Args:
            content: Contenido

        Returns:
            Diccionario de secciones
        """
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            # Detectar headers (Markdown)
            if line.startswith('#'):
                if current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = line.lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = ' '.join(current_content)
        
        # Si no hay headers, dividir por párrafos
        if len(sections) == 1:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            sections = {
                f"Section {i+1}": para
                for i, para in enumerate(paragraphs)
            }
        
        return sections

    def get_topic_keywords(
        self,
        content: str,
        topic: str,
        num_keywords: int = 10
    ) -> List[str]:
        """
        Obtener palabras clave relacionadas con un tema.

        Args:
            content: Contenido
            topic: Tema
            num_keywords: Número de palabras clave

        Returns:
            Lista de palabras clave
        """
        words = self._tokenize(content)
        
        # Encontrar contexto alrededor del tema
        topic_lower = topic.lower()
        context_words = []
        
        for i, word in enumerate(words):
            if word == topic_lower:
                # Agregar palabras cercanas
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                context_words.extend(words[start:end])
        
        # Contar palabras del contexto
        context_freq = Counter(context_words)
        
        # Filtrar el tema mismo y stop words
        filtered = {
            word: count
            for word, count in context_freq.items()
            if word != topic_lower and word not in self.stop_words
        }
        
        # Obtener top keywords
        top_keywords = sorted(
            filtered.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_keywords]
        
        return [word for word, _ in top_keywords]






