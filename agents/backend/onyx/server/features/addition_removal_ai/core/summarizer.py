"""
Summarizer - Sistema de generación de resúmenes
"""

import logging
from typing import Dict, Any, Optional, List
import re

logger = logging.getLogger(__name__)


class Summarizer:
    """Generador de resúmenes"""

    def __init__(self):
        """Inicializar generador"""
        pass

    def summarize(
        self,
        content: str,
        max_sentences: int = 3,
        method: str = "extractive"
    ) -> Dict[str, Any]:
        """
        Generar resumen del contenido.

        Args:
            content: Contenido a resumir
            max_sentences: Número máximo de oraciones
            method: Método (extractive, abstractive)

        Returns:
            Resumen
        """
        if method == "extractive":
            return self._extractive_summary(content, max_sentences)
        else:
            return self._abstractive_summary(content, max_sentences)

    def _extractive_summary(self, content: str, max_sentences: int) -> Dict[str, Any]:
        """Resumen extractivo"""
        # Dividir en oraciones
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            summary = '. '.join(sentences) + '.'
            return {
                "summary": summary,
                "method": "extractive",
                "original_length": len(content),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(content) if content else 0.0,
                "sentences_used": len(sentences),
                "sentences_total": len(sentences)
            }
        
        # Calcular importancia de oraciones
        sentence_scores = []
        words = content.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 3:  # Ignorar palabras muy cortas
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalizar frecuencias
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Puntuar oraciones
        for i, sentence in enumerate(sentences):
            score = 0.0
            sentence_words = re.sub(r'[^\w\s]', '', sentence.lower()).split()
            
            # Puntuación por frecuencia de palabras
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
            
            # Bonus para oraciones al inicio
            if i < len(sentences) * 0.1:
                score *= 1.2
            
            # Bonus para oraciones con números o fechas
            if re.search(r'\d', sentence):
                score *= 1.1
            
            sentence_scores.append((score, i, sentence))
        
        # Ordenar por puntuación
        sentence_scores.sort(reverse=True)
        
        # Seleccionar mejores oraciones
        selected = sentence_scores[:max_sentences]
        selected.sort(key=lambda x: x[1])  # Ordenar por posición original
        
        summary = '. '.join([s[2] for s in selected]) + '.'
        
        return {
            "summary": summary,
            "method": "extractive",
            "original_length": len(content),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(content) if content else 0.0,
            "sentences_used": len(selected),
            "sentences_total": len(sentences)
        }

    def _abstractive_summary(self, content: str, max_sentences: int) -> Dict[str, Any]:
        """Resumen abstractivo (simplificado)"""
        # Por ahora, usar extractivo como fallback
        # En producción, se usaría un modelo de IA
        return self._extractive_summary(content, max_sentences)

    def summarize_by_sections(self, content: str) -> Dict[str, Any]:
        """
        Resumir por secciones.

        Args:
            content: Contenido con secciones

        Returns:
            Resumen por secciones
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
        
        # Resumir cada sección
        section_summaries = {}
        for section, section_content in sections.items():
            if len(section_content) > 200:
                summary = self.summarize(section_content, max_sentences=2)
                section_summaries[section] = summary["summary"]
            else:
                section_summaries[section] = section_content
        
        return {
            "sections": section_summaries,
            "total_sections": len(sections),
            "method": "section-based"
        }

    def generate_key_points(self, content: str, max_points: int = 5) -> List[str]:
        """
        Generar puntos clave.

        Args:
            content: Contenido
            max_points: Número máximo de puntos

        Returns:
            Lista de puntos clave
        """
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Identificar oraciones importantes
        important_sentences = []
        for sentence in sentences:
            # Criterios de importancia
            is_important = (
                len(sentence) > 50 or  # Oraciones largas
                any(keyword in sentence.lower() for keyword in [
                    'importante', 'clave', 'principal', 'esencial',
                    'debe', 'necesario', 'crucial'
                ]) or
                re.search(r'\d+%', sentence) or  # Contiene porcentajes
                sentence[0].isupper() and len(sentence) > 30  # Oración destacada
            )
            
            if is_important:
                important_sentences.append(sentence)
        
        # Limitar y formatear
        points = important_sentences[:max_points]
        return [f"• {point}" for point in points]






