"""
Content Generator - Sistema de generación de contenido automático
"""

import logging
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Generador de contenido automático"""

    def __init__(self):
        """Inicializar generador"""
        # Plantillas de contenido
        self.templates = {
            "introduction": [
                "Este documento presenta {topic}.",
                "En este artículo, exploraremos {topic}.",
                "El siguiente contenido aborda {topic}.",
                "Este texto trata sobre {topic}."
            ],
            "conclusion": [
                "En conclusión, {summary}.",
                "Para resumir, {summary}.",
                "En resumen, {summary}.",
                "Finalmente, {summary}."
            ],
            "paragraph": [
                "{content} Además, es importante considerar {additional}.",
                "{content} Por otro lado, {additional}.",
                "{content} Sin embargo, {additional}.",
                "{content} Por lo tanto, {additional}."
            ]
        }

    def generate_introduction(
        self,
        topic: str,
        length: str = "medium"
    ) -> str:
        """
        Generar introducción.

        Args:
            topic: Tema
            length: Longitud (short, medium, long)

        Returns:
            Introducción generada
        """
        template = random.choice(self.templates["introduction"])
        intro = template.format(topic=topic)
        
        if length == "long":
            intro += " Este tema es de gran importancia y será analizado en detalle."
        elif length == "short":
            pass  # Ya es corta
        
        return intro

    def generate_conclusion(
        self,
        summary: str,
        length: str = "medium"
    ) -> str:
        """
        Generar conclusión.

        Args:
            summary: Resumen
            length: Longitud

        Returns:
            Conclusión generada
        """
        template = random.choice(self.templates["conclusion"])
        conclusion = template.format(summary=summary)
        
        if length == "long":
            conclusion += " Se espera que esta información sea útil para el lector."
        
        return conclusion

    def expand_content(
        self,
        content: str,
        target_length: int,
        style: str = "formal"
    ) -> str:
        """
        Expandir contenido.

        Args:
            content: Contenido original
            target_length: Longitud objetivo
            style: Estilo (formal, informal, technical)

        Returns:
            Contenido expandido
        """
        current_length = len(content)
        
        if current_length >= target_length:
            return content
        
        # Calcular cuánto expandir
        expansion_needed = target_length - current_length
        
        # Dividir en oraciones
        sentences = content.split('. ')
        
        # Agregar contenido adicional
        expanded = content
        
        # Frases de expansión según estilo
        expansion_phrases = {
            "formal": [
                "Es importante mencionar que",
                "Cabe destacar que",
                "Vale la pena señalar que",
                "Es relevante considerar que"
            ],
            "informal": [
                "También hay que decir que",
                "Otra cosa es que",
                "Además, hay que tener en cuenta que",
                "Por si fuera poco,"
            ],
            "technical": [
                "Desde un punto de vista técnico,",
                "Técnicamente hablando,",
                "En términos técnicos,",
                "Desde la perspectiva técnica,"
            ]
        }
        
        phrases = expansion_phrases.get(style, expansion_phrases["formal"])
        
        while len(expanded) < target_length:
            phrase = random.choice(phrases)
            additional = f"{phrase} este aspecto requiere atención adicional. "
            expanded += additional
            
            if len(expanded) >= target_length:
                break
        
        return expanded[:target_length]

    def generate_bullet_points(
        self,
        topic: str,
        num_points: int = 5
    ) -> List[str]:
        """
        Generar puntos de lista.

        Args:
            topic: Tema
            num_points: Número de puntos

        Returns:
            Lista de puntos
        """
        points = []
        
        for i in range(num_points):
            point = f"Punto importante {i+1} relacionado con {topic}"
            points.append(point)
        
        return points

    def generate_from_template(
        self,
        template_type: str,
        **kwargs
    ) -> str:
        """
        Generar contenido desde plantilla.

        Args:
            template_type: Tipo de plantilla
            **kwargs: Argumentos para la plantilla

        Returns:
            Contenido generado
        """
        if template_type not in self.templates:
            return ""
        
        templates = self.templates[template_type]
        template = random.choice(templates)
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Faltan argumentos en plantilla: {e}")
            return template

    def suggest_improvements(
        self,
        content: str
    ) -> List[str]:
        """
        Sugerir mejoras al contenido.

        Args:
            content: Contenido

        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        # Verificar longitud
        if len(content) < 100:
            suggestions.append("El contenido es muy corto. Considera expandirlo.")
        
        # Verificar párrafos
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            suggestions.append("Considera dividir el contenido en más párrafos.")
        
        # Verificar oraciones
        sentences = [s for s in content.split('.') if s.strip()]
        if len(sentences) < 3:
            suggestions.append("El contenido tiene muy pocas oraciones. Considera agregar más.")
        
        # Verificar estructura
        if not any(keyword in content.lower() for keyword in ['introducción', 'conclusión', 'introduction', 'conclusion']):
            suggestions.append("Considera agregar una introducción y conclusión.")
        
        return suggestions






