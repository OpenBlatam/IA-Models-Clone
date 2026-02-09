"""
Content Optimizer - Optimizador de Contenido
=============================================

Utilidades para optimizar contenido para diferentes plataformas.
"""

import re
from typing import Dict, List, Optional
from datetime import datetime


class ContentOptimizer:
    """Optimizador de contenido para redes sociales"""
    
    # Límites de caracteres por plataforma
    PLATFORM_LIMITS = {
        "twitter": 280,
        "x": 280,
        "facebook": 5000,
        "instagram": 2200,
        "linkedin": 3000,
        "tiktok": 2200,
        "youtube": 10000,
    }
    
    # Mejores horarios por plataforma (en horas)
    BEST_POSTING_TIMES = {
        "facebook": [9, 13, 15, 17],
        "instagram": [11, 14, 15, 17],
        "twitter": [8, 12, 15, 17],
        "linkedin": [8, 12, 17],
        "tiktok": [9, 12, 19, 21],
        "youtube": [14, 15, 16, 17],
    }
    
    @staticmethod
    def optimize_length(content: str, platform: str) -> str:
        """
        Optimizar longitud de contenido
        
        Args:
            content: Contenido original
            platform: Plataforma objetivo
            
        Returns:
            Contenido optimizado
        """
        limit = ContentOptimizer.PLATFORM_LIMITS.get(platform.lower(), 5000)
        
        if len(content) <= limit:
            return content
        
        # Truncar inteligentemente
        truncated = content[:limit - 3]
        
        # Intentar truncar en un punto lógico
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        last_space = truncated.rfind(' ')
        
        cut_point = max(last_period, last_newline, last_space)
        
        if cut_point > limit * 0.8:  # Si el punto de corte es razonable
            truncated = truncated[:cut_point]
        
        return truncated + "..."
    
    @staticmethod
    def add_hashtags(content: str, hashtags: List[str], max_hashtags: int = 10) -> str:
        """
        Agregar hashtags al contenido
        
        Args:
            content: Contenido original
            hashtags: Lista de hashtags
            max_hashtags: Máximo de hashtags a agregar
            
        Returns:
            Contenido con hashtags
        """
        if not hashtags:
            return content
        
        # Limitar número de hashtags
        hashtags = hashtags[:max_hashtags]
        
        # Formatear hashtags
        hashtag_str = " ".join([f"#{tag}" if not tag.startswith("#") else tag for tag in hashtags])
        
        # Agregar al final
        return f"{content}\n\n{hashtag_str}"
    
    @staticmethod
    def optimize_for_platform(
        content: str,
        platform: str,
        hashtags: Optional[List[str]] = None
    ) -> str:
        """
        Optimizar contenido completo para una plataforma
        
        Args:
            content: Contenido original
            platform: Plataforma objetivo
            hashtags: Hashtags opcionales
            
        Returns:
            Contenido optimizado
        """
        # Optimizar longitud
        optimized = ContentOptimizer.optimize_length(content, platform)
        
        # Agregar hashtags si se proporcionan
        if hashtags:
            optimized = ContentOptimizer.add_hashtags(optimized, hashtags)
        
        return optimized
    
    @staticmethod
    def suggest_posting_time(platform: str) -> List[int]:
        """
        Sugerir mejores horarios para publicar
        
        Args:
            platform: Plataforma
            
        Returns:
            Lista de horas sugeridas
        """
        return ContentOptimizer.BEST_POSTING_TIMES.get(platform.lower(), [12, 15, 18])
    
    @staticmethod
    def calculate_read_time(content: str, words_per_minute: int = 200) -> float:
        """
        Calcular tiempo de lectura estimado
        
        Args:
            content: Contenido
            words_per_minute: Palabras por minuto (default: 200)
            
        Returns:
            Tiempo en minutos
        """
        words = len(content.split())
        return words / words_per_minute
    
    @staticmethod
    def extract_keywords(content: str, max_keywords: int = 10) -> List[str]:
        """
        Extraer palabras clave del contenido
        
        Args:
            content: Contenido
            max_keywords: Máximo de keywords
            
        Returns:
            Lista de keywords
        """
        # Remover stop words comunes
        stop_words = {
            "el", "la", "los", "las", "un", "una", "de", "del", "en", "a", "y", "o",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"
        }
        
        # Extraer palabras
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Filtrar stop words y palabras muy cortas
        keywords = [
            word for word in words
            if word not in stop_words and len(word) > 3
        ]
        
        # Contar frecuencia
        from collections import Counter
        word_freq = Counter(keywords)
        
        # Retornar más frecuentes
        return [word for word, _ in word_freq.most_common(max_keywords)]




