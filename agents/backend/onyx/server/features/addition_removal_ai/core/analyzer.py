"""
Context Analyzer - Analizador de contexto para operaciones de adición/eliminación
"""

import logging
from typing import Dict, Any, Optional, List
import re
from .cache import AnalysisCache

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """Analizador de contexto para entender el contenido antes de modificarlo"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el analizador de contexto.

        Args:
            config: Configuración opcional
        """
        self.config = config or {}
        self.cache = AnalysisCache(max_size=100, ttl_seconds=3600)

    async def analyze(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analizar el contenido y su contexto.

        Args:
            content: Contenido a analizar
            context: Contexto adicional opcional

        Returns:
            Diccionario con el análisis
        """
        # Verificar cache primero
        cached = self.cache.get_analysis(content, "analyze")
        if cached:
            logger.debug("Análisis obtenido del cache")
            if context:
                cached["context"] = context
            return cached
        
        analysis = {
            "length": len(content),
            "word_count": len(content.split()),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "sentence_count": len(re.findall(r'[.!?]+', content)),
            "structure": self._analyze_structure(content),
            "topics": self._extract_topics(content),
            "language": self._detect_language(content)
        }
        
        if context:
            analysis["context"] = context
        
        # Almacenar en cache
        self.cache.set_analysis(content, analysis, "analyze")
        
        return analysis

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analizar la estructura del contenido"""
        lines = content.split('\n')
        return {
            "has_headers": bool(re.search(r'^#+\s', content, re.MULTILINE)),
            "has_lists": bool(re.search(r'^[\*\-\+]\s', content, re.MULTILINE)),
            "has_code_blocks": bool(re.search(r'```', content)),
            "line_count": len(lines),
            "avg_line_length": sum(len(l) for l in lines) / len(lines) if lines else 0
        }

    def _extract_topics(self, content: str) -> List[str]:
        """Extraer temas principales del contenido (simplificado)"""
        # Implementación básica - puede mejorarse con NLP
        words = re.findall(r'\b\w{4,}\b', content.lower())
        from collections import Counter
        common_words = Counter(words).most_common(5)
        return [word for word, count in common_words]

    def _detect_language(self, content: str) -> str:
        """Detectar idioma del contenido (simplificado)"""
        # Implementación básica - puede mejorarse
        spanish_indicators = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no']
        english_indicators = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it']
        
        content_lower = content.lower()
        spanish_count = sum(1 for word in spanish_indicators if word in content_lower)
        english_count = sum(1 for word in english_indicators if word in content_lower)
        
        return "es" if spanish_count > english_count else "en"

    async def suggest_position(
        self,
        content: str,
        addition: str,
        analysis: Dict[str, Any]
    ) -> str:
        """
        Sugerir la mejor posición para agregar contenido.

        Args:
            content: Contenido original
            addition: Contenido a agregar
            analysis: Análisis del contenido

        Returns:
            Posición sugerida
        """
        # Lógica básica de sugerencia
        structure = analysis.get("structure", {})
        
        # Si tiene estructura de encabezados, agregar al final
        if structure.get("has_headers"):
            return "end"
        
        # Si es muy corto, agregar al final
        if analysis.get("length", 0) < 500:
            return "end"
        
        # Por defecto, agregar al final
        return "end"

    async def suggest_removal(
        self,
        content: str,
        analysis: Dict[str, Any]
    ) -> Optional[str]:
        """
        Sugerir qué eliminar del contenido.

        Args:
            content: Contenido a analizar
            analysis: Análisis del contenido

        Returns:
            Patrón sugerido para eliminar (None si no hay sugerencia)
        """
        # Implementación básica - puede mejorarse con IA
        # Por ahora, retornar None para que se especifique manualmente
        return None

