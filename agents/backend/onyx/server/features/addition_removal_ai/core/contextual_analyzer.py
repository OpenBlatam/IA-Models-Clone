"""
Contextual Analyzer - Sistema de análisis de contenido contextual
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextualAnalyzer:
    """Analizador contextual"""

    def __init__(self):
        """Inicializar analizador"""
        # Contextos comunes
        self.contexts = {
            "temporal": ["hoy", "ayer", "mañana", "semana", "mes", "año", "now", "today", "yesterday", "week", "month", "year"],
            "geographic": ["ciudad", "país", "región", "mundo", "city", "country", "region", "world"],
            "social": ["persona", "gente", "comunidad", "sociedad", "person", "people", "community", "society"],
            "technical": ["tecnología", "sistema", "aplicación", "software", "technology", "system", "application", "software"],
            "business": ["negocio", "empresa", "mercado", "cliente", "business", "company", "market", "client"],
            "educational": ["aprender", "enseñar", "estudiar", "educación", "learn", "teach", "study", "education"]
        }

    def analyze_context(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analizar contexto del contenido.

        Args:
            content: Contenido
            metadata: Metadatos adicionales (opcional)

        Returns:
            Análisis contextual
        """
        content_lower = content.lower()
        context_scores = {}
        
        # Analizar cada tipo de contexto
        for context_type, keywords in self.contexts.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            context_scores[context_type] = {
                "score": score,
                "percentage": (score / len(keywords) * 100) if keywords else 0
            }
        
        # Determinar contexto dominante
        dominant_context = max(
            context_scores.items(),
            key=lambda x: x[1]["score"]
        )[0] if context_scores else None
        
        # Análisis de referencias temporales
        temporal_references = self._extract_temporal_references(content)
        
        # Análisis de referencias geográficas
        geographic_references = self._extract_geographic_references(content)
        
        return {
            "context_scores": context_scores,
            "dominant_context": dominant_context,
            "temporal_references": temporal_references,
            "geographic_references": geographic_references,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

    def analyze_contextual_relevance(
        self,
        content: str,
        target_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analizar relevancia contextual del contenido.

        Args:
            content: Contenido
            target_context: Contexto objetivo

        Returns:
            Análisis de relevancia contextual
        """
        content_context = self.analyze_context(content)
        
        # Comparar con contexto objetivo
        relevance_scores = {}
        
        for context_type in self.contexts.keys():
            content_score = content_context["context_scores"].get(context_type, {}).get("score", 0)
            target_score = target_context.get(context_type, 0)
            
            if target_score > 0:
                relevance = min(1.0, content_score / target_score) if target_score > 0 else 0
            else:
                relevance = 1.0 if content_score == 0 else 0.5
            
            relevance_scores[context_type] = relevance
        
        # Calcular relevancia general
        overall_relevance = sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0
        
        return {
            "overall_relevance": overall_relevance,
            "relevance_by_context": relevance_scores,
            "content_context": content_context,
            "target_context": target_context,
            "is_relevant": overall_relevance > 0.7
        }

    def _extract_temporal_references(self, content: str) -> List[str]:
        """Extraer referencias temporales"""
        temporal_patterns = [
            r'\b\d{4}\b',  # Años
            r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(lunes|martes|miércoles|jueves|viernes|sábado|domingo)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        
        references = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))

    def _extract_geographic_references(self, content: str) -> List[str]:
        """Extraer referencias geográficas"""
        # Patrones básicos (en producción, usar NER más sofisticado)
        geographic_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'  # Nombres propios (simplificado)
        ]
        
        references = []
        for pattern in geographic_patterns:
            matches = re.findall(pattern, content)
            references.extend(matches)
        
        # Filtrar palabras comunes que no son lugares
        common_words = {'The', 'This', 'That', 'These', 'Those', 'El', 'La', 'Los', 'Las'}
        references = [r for r in references if r not in common_words]
        
        return list(set(references))[:10]  # Limitar a 10






