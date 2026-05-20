"""
Memory Context Builder Module
=============================

Constructor de contexto de memoria para prompts y operaciones del agente.
Integra memoria episódica y semántica para proporcionar contexto relevante.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MemoryContextBuilder:
    """
    Constructor de contexto de memoria.
    
    Combina información de memoria episódica y semántica
    para crear contexto relevante para operaciones del agente.
    """
    
    def __init__(
        self,
        episodic_memory,
        semantic_memory,
        max_episodic_items: int = 5,
        max_semantic_items: int = 3,
        relevance_threshold: float = 0.5
    ):
        """
        Inicializar constructor de contexto.
        
        Args:
            episodic_memory: Memoria episódica
            semantic_memory: Memoria semántica
            max_episodic_items: Máximo de items episódicos a incluir
            max_semantic_items: Máximo de items semánticos a incluir
            relevance_threshold: Umbral de relevancia para filtrar
        """
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.max_episodic_items = max_episodic_items
        self.max_semantic_items = max_semantic_items
        self.relevance_threshold = relevance_threshold
    
    def build_context_for_task(
        self,
        task_description: str,
        include_recent: bool = True,
        include_similar: bool = True,
        include_insights: bool = True
    ) -> str:
        """
        Construir contexto de memoria para una tarea.
        
        Args:
            task_description: Descripción de la tarea
            include_recent: Incluir experiencias recientes
            include_similar: Incluir experiencias similares
            include_insights: Incluir insights relevantes
            
        Returns:
            Contexto formateado como string
        """
        context_parts = []
        
        # Experiencias recientes
        if include_recent:
            recent = self._get_recent_experiences()
            if recent:
                context_parts.append("Experiencias recientes:")
                for exp in recent[:self.max_episodic_items]:
                    context_parts.append(f"  - {exp.get('content', '')[:200]}")
        
        # Experiencias similares
        if include_similar:
            similar = self._get_similar_experiences(task_description)
            if similar:
                context_parts.append("\nExperiencias similares:")
                for exp in similar[:self.max_episodic_items]:
                    context_parts.append(f"  - {exp.get('content', '')[:200]}")
        
        # Insights relevantes
        if include_insights:
            insights = self._get_relevant_insights(task_description)
            if insights:
                context_parts.append("\nInsights relevantes:")
                for insight in insights[:self.max_semantic_items]:
                    context_parts.append(f"  - {insight}")
        
        return "\n".join(context_parts) if context_parts else "No hay contexto de memoria relevante."
    
    def _get_recent_experiences(self) -> List[Dict[str, Any]]:
        """Obtener experiencias recientes de memoria episódica."""
        if not self.episodic_memory:
            return []
        
        try:
            memories = getattr(self.episodic_memory, 'memories', [])
            if not memories:
                return []
            
            # Ordenar por timestamp (más reciente primero)
            sorted_memories = sorted(
                memories,
                key=lambda m: m.get('timestamp', datetime.min),
                reverse=True
            )
            
            return sorted_memories[:self.max_episodic_items]
        except Exception as e:
            logger.warning(f"Error getting recent experiences: {e}")
            return []
    
    def _get_similar_experiences(self, task_description: str) -> List[Dict[str, Any]]:
        """Obtener experiencias similares a la tarea."""
        if not self.episodic_memory:
            return []
        
        try:
            memories = getattr(self.episodic_memory, 'memories', [])
            if not memories:
                return []
            
            # Filtrar por relevancia (simplificado - buscar palabras clave)
            task_keywords = set(task_description.lower().split())
            similar = []
            
            for memory in memories:
                content = memory.get('content', '').lower()
                content_keywords = set(content.split())
                
                # Calcular similitud simple (Jaccard)
                if task_keywords and content_keywords:
                    intersection = len(task_keywords & content_keywords)
                    union = len(task_keywords | content_keywords)
                    similarity = intersection / union if union > 0 else 0.0
                    
                    if similarity >= self.relevance_threshold:
                        similar.append((memory, similarity))
            
            # Ordenar por similitud
            similar.sort(key=lambda x: x[1], reverse=True)
            return [mem for mem, _ in similar[:self.max_episodic_items]]
        
        except Exception as e:
            logger.warning(f"Error getting similar experiences: {e}")
            return []
    
    def _get_relevant_insights(self, task_description: str) -> List[str]:
        """Obtener insights relevantes de memoria semántica."""
        if not self.semantic_memory:
            return []
        
        try:
            # Buscar en memoria semántica
            facts = getattr(self.semantic_memory, 'facts', [])
            if not facts:
                return []
            
            # Filtrar por relevancia
            task_keywords = set(task_description.lower().split())
            relevant = []
            
            for fact in facts:
                fact_text = str(fact).lower()
                fact_keywords = set(fact_text.split())
                
                if task_keywords and fact_keywords:
                    intersection = len(task_keywords & fact_keywords)
                    if intersection > 0:
                        relevant.append(str(fact))
            
            return relevant[:self.max_semantic_items]
        
        except Exception as e:
            logger.warning(f"Error getting relevant insights: {e}")
            return []
    
    def build_summary_context(self, days: int = 7) -> str:
        """
        Construir resumen de contexto de los últimos días.
        
        Args:
            days: Número de días a considerar
            
        Returns:
            Resumen formateado
        """
        if not self.episodic_memory:
            return "No hay memoria disponible."
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            memories = getattr(self.episodic_memory, 'memories', [])
            
            recent_memories = [
                m for m in memories
                if m.get('timestamp', datetime.min) >= cutoff_date
            ]
            
            if not recent_memories:
                return f"No hay experiencias en los últimos {days} días."
            
            summary = f"Resumen de los últimos {days} días ({len(recent_memories)} experiencias):\n"
            
            # Agrupar por tipo o importancia
            important = [m for m in recent_memories if m.get('importance', 0) >= 0.8]
            if important:
                summary += f"\nExperiencias importantes ({len(important)}):\n"
                for mem in important[:5]:
                    summary += f"  - {mem.get('content', '')[:150]}\n"
            
            return summary
        
        except Exception as e:
            logger.warning(f"Error building summary context: {e}")
            return "Error construyendo resumen de contexto."


def create_memory_context_builder(
    episodic_memory,
    semantic_memory,
    config: Optional[Dict[str, Any]] = None
) -> MemoryContextBuilder:
    """
    Factory function para crear MemoryContextBuilder.
    
    Args:
        episodic_memory: Memoria episódica
        semantic_memory: Memoria semántica
        config: Configuración opcional
        
    Returns:
        MemoryContextBuilder configurado
    """
    max_episodic = config.get("max_episodic_context_items", 5) if config else 5
    max_semantic = config.get("max_semantic_context_items", 3) if config else 3
    threshold = config.get("relevance_threshold", 0.5) if config else 0.5
    
    return MemoryContextBuilder(
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        max_episodic_items=max_episodic,
        max_semantic_items=max_semantic,
        relevance_threshold=threshold
    )
