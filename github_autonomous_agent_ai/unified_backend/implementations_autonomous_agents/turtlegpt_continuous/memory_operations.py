"""
Memory Operations Module
========================

Operaciones centralizadas para trabajar con memoria episódica y semántica.
Proporciona una interfaz unificada para almacenar y recuperar memorias.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryOperations:
    """
    Operaciones centralizadas para memoria.
    
    Proporciona una interfaz unificada para trabajar con
    memoria episódica y semántica.
    """
    
    def __init__(
        self,
        episodic_memory,
        semantic_memory
    ):
        """
        Inicializar operaciones de memoria.
        
        Args:
            episodic_memory: Memoria episódica
            semantic_memory: Memoria semántica
        """
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
    
    def store_episodic_memory(
        self,
        content: str,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Almacenar memoria episódica.
        
        Args:
            content: Contenido de la memoria
            importance: Importancia (0.0 a 1.0)
            metadata: Metadatos adicionales
            
        Returns:
            ID de la memoria almacenada
        """
        try:
            if not self.episodic_memory:
                logger.warning("Episodic memory not available")
                return ""
            
            memory_id = self.episodic_memory.add_memory(
                content=content,
                importance=importance,
                metadata=metadata or {}
            )
            logger.debug(f"Stored episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error storing episodic memory: {e}", exc_info=True)
            return ""
    
    def store_semantic_memory(
        self,
        fact: str,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Almacenar memoria semántica.
        
        Args:
            fact: Hecho a almacenar
            importance: Importancia (0.0 a 1.0)
            metadata: Metadatos adicionales
            
        Returns:
            ID del hecho almacenado
        """
        try:
            if not self.semantic_memory:
                logger.warning("Semantic memory not available")
                return ""
            
            fact_id = self.semantic_memory.add_fact(
                fact=fact,
                importance=importance,
                metadata=metadata or {}
            )
            logger.debug(f"Stored semantic memory: {fact_id}")
            return fact_id
        except Exception as e:
            logger.error(f"Error storing semantic memory: {e}", exc_info=True)
            return ""
    
    def get_recent_episodic_memories(
        self,
        count: int = 5,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Obtener memorias episódicas recientes.
        
        Args:
            count: Número de memorias a obtener
            min_importance: Importancia mínima
            
        Returns:
            Lista de memorias
        """
        try:
            if not self.episodic_memory:
                return []
            
            memories = getattr(self.episodic_memory, 'memories', [])
            if not memories:
                return []
            
            # Filtrar por importancia y ordenar por fecha
            filtered = [
                mem for mem in memories
                if mem.get('importance', 0.0) >= min_importance
            ]
            
            # Ordenar por timestamp (más reciente primero)
            filtered.sort(
                key=lambda x: x.get('timestamp', datetime.min),
                reverse=True
            )
            
            return filtered[:count]
        except Exception as e:
            logger.error(f"Error getting recent episodic memories: {e}", exc_info=True)
            return []
    
    def get_semantic_facts(
        self,
        query: Optional[str] = None,
        count: int = 10,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Obtener hechos semánticos.
        
        Args:
            query: Consulta para filtrar (opcional)
            count: Número de hechos a obtener
            min_importance: Importancia mínima
            
        Returns:
            Lista de hechos
        """
        try:
            if not self.semantic_memory:
                return []
            
            facts = getattr(self.semantic_memory, 'facts', [])
            if not facts:
                return []
            
            # Filtrar por importancia
            filtered = [
                fact for fact in facts
                if fact.get('importance', 0.0) >= min_importance
            ]
            
            # Filtrar por consulta si se proporciona
            if query:
                query_lower = query.lower()
                filtered = [
                    fact for fact in filtered
                    if query_lower in fact.get('fact', '').lower()
                ]
            
            # Ordenar por importancia
            filtered.sort(
                key=lambda x: x.get('importance', 0.0),
                reverse=True
            )
            
            return filtered[:count]
        except Exception as e:
            logger.error(f"Error getting semantic facts: {e}", exc_info=True)
            return []
    
    def store_task_experience(
        self,
        task_description: str,
        result: Any,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Almacenar experiencia de una tarea.
        
        Args:
            task_description: Descripción de la tarea
            result: Resultado de la tarea
            success: Si la tarea fue exitosa
            metadata: Metadatos adicionales
        """
        try:
            # Almacenar como memoria episódica
            content = f"Task: {task_description}\nResult: {result}\nSuccess: {success}"
            importance = 0.8 if success else 0.5
            
            self.store_episodic_memory(
                content=content,
                importance=importance,
                metadata={
                    "type": "task_experience",
                    "success": success,
                    **(metadata or {})
                }
            )
            
            # Si fue exitosa, extraer hechos semánticos
            if success and isinstance(result, str):
                # Intentar extraer hechos del resultado
                facts = self._extract_facts_from_result(result)
                for fact in facts:
                    self.store_semantic_memory(
                        fact=fact,
                        importance=0.6
                    )
        except Exception as e:
            logger.error(f"Error storing task experience: {e}", exc_info=True)
    
    def _extract_facts_from_result(self, result: str) -> List[str]:
        """
        Extraer hechos del resultado de una tarea.
        
        Args:
            result: Resultado de la tarea
            
        Returns:
            Lista de hechos extraídos
        """
        # Implementación simple: buscar oraciones que parezcan hechos
        facts = []
        sentences = result.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                # Filtrar oraciones que parezcan hechos
                if any(keyword in sentence.lower() for keyword in ['is', 'are', 'has', 'have', 'can', 'should']):
                    facts.append(sentence)
        
        return facts[:3]  # Limitar a 3 hechos
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de memoria.
        
        Returns:
            Dict con estadísticas
        """
        stats = {
            "episodic_count": 0,
            "semantic_count": 0,
            "episodic_available": self.episodic_memory is not None,
            "semantic_available": self.semantic_memory is not None
        }
        
        try:
            if self.episodic_memory:
                memories = getattr(self.episodic_memory, 'memories', [])
                stats["episodic_count"] = len(memories)
            
            if self.semantic_memory:
                facts = getattr(self.semantic_memory, 'facts', [])
                stats["semantic_count"] = len(facts)
        except Exception as e:
            logger.warning(f"Error getting memory stats: {e}")
        
        return stats


def create_memory_operations(
    episodic_memory,
    semantic_memory
) -> MemoryOperations:
    """
    Factory function para crear MemoryOperations.
    
    Args:
        episodic_memory: Memoria episódica
        semantic_memory: Memoria semántica
        
    Returns:
        Instancia de MemoryOperations
    """
    return MemoryOperations(episodic_memory, semantic_memory)


