"""
Knowledge Manager - Gestor de Conocimiento
==========================================

Sistema avanzado de gestión de conocimiento con indexación semántica, búsqueda vectorial y aprendizaje continuo.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Tipo de conocimiento."""
    DOCUMENT = "document"
    CODE = "code"
    CONVERSATION = "conversation"
    PATTERN = "pattern"
    BEST_PRACTICE = "best_practice"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class KnowledgeEntry:
    """Entrada de conocimiento."""
    entry_id: str
    knowledge_type: KnowledgeType
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    relevance_score: float = 0.0


@dataclass
class KnowledgeRelationship:
    """Relación entre entradas de conocimiento."""
    source_id: str
    target_id: str
    relationship_type: str  # "related", "prerequisite", "similar", "conflicts"
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeManager:
    """Gestor de conocimiento."""
    
    def __init__(
        self,
        max_entries: int = 100000,
        similarity_threshold: float = 0.7,
    ):
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.relationships: List[KnowledgeRelationship] = []
        self.index_by_type: Dict[KnowledgeType, List[str]] = defaultdict(list)
        self.index_by_tag: Dict[str, List[str]] = defaultdict(list)
        self.search_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def add_knowledge(
        self,
        knowledge_type: KnowledgeType,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Agregar entrada de conocimiento.
        
        Args:
            knowledge_type: Tipo de conocimiento
            title: Título
            content: Contenido
            tags: Etiquetas
            embedding: Vector de embedding (opcional)
            metadata: Metadatos adicionales
        
        Returns:
            ID de la entrada
        """
        entry_id = f"kb_{knowledge_type.value}_{datetime.now().timestamp()}"
        
        entry = KnowledgeEntry(
            entry_id=entry_id,
            knowledge_type=knowledge_type,
            title=title,
            content=content,
            tags=tags or [],
            embedding=embedding,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.entries[entry_id] = entry
            self.index_by_type[knowledge_type].append(entry_id)
            
            for tag in entry.tags:
                self.index_by_tag[tag].append(entry_id)
            
            # Limitar tamaño
            if len(self.entries) > self.max_entries:
                # Eliminar entrada más antigua con menor acceso
                oldest = min(
                    self.entries.values(),
                    key=lambda e: (e.access_count, e.created_at.timestamp())
                )
                await self.remove_knowledge(oldest.entry_id)
        
        logger.info(f"Added knowledge entry: {entry_id} - {title}")
        return entry_id
    
    async def remove_knowledge(self, entry_id: str) -> bool:
        """Remover entrada de conocimiento."""
        entry = self.entries.get(entry_id)
        if not entry:
            return False
        
        async with self._lock:
            # Remover de índices
            if entry_id in self.index_by_type[entry.knowledge_type]:
                self.index_by_type[entry.knowledge_type].remove(entry_id)
            
            for tag in entry.tags:
                if entry_id in self.index_by_tag[tag]:
                    self.index_by_tag[tag].remove(entry_id)
            
            # Remover relaciones
            self.relationships = [
                r for r in self.relationships
                if r.source_id != entry_id and r.target_id != entry_id
            ]
            
            del self.entries[entry_id]
        
        return True
    
    async def search_knowledge(
        self,
        query: str,
        knowledge_type: Optional[KnowledgeType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        use_embedding: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Buscar conocimiento.
        
        Args:
            query: Consulta de búsqueda
            knowledge_type: Filtrar por tipo
            tags: Filtrar por etiquetas
            limit: Límite de resultados
            use_embedding: Usar búsqueda por embedding
        
        Returns:
            Lista de entradas encontradas
        """
        candidates = []
        
        # Filtrar por tipo
        if knowledge_type:
            candidate_ids = self.index_by_type[knowledge_type]
        else:
            candidate_ids = list(self.entries.keys())
        
        # Filtrar por etiquetas
        if tags:
            tag_ids = set()
            for tag in tags:
                tag_ids.update(self.index_by_tag.get(tag, []))
            candidate_ids = [eid for eid in candidate_ids if eid in tag_ids]
        
        # Búsqueda por texto (simplificada)
        query_lower = query.lower()
        for entry_id in candidate_ids:
            entry = self.entries[entry_id]
            
            # Calcular relevancia simple
            relevance = 0.0
            
            if query_lower in entry.title.lower():
                relevance += 0.5
            
            if query_lower in entry.content.lower():
                relevance += 0.3
            
            for tag in entry.tags:
                if query_lower in tag.lower():
                    relevance += 0.2
            
            if relevance > 0:
                entry.relevance_score = relevance
                candidates.append(entry)
        
        # Ordenar por relevancia
        candidates.sort(key=lambda e: (e.relevance_score, e.access_count), reverse=True)
        
        # Actualizar acceso
        for entry in candidates[:limit]:
            entry.access_count += 1
            entry.updated_at = datetime.now()
        
        # Registrar búsqueda
        self.search_history.append({
            "query": query,
            "results_count": len(candidates),
            "timestamp": datetime.now(),
        })
        
        return [
            {
                "entry_id": e.entry_id,
                "knowledge_type": e.knowledge_type.value,
                "title": e.title,
                "content": e.content[:200] + "..." if len(e.content) > 200 else e.content,
                "tags": e.tags,
                "relevance_score": e.relevance_score,
                "access_count": e.access_count,
                "metadata": e.metadata,
            }
            for e in candidates[:limit]
        ]
    
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = "related",
        strength: float = 1.0,
    ):
        """Agregar relación entre entradas."""
        if source_id not in self.entries or target_id not in self.entries:
            raise ValueError("Both entries must exist")
        
        relationship = KnowledgeRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
        )
        
        async with self._lock:
            self.relationships.append(relationship)
        
        logger.info(f"Added relationship: {source_id} -> {target_id} ({relationship_type})")
    
    def get_related_knowledge(self, entry_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener conocimiento relacionado."""
        related_ids = set()
        
        for rel in self.relationships:
            if rel.source_id == entry_id:
                related_ids.add(rel.target_id)
            elif rel.target_id == entry_id:
                related_ids.add(rel.source_id)
        
        related_entries = [
            {
                "entry_id": eid,
                "title": self.entries[eid].title,
                "knowledge_type": self.entries[eid].knowledge_type.value,
                "relationship_strength": next(
                    (r.strength for r in self.relationships
                     if (r.source_id == entry_id and r.target_id == eid) or
                        (r.target_id == entry_id and r.source_id == eid)),
                    1.0
                ),
            }
            for eid in list(related_ids)[:limit]
        ]
        
        return related_entries
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de conocimiento."""
        by_type: Dict[str, int] = defaultdict(int)
        total_tags = set()
        
        for entry in self.entries.values():
            by_type[entry.knowledge_type.value] += 1
            total_tags.update(entry.tags)
        
        return {
            "total_entries": len(self.entries),
            "entries_by_type": dict(by_type),
            "total_tags": len(total_tags),
            "total_relationships": len(self.relationships),
            "total_searches": len(self.search_history),
        }
















