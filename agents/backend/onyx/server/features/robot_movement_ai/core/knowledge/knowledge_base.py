"""
Knowledge Base System
====================

Sistema de base de conocimientos.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """Entrada de conocimiento."""
    entry_id: str
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeBase:
    """
    Base de conocimientos.
    
    Almacena y gestiona conocimiento del sistema.
    """
    
    def __init__(self, storage_path: str = "data/knowledge_base.json"):
        """
        Inicializar base de conocimientos.
        
        Args:
            storage_path: Ruta de almacenamiento
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: Dict[str, KnowledgeEntry] = {}
        self._load_knowledge()
    
    def _load_knowledge(self) -> None:
        """Cargar conocimiento desde archivo."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = KnowledgeEntry(**entry_data)
                        self.entries[entry.entry_id] = entry
                logger.info(f"Loaded {len(self.entries)} knowledge entries")
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
    
    def _save_knowledge(self) -> None:
        """Guardar conocimiento en archivo."""
        try:
            data = {
                "entries": [
                    {
                        "entry_id": e.entry_id,
                        "title": e.title,
                        "content": e.content,
                        "category": e.category,
                        "tags": e.tags,
                        "metadata": e.metadata,
                        "created_at": e.created_at,
                        "updated_at": e.updated_at
                    }
                    for e in self.entries.values()
                ]
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    def add_entry(
        self,
        entry_id: str,
        title: str,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeEntry:
        """
        Agregar entrada de conocimiento.
        
        Args:
            entry_id: ID único de la entrada
            title: Título
            content: Contenido
            category: Categoría
            tags: Tags
            metadata: Metadata
            
        Returns:
            Entrada creada
        """
        entry = KnowledgeEntry(
            entry_id=entry_id,
            title=title,
            content=content,
            category=category,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.entries[entry_id] = entry
        self._save_knowledge()
        logger.info(f"Added knowledge entry: {title} ({entry_id})")
        
        return entry
    
    def update_entry(
        self,
        entry_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[KnowledgeEntry]:
        """
        Actualizar entrada de conocimiento.
        
        Args:
            entry_id: ID de la entrada
            title: Nuevo título (opcional)
            content: Nuevo contenido (opcional)
            category: Nueva categoría (opcional)
            tags: Nuevos tags (opcional)
            metadata: Nueva metadata (opcional)
            
        Returns:
            Entrada actualizada o None
        """
        if entry_id not in self.entries:
            return None
        
        entry = self.entries[entry_id]
        
        if title is not None:
            entry.title = title
        if content is not None:
            entry.content = content
        if category is not None:
            entry.category = category
        if tags is not None:
            entry.tags = tags
        if metadata is not None:
            entry.metadata.update(metadata)
        
        entry.updated_at = datetime.now().isoformat()
        self._save_knowledge()
        
        return entry
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[KnowledgeEntry]:
        """
        Buscar en base de conocimientos.
        
        Args:
            query: Consulta de búsqueda
            category: Filtrar por categoría
            tags: Filtrar por tags
            limit: Límite de resultados
            
        Returns:
            Lista de entradas encontradas
        """
        results = []
        query_lower = query.lower()
        
        for entry in self.entries.values():
            # Filtrar por categoría
            if category and entry.category != category:
                continue
            
            # Filtrar por tags
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Buscar en título y contenido
            if (query_lower in entry.title.lower() or
                query_lower in entry.content.lower() or
                any(query_lower in tag.lower() for tag in entry.tags)):
                results.append(entry)
        
        return results[:limit]
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Obtener entrada por ID."""
        return self.entries.get(entry_id)
    
    def list_entries(
        self,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[KnowledgeEntry]:
        """
        Listar entradas.
        
        Args:
            category: Filtrar por categoría
            limit: Límite de resultados
            
        Returns:
            Lista de entradas
        """
        entries = list(self.entries.values())
        
        if category:
            entries = [e for e in entries if e.category == category]
        
        return entries[:limit]
    
    def delete_entry(self, entry_id: str) -> bool:
        """Eliminar entrada."""
        if entry_id in self.entries:
            del self.entries[entry_id]
            self._save_knowledge()
            return True
        return False


# Instancia global
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(storage_path: str = "data/knowledge_base.json") -> KnowledgeBase:
    """Obtener instancia global de la base de conocimientos."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(storage_path=storage_path)
    return _knowledge_base






