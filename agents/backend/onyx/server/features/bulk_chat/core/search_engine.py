"""
Search Engine - Sistema de Búsqueda Avanzada
===========================================

Sistema de búsqueda con índices y filtros avanzados.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Resultado de búsqueda."""
    item_id: str
    item_type: str
    score: float
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    highlights: List[str] = field(default_factory=list)


class SearchEngine:
    """Motor de búsqueda."""
    
    def __init__(self):
        self.index: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def index_item(
        self,
        item_id: str,
        item_type: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Indexar item.
        
        Args:
            item_id: ID único del item
            item_type: Tipo de item
            title: Título
            content: Contenido
            metadata: Metadatos adicionales
        """
        async with self._lock:
            self.index[item_id] = {
                "item_id": item_id,
                "item_type": item_type,
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "indexed_at": datetime.now(),
            }
        
        logger.debug(f"Indexed item: {item_id}")
    
    async def search(
        self,
        query: str,
        item_type: Optional[str] = None,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Buscar items.
        
        Args:
            query: Consulta de búsqueda
            item_type: Filtrar por tipo (opcional)
            limit: Número máximo de resultados
            filters: Filtros adicionales
        
        Returns:
            Lista de resultados
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for item_id, item_data in self.index.items():
            # Filtrar por tipo
            if item_type and item_data["item_type"] != item_type:
                continue
            
            # Filtrar por filtros
            if filters:
                match = True
                for key, value in filters.items():
                    if item_data.get("metadata", {}).get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Calcular score
            score = self._calculate_score(item_data, query_words)
            
            if score > 0:
                highlights = self._extract_highlights(item_data["content"], query_words)
                
                results.append(SearchResult(
                    item_id=item_id,
                    item_type=item_data["item_type"],
                    score=score,
                    title=item_data["title"],
                    content=item_data["content"][:500],  # Primeros 500 caracteres
                    metadata=item_data["metadata"],
                    highlights=highlights,
                ))
        
        # Ordenar por score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results[:limit]
    
    def _calculate_score(
        self,
        item_data: Dict[str, Any],
        query_words: set,
    ) -> float:
        """Calcular score de relevancia."""
        score = 0.0
        
        title = item_data["title"].lower()
        content = item_data["content"].lower()
        
        # Score por palabras en título (mayor peso)
        for word in query_words:
            if word in title:
                score += 10.0
            if word in content:
                score += 1.0
        
        # Bonus por coincidencia exacta
        query_str = " ".join(query_words)
        if query_str in title:
            score += 20.0
        if query_str in content:
            score += 5.0
        
        return score
    
    def _extract_highlights(
        self,
        content: str,
        query_words: set,
        max_highlights: int = 3,
    ) -> List[str]:
        """Extraer fragmentos destacados."""
        highlights = []
        content_lower = content.lower()
        
        for word in query_words:
            # Buscar primera ocurrencia
            idx = content_lower.find(word)
            if idx != -1:
                # Extraer contexto (50 caracteres antes y después)
                start = max(0, idx - 50)
                end = min(len(content), idx + len(word) + 50)
                highlight = content[start:end]
                
                if highlight not in highlights:
                    highlights.append(highlight)
                
                if len(highlights) >= max_highlights:
                    break
        
        return highlights
    
    async def delete_item(self, item_id: str):
        """Eliminar item del índice."""
        async with self._lock:
            if item_id in self.index:
                del self.index[item_id]
                logger.debug(f"Deleted item from index: {item_id}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del índice."""
        items_by_type = {}
        for item_data in self.index.values():
            item_type = item_data["item_type"]
            items_by_type[item_type] = items_by_type.get(item_type, 0) + 1
        
        return {
            "total_items": len(self.index),
            "items_by_type": items_by_type,
        }
































