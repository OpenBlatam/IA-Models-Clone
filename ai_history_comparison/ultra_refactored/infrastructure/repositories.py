"""
Infrastructure Repositories - Repositorios de Infraestructura
===========================================================

Implementaciones concretas de repositorios para acceso a datos.
"""

from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime

from ..domain.models import HistoryEntry, ComparisonResult, ModelType
from ..domain.exceptions import NotFoundException, RepositoryException
from ..application.interfaces import IHistoryRepository, IComparisonRepository


class InMemoryHistoryRepository(IHistoryRepository):
    """
    Repositorio en memoria para entradas de historial.
    
    Implementación simple para desarrollo y testing.
    """
    
    def __init__(self):
        self._entries: Dict[str, HistoryEntry] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, entry: HistoryEntry) -> HistoryEntry:
        """Guardar una entrada de historial."""
        try:
            async with self._lock:
                # Actualizar timestamp si es nueva
                if entry.id not in self._entries:
                    entry.timestamp = datetime.utcnow()
                
                self._entries[entry.id] = entry
                return entry
                
        except Exception as e:
            raise RepositoryException(f"Failed to save history entry: {e}", "save")
    
    async def get_by_id(self, entry_id: str) -> Optional[HistoryEntry]:
        """Obtener entrada por ID."""
        try:
            async with self._lock:
                return self._entries.get(entry_id)
                
        except Exception as e:
            raise RepositoryException(f"Failed to get history entry {entry_id}: {e}", "get_by_id")
    
    async def list(
        self,
        user_id: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[HistoryEntry]:
        """Listar entradas con filtros."""
        try:
            async with self._lock:
                entries = list(self._entries.values())
                
                # Aplicar filtros
                if user_id:
                    entries = [e for e in entries if e.user_id == user_id]
                
                if model_type:
                    entries = [e for e in entries if e.model_type == model_type]
                
                # Ordenar por timestamp descendente
                entries.sort(key=lambda x: x.timestamp, reverse=True)
                
                # Aplicar paginación
                return entries[offset:offset + limit]
                
        except Exception as e:
            raise RepositoryException(f"Failed to list history entries: {e}", "list")
    
    async def delete(self, entry_id: str) -> bool:
        """Eliminar entrada por ID."""
        try:
            async with self._lock:
                if entry_id in self._entries:
                    del self._entries[entry_id]
                    return True
                return False
                
        except Exception as e:
            raise RepositoryException(f"Failed to delete history entry {entry_id}: {e}", "delete")
    
    async def count(
        self,
        user_id: Optional[str] = None,
        model_type: Optional[ModelType] = None
    ) -> int:
        """Contar entradas con filtros."""
        try:
            async with self._lock:
                entries = list(self._entries.values())
                
                # Aplicar filtros
                if user_id:
                    entries = [e for e in entries if e.user_id == user_id]
                
                if model_type:
                    entries = [e for e in entries if e.model_type == model_type]
                
                return len(entries)
                
        except Exception as e:
            raise RepositoryException(f"Failed to count history entries: {e}", "count")
    
    async def clear(self) -> None:
        """Limpiar todas las entradas (para testing)."""
        async with self._lock:
            self._entries.clear()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del repositorio."""
        try:
            async with self._lock:
                entries = list(self._entries.values())
                
                if not entries:
                    return {
                        "total_entries": 0,
                        "entries_by_model": {},
                        "average_quality_score": 0.0,
                        "most_common_model": None
                    }
                
                # Contar por modelo
                model_counts = {}
                quality_scores = []
                
                for entry in entries:
                    model_counts[entry.model_type] = model_counts.get(entry.model_type, 0) + 1
                    if entry.quality_score is not None:
                        quality_scores.append(entry.quality_score)
                
                # Modelo más común
                most_common_model = max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else None
                
                # Score promedio de calidad
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
                
                return {
                    "total_entries": len(entries),
                    "entries_by_model": model_counts,
                    "average_quality_score": avg_quality,
                    "most_common_model": most_common_model
                }
                
        except Exception as e:
            raise RepositoryException(f"Failed to get statistics: {e}", "get_statistics")


class InMemoryComparisonRepository(IComparisonRepository):
    """
    Repositorio en memoria para resultados de comparación.
    
    Implementación simple para desarrollo y testing.
    """
    
    def __init__(self):
        self._comparisons: Dict[str, ComparisonResult] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, comparison: ComparisonResult) -> ComparisonResult:
        """Guardar resultado de comparación."""
        try:
            async with self._lock:
                # Actualizar timestamp si es nueva
                if comparison.id not in self._comparisons:
                    comparison.timestamp = datetime.utcnow()
                
                self._comparisons[comparison.id] = comparison
                return comparison
                
        except Exception as e:
            raise RepositoryException(f"Failed to save comparison: {e}", "save")
    
    async def get_by_id(self, comparison_id: str) -> Optional[ComparisonResult]:
        """Obtener comparación por ID."""
        try:
            async with self._lock:
                return self._comparisons.get(comparison_id)
                
        except Exception as e:
            raise RepositoryException(f"Failed to get comparison {comparison_id}: {e}", "get_by_id")
    
    async def list(
        self,
        entry_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ComparisonResult]:
        """Listar comparaciones con filtros."""
        try:
            async with self._lock:
                comparisons = list(self._comparisons.values())
                
                # Aplicar filtros
                if entry_id:
                    comparisons = [
                        c for c in comparisons 
                        if c.entry_1_id == entry_id or c.entry_2_id == entry_id
                    ]
                
                # Ordenar por timestamp descendente
                comparisons.sort(key=lambda x: x.timestamp, reverse=True)
                
                # Aplicar paginación
                return comparisons[offset:offset + limit]
                
        except Exception as e:
            raise RepositoryException(f"Failed to list comparisons: {e}", "list")
    
    async def delete(self, comparison_id: str) -> bool:
        """Eliminar comparación por ID."""
        try:
            async with self._lock:
                if comparison_id in self._comparisons:
                    del self._comparisons[comparison_id]
                    return True
                return False
                
        except Exception as e:
            raise RepositoryException(f"Failed to delete comparison {comparison_id}: {e}", "delete")
    
    async def count(self, entry_id: Optional[str] = None) -> int:
        """Contar comparaciones con filtros."""
        try:
            async with self._lock:
                comparisons = list(self._comparisons.values())
                
                # Aplicar filtros
                if entry_id:
                    comparisons = [
                        c for c in comparisons 
                        if c.entry_1_id == entry_id or c.entry_2_id == entry_id
                    ]
                
                return len(comparisons)
                
        except Exception as e:
            raise RepositoryException(f"Failed to count comparisons: {e}", "count")
    
    async def clear(self) -> None:
        """Limpiar todas las comparaciones (para testing)."""
        async with self._lock:
            self._comparisons.clear()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del repositorio."""
        try:
            async with self._lock:
                comparisons = list(self._comparisons.values())
                
                if not comparisons:
                    return {
                        "total_comparisons": 0,
                        "average_similarity_score": 0.0,
                        "high_similarity_count": 0,
                        "low_similarity_count": 0
                    }
                
                # Calcular estadísticas
                similarity_scores = [c.similarity_score for c in comparisons]
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                
                high_similarity_count = len([s for s in similarity_scores if s >= 0.8])
                low_similarity_count = len([s for s in similarity_scores if s < 0.5])
                
                return {
                    "total_comparisons": len(comparisons),
                    "average_similarity_score": avg_similarity,
                    "high_similarity_count": high_similarity_count,
                    "low_similarity_count": low_similarity_count
                }
                
        except Exception as e:
            raise RepositoryException(f"Failed to get statistics: {e}", "get_statistics")
    
    async def find_similar_entries(
        self, 
        entry_id: str, 
        threshold: float = 0.7, 
        limit: int = 10
    ) -> List[ComparisonResult]:
        """Encontrar entradas similares a una entrada dada."""
        try:
            async with self._lock:
                comparisons = list(self._comparisons.values())
                
                # Filtrar comparaciones que incluyan la entrada
                relevant_comparisons = [
                    c for c in comparisons 
                    if c.entry_1_id == entry_id or c.entry_2_id == entry_id
                ]
                
                # Filtrar por threshold de similitud
                similar_comparisons = [
                    c for c in relevant_comparisons 
                    if c.similarity_score >= threshold
                ]
                
                # Ordenar por similitud descendente
                similar_comparisons.sort(key=lambda x: x.similarity_score, reverse=True)
                
                return similar_comparisons[:limit]
                
        except Exception as e:
            raise RepositoryException(f"Failed to find similar entries: {e}", "find_similar_entries")




