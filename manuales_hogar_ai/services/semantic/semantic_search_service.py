"""
Semantic Search Service
=======================

Servicio principal para búsqueda semántica.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...core.base.service_base import BaseService
from ...database.models import Manual
from .vector_index_manager import VectorIndexManager
from .embedding_service_wrapper import EmbeddingServiceWrapper


class SemanticSearchService(BaseService):
    """Servicio de búsqueda semántica."""
    
    def __init__(self, db: AsyncSession, use_vector_index: bool = True):
        """
        Inicializar servicio.
        
        Args:
            db: Sesión de base de datos
            use_vector_index: Usar índice vectorial
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.embedding_service = EmbeddingServiceWrapper()
        self.vector_index_manager = VectorIndexManager(
            self.embedding_service,
            use_vector_index
        )
    
    async def search_semantic(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.5,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica de manuales.
        
        Args:
            query: Query de búsqueda
            limit: Límite de resultados
            threshold: Umbral de similitud
            category: Filtrar por categoría
        
        Returns:
            Lista de manuales con scores
        """
        try:
            await self.vector_index_manager.ensure_initialized()
            
            if self.vector_index_manager.vector_index and self.vector_index_manager._index_loaded:
                return await self._search_with_index(query, limit, threshold, category)
            
            return await self._search_traditional(query, limit, threshold, category)
        
        except Exception as e:
            self.log_error(f"Error en búsqueda semántica: {str(e)}")
            return []
    
    async def _search_with_index(
        self,
        query: str,
        limit: int,
        threshold: float,
        category: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Búsqueda usando índice vectorial."""
        try:
            query_embedding = self.embedding_service.encode(query)
            distances, ids = self.vector_index_manager.search(query_embedding, k=limit * 2)
            
            query_db = select(Manual).where(Manual.id.in_(ids.tolist()))
            if category:
                query_db = query_db.where(Manual.category == category)
            
            result = await self.db.execute(query_db)
            manuals_dict = {m.id: m for m in result.scalars().all()}
            
            formatted_results = []
            for dist, manual_id in zip(distances, ids):
                if manual_id in manuals_dict:
                    similarity = max(0.0, 1.0 - float(dist))
                    if similarity >= threshold:
                        formatted_results.append({
                            "manual": manuals_dict[manual_id],
                            "similarity": similarity,
                            "score": similarity
                        })
                        if len(formatted_results) >= limit:
                            break
            
            return formatted_results
        
        except Exception as e:
            self.log_error(f"Error en búsqueda con índice: {str(e)}")
            return []
    
    async def _search_traditional(
        self,
        query: str,
        limit: int,
        threshold: float,
        category: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Búsqueda tradicional (fallback)."""
        try:
            query_db = select(Manual).where(Manual.is_public == True)
            if category:
                query_db = query_db.where(Manual.category == category)
            
            result = await self.db.execute(query_db)
            manuals = list(result.scalars().all())
            
            if not manuals:
                return []
            
            texts = [
                f"{m.title or ''} {m.problem_description}".strip()
                for m in manuals
            ]
            
            results = self.embedding_service.find_similar(
                query=query,
                texts=texts,
                top_k=limit,
                threshold=threshold
            )
            
            formatted_results = []
            for idx, text, similarity in results:
                manual = manuals[idx]
                formatted_results.append({
                    "manual": manual,
                    "similarity": similarity,
                    "score": float(similarity)
                })
            
            return formatted_results
        
        except Exception as e:
            self.log_error(f"Error en búsqueda tradicional: {str(e)}")
            return []
    
    async def get_related_manuals(
        self,
        manual_id: int,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Obtener manuales relacionados.
        
        Args:
            manual_id: ID del manual
            limit: Número de resultados
        
        Returns:
            Lista de manuales relacionados
        """
        try:
            query = select(Manual).where(Manual.id == manual_id)
            result = await self.db.execute(query)
            reference_manual = result.scalar_one_or_none()
            
            if not reference_manual:
                return []
            
            query_text = f"{reference_manual.title or ''} {reference_manual.problem_description}".strip()
            
            query_db = select(Manual).where(
                Manual.id != manual_id,
                Manual.is_public == True
            )
            
            result = await self.db.execute(query_db)
            manuals = list(result.scalars().all())
            
            if not manuals:
                return []
            
            texts = [
                f"{m.title or ''} {m.problem_description}".strip()
                for m in manuals
            ]
            
            results = self.embedding_service.find_similar(
                query=query_text,
                texts=texts,
                top_k=limit,
                threshold=0.3
            )
            
            formatted_results = []
            for idx, text, similarity in results:
                manual = manuals[idx]
                formatted_results.append({
                    "manual": manual,
                    "similarity": similarity
                })
            
            return formatted_results
        
        except Exception as e:
            self.log_error(f"Error obteniendo manuales relacionados: {str(e)}")
            return []

