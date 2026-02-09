"""
Vector Index Manager
===================

Gestor especializado para índice vectorial.
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...core.base.service_base import BaseService
from ...database.models import Manual
from ...ml.optimizations.vector_index import VectorIndex
from .embedding_service_wrapper import EmbeddingServiceWrapper


class VectorIndexManager(BaseService):
    """Gestor de índice vectorial."""
    
    def __init__(
        self,
        embedding_service: EmbeddingServiceWrapper,
        use_vector_index: bool = True
    ):
        """
        Inicializar gestor.
        
        Args:
            embedding_service: Servicio de embeddings
            use_vector_index: Usar índice vectorial
        """
        super().__init__(logger_name=__name__)
        self.embedding_service = embedding_service
        self.use_vector_index = use_vector_index
        self.vector_index: Optional[VectorIndex] = None
        self._index_loaded = False
    
    async def ensure_initialized(self):
        """Asegurar que el índice esté inicializado."""
        if not self.use_vector_index:
            return
        
        if self.vector_index is None and not self._index_loaded:
            try:
                dim = self.embedding_service.get_embedding_dimension()
                self.vector_index = VectorIndex(
                    dimension=dim,
                    use_gpu=self.embedding_service.device == "cuda",
                    index_type="hnsw"
                )
                self._index_loaded = True
                self.log_info("Índice vectorial inicializado")
            except Exception as e:
                self.log_warning(f"No se pudo inicializar índice vectorial: {str(e)}")
                self.vector_index = None
    
    async def load_vectors(
        self,
        db: AsyncSession,
        limit: int = 10000
    ):
        """
        Cargar vectores de manuales al índice.
        
        Args:
            db: Sesión de base de datos
            limit: Límite de manuales a cargar
        """
        if not self.vector_index:
            return
        
        try:
            query = select(Manual).where(Manual.is_public == True).limit(limit)
            result = await db.execute(query)
            manuals = list(result.scalars().all())
            
            if not manuals:
                return
            
            texts = [
                f"{m.title or ''} {m.problem_description}".strip()
                for m in manuals
            ]
            
            embeddings = self.embedding_service.encode_batch(texts, batch_size)
            ids = [m.id for m in manuals]
            
            self.vector_index.add(embeddings, ids)
            self.log_info(f"Cargados {len(manuals)} manuales al índice vectorial")
        except Exception as e:
            self.log_error(f"Error cargando vectores: {str(e)}")
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10
    ) -> tuple:
        """
        Buscar en índice vectorial.
        
        Args:
            query_embedding: Embedding de la query
            k: Número de resultados
        
        Returns:
            Tuple de (distancias, ids)
        """
        if not self.vector_index:
            return [], []
        
        return self.vector_index.search(query_embedding, k=k)

