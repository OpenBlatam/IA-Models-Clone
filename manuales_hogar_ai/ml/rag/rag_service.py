"""
RAG Service
===========

Retrieval Augmented Generation para mejorar generación con contexto.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..embeddings.embedding_service import EmbeddingService
from ..optimizations.vector_index import VectorIndex
from ...database.models import Manual
from ...services.semantic_search_service import SemanticSearchService

logger = logging.getLogger(__name__)


class RAGService:
    """Servicio RAG para generación mejorada con contexto."""
    
    def __init__(
        self,
        db: AsyncSession,
        embedding_service: EmbeddingService,
        semantic_search: SemanticSearchService
    ):
        """
        Inicializar servicio RAG.
        
        Args:
            db: Sesión de base de datos
            embedding_service: Servicio de embeddings
            semantic_search: Servicio de búsqueda semántica
        """
        self.db = db
        self.embedding_service = embedding_service
        self.semantic_search = semantic_search
        self._logger = logger
    
    async def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recuperar contexto relevante.
        
        Args:
            query: Query de búsqueda
            top_k: Número de documentos a recuperar
            category: Filtrar por categoría
        
        Returns:
            Lista de documentos relevantes
        """
        try:
            # Buscar manuales similares
            results = await self.semantic_search.search_semantic(
                query=query,
                limit=top_k,
                threshold=0.3,
                category=category
            )
            
            # Formatear contexto
            context = []
            for result in results:
                manual = result["manual"]
                context.append({
                    "title": manual.title,
                    "problem": manual.problem_description,
                    "content": manual.manual_content[:500] if manual.manual_content else "",  # Limitar tamaño
                    "category": manual.category,
                    "similarity": result["similarity"]
                })
            
            return context
        
        except Exception as e:
            self._logger.error(f"Error recuperando contexto: {str(e)}")
            return []
    
    def build_rag_prompt(
        self,
        problem_description: str,
        context: List[Dict[str, Any]],
        category: str = "general"
    ) -> str:
        """
        Construir prompt con contexto RAG.
        
        Args:
            problem_description: Descripción del problema
            context: Contexto recuperado
            category: Categoría
        
        Returns:
            Prompt mejorado
        """
        category_names = {
            "plomeria": "Plomería",
            "techos": "Reparación de Techos",
            "carpinteria": "Carpintería",
            "electricidad": "Electricidad",
            "albanileria": "Albañilería",
            "pintura": "Pintura",
            "herreria": "Herrería",
            "jardineria": "Jardinería",
            "general": "Reparación General"
        }
        
        category_name = category_names.get(category, "Reparación General")
        
        # Construir contexto
        context_text = ""
        if context:
            context_text = "\n\nCONTEXTO DE MANUALES SIMILARES:\n"
            for i, ctx in enumerate(context, 1):
                context_text += f"\n--- Manual Similar {i} ---\n"
                context_text += f"Problema: {ctx['problem']}\n"
                context_text += f"Solución: {ctx['content']}\n"
        
        prompt = f"""Genera un manual paso a paso tipo LEGO para {category_name}.

PROBLEMA ACTUAL:
{problem_description}
{context_text}

INSTRUCCIONES:
- Usa el contexto de manuales similares como referencia
- Adapta las soluciones al problema específico
- Mantén el formato paso a paso tipo LEGO
- Incluye advertencias de seguridad cuando sea necesario

MANUAL:
"""
        return prompt
    
    async def generate_with_rag(
        self,
        problem_description: str,
        category: str = "general",
        model_generator=None,
        top_k: int = 3
    ) -> str:
        """
        Generar manual usando RAG.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría
            model_generator: Generador de modelo
            top_k: Número de documentos a recuperar
        
        Returns:
            Manual generado
        """
        try:
            # Recuperar contexto
            context = await self.retrieve_context(
                query=problem_description,
                top_k=top_k,
                category=category
            )
            
            # Construir prompt con contexto
            prompt = self.build_rag_prompt(
                problem_description=problem_description,
                context=context,
                category=category
            )
            
            # Generar con modelo
            if model_generator:
                manual = model_generator.generate(prompt)
            else:
                # Fallback: retornar prompt (para usar con OpenRouter)
                manual = prompt
            
            return manual
        
        except Exception as e:
            self._logger.error(f"Error en generación RAG: {str(e)}")
            raise




