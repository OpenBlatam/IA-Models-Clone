"""
Smart Search - Sistema de búsqueda avanzada en papers
======================================================
"""

import logging
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)


class SmartSearch:
    """
    Sistema de búsqueda avanzada con múltiples estrategias.
    """
    
    def __init__(self, vector_store=None):
        """
        Inicializar búsqueda inteligente.
        
        Args:
            vector_store: Instancia de VectorStore (opcional)
        """
        self.vector_store = vector_store
    
    def search(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda avanzada con múltiples estrategias.
        
        Args:
            query: Consulta de búsqueda
            search_type: Tipo de búsqueda (semantic, keyword, hybrid)
            limit: Límite de resultados
            
        Returns:
            Lista de papers relevantes
        """
        try:
            if search_type == "semantic":
                return self._semantic_search(query, limit)
            elif search_type == "keyword":
                return self._keyword_search(query, limit)
            elif search_type == "hybrid":
                return self._hybrid_search(query, limit)
            else:
                logger.warning(f"Tipo de búsqueda desconocido: {search_type}")
                return self._keyword_search(query, limit)
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Búsqueda semántica usando vector store"""
        if not self.vector_store:
            logger.warning("Vector store no disponible para búsqueda semántica")
            return []
        
        return self.vector_store.search_relevant_papers(query, top_k=limit)
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Búsqueda por keywords"""
        # En producción, esto buscaría en PaperStorage
        # Por ahora, retorna estructura básica
        keywords = query.lower().split()
        
        # Simulación de búsqueda por keywords
        results = []
        
        # Aquí se integraría con PaperStorage para búsqueda real
        # results = paper_storage.search_by_keywords(keywords)
        
        return results[:limit]
    
    def _hybrid_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Búsqueda híbrida (semántica + keywords)"""
        semantic_results = self._semantic_search(query, limit * 2)
        keyword_results = self._keyword_search(query, limit * 2)
        
        # Combinar y deduplicar
        combined = {}
        
        # Agregar resultados semánticos con peso
        for i, result in enumerate(semantic_results):
            paper_id = result.get("paper_id", f"sem_{i}")
            if paper_id not in combined:
                combined[paper_id] = {
                    **result,
                    "score": result.get("distance", 1.0) * 0.7  # Peso semántico
                }
        
        # Agregar resultados de keywords con peso
        for i, result in enumerate(keyword_results):
            paper_id = result.get("paper_id", f"kw_{i}")
            if paper_id not in combined:
                combined[paper_id] = {
                    **result,
                    "score": 0.3  # Peso keywords
                }
            else:
                # Mejorar score si está en ambos
                combined[paper_id]["score"] += 0.3
        
        # Ordenar por score (menor es mejor para distancia)
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x.get("score", 1.0)
        )
        
        return sorted_results[:limit]
    
    def search_by_code_context(
        self,
        code: str,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca papers relevantes basándose en contexto del código.
        
        Args:
            code: Código a analizar
            language: Lenguaje de programación
            
        Returns:
            Papers relevantes
        """
        # Extraer conceptos del código
        concepts = self._extract_concepts_from_code(code, language)
        
        # Construir query desde conceptos
        query = " ".join(concepts)
        
        # Búsqueda híbrida
        return self._hybrid_search(query, limit=5)
    
    def _extract_concepts_from_code(self, code: str, language: Optional[str]) -> List[str]:
        """Extrae conceptos del código"""
        concepts = []
        
        # Conceptos comunes
        if "class" in code:
            concepts.append("object-oriented programming")
        if "async" in code or "await" in code:
            concepts.append("asynchronous programming")
        if "def" in code or "function" in code:
            concepts.append("function optimization")
        if "import" in code or "require" in code:
            concepts.append("dependency management")
        
        # Conceptos específicos por lenguaje
        if language == "python":
            if "@" in code:
                concepts.append("python decorators")
            if "yield" in code:
                concepts.append("python generators")
        elif language in ["javascript", "typescript"]:
            if "=>" in code:
                concepts.append("arrow functions")
            if "Promise" in code:
                concepts.append("promises async")
        
        return concepts




