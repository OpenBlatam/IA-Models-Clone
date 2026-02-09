"""
Motor de Búsqueda Avanzado

Proporciona:
- Búsqueda full-text
- Búsqueda por similitud
- Filtros avanzados
- Ordenamiento personalizado
- Búsqueda fuzzy
- Autocompletado
"""

import logging
import re
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import difflib

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Resultado de búsqueda"""
    id: str
    score: float
    data: Dict[str, Any]
    highlights: List[str] = field(default_factory=list)


@dataclass
class SearchIndex:
    """Índice de búsqueda"""
    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inverted_index: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    metadata_index: Dict[str, Dict[str, set]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))


class AdvancedSearchEngine:
    """Motor de búsqueda avanzado"""
    
    def __init__(self):
        self.index = SearchIndex()
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were"
        }
        logger.info("AdvancedSearchEngine initialized")
    
    def index_document(
        self,
        doc_id: str,
        content: Dict[str, Any],
        text_fields: List[str] = None
    ):
        """
        Indexa un documento
        
        Args:
            doc_id: ID del documento
            content: Contenido del documento
            text_fields: Campos de texto a indexar
        """
        if text_fields is None:
            text_fields = ["title", "description", "prompt", "tags"]
        
        # Almacenar documento
        self.index.documents[doc_id] = content
        
        # Indexar campos de texto
        for field in text_fields:
            if field in content:
                text = str(content[field]).lower()
                tokens = self._tokenize(text)
                
                for token in tokens:
                    self.index.inverted_index[token].add(doc_id)
        
        # Indexar metadatos
        for key, value in content.items():
            if key not in text_fields and value:
                if isinstance(value, list):
                    for item in value:
                        self.index.metadata_index[key][str(item).lower()].add(doc_id)
                else:
                    self.index.metadata_index[key][str(value).lower()].add(doc_id)
    
    def remove_document(self, doc_id: str):
        """Elimina un documento del índice"""
        if doc_id in self.index.documents:
            del self.index.documents[doc_id]
            
            # Limpiar índices invertidos
            for token, doc_set in self.index.inverted_index.items():
                doc_set.discard(doc_id)
            
            # Limpiar índices de metadatos
            for field_index in self.index.metadata_index.values():
                for value_set in field_index.values():
                    value_set.discard(doc_id)
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: Optional[str] = None,
        fuzzy: bool = False
    ) -> List[SearchResult]:
        """
        Busca documentos
        
        Args:
            query: Consulta de búsqueda
            filters: Filtros adicionales
            limit: Límite de resultados
            offset: Offset de resultados
            sort_by: Campo para ordenar
            fuzzy: Usar búsqueda fuzzy
        
        Returns:
            Lista de resultados ordenados por relevancia
        """
        query_tokens = self._tokenize(query.lower())
        
        # Obtener documentos candidatos
        candidate_docs = set()
        
        if fuzzy:
            # Búsqueda fuzzy
            for token in query_tokens:
                # Buscar tokens similares
                similar_tokens = self._find_similar_tokens(token)
                for similar_token in similar_tokens:
                    candidate_docs.update(self.index.inverted_index.get(similar_token, set()))
        else:
            # Búsqueda exacta
            for token in query_tokens:
                candidate_docs.update(self.index.inverted_index.get(token, set()))
        
        # Aplicar filtros
        if filters:
            filtered_docs = self._apply_filters(candidate_docs, filters)
        else:
            filtered_docs = candidate_docs
        
        # Calcular scores
        results = []
        for doc_id in filtered_docs:
            if doc_id not in self.index.documents:
                continue
            
            doc = self.index.documents[doc_id]
            score = self._calculate_score(query_tokens, doc, query)
            highlights = self._extract_highlights(query_tokens, doc)
            
            results.append(SearchResult(
                id=doc_id,
                score=score,
                data=doc,
                highlights=highlights
            ))
        
        # Ordenar
        if sort_by:
            results.sort(key=lambda x: x.data.get(sort_by, 0), reverse=True)
        else:
            results.sort(key=lambda x: x.score, reverse=True)
        
        # Paginación
        return results[offset:offset + limit]
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[str]:
        """
        Autocompletado de búsqueda
        
        Args:
            prefix: Prefijo de búsqueda
            limit: Límite de sugerencias
        
        Returns:
            Lista de sugerencias
        """
        prefix_lower = prefix.lower()
        suggestions = set()
        
        # Buscar en tokens del índice
        for token in self.index.inverted_index.keys():
            if token.startswith(prefix_lower):
                suggestions.add(token)
                if len(suggestions) >= limit * 2:  # Obtener más para filtrar
                    break
        
        # Buscar en documentos
        for doc in self.index.documents.values():
            for field in ["title", "description", "prompt"]:
                if field in doc:
                    text = str(doc[field]).lower()
                    words = self._tokenize(text)
                    for word in words:
                        if word.startswith(prefix_lower):
                            suggestions.add(word)
        
        return sorted(list(suggestions))[:limit]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokeniza un texto"""
        # Remover caracteres especiales y dividir
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Filtrar stop words y tokens muy cortos
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        return tokens
    
    def _find_similar_tokens(self, token: str, threshold: float = 0.8) -> List[str]:
        """Encuentra tokens similares usando fuzzy matching"""
        similar = []
        for indexed_token in self.index.inverted_index.keys():
            ratio = difflib.SequenceMatcher(None, token, indexed_token).ratio()
            if ratio >= threshold:
                similar.append(indexed_token)
        return similar
    
    def _apply_filters(
        self,
        doc_ids: set,
        filters: Dict[str, Any]
    ) -> set:
        """Aplica filtros a los documentos"""
        filtered = doc_ids.copy()
        
        for field, value in filters.items():
            if field in self.index.metadata_index:
                if isinstance(value, list):
                    # OR de múltiples valores
                    matching_docs = set()
                    for v in value:
                        matching_docs.update(
                            self.index.metadata_index[field].get(str(v).lower(), set())
                        )
                    filtered &= matching_docs
                else:
                    # Valor único
                    matching_docs = self.index.metadata_index[field].get(
                        str(value).lower(), set()
                    )
                    filtered &= matching_docs
            else:
                # Filtro directo en documentos
                matching_docs = {
                    doc_id for doc_id in filtered
                    if doc_id in self.index.documents
                    and self._matches_filter(self.index.documents[doc_id], field, value)
                }
                filtered = matching_docs
        
        return filtered
    
    def _matches_filter(self, doc: Dict[str, Any], field: str, value: Any) -> bool:
        """Verifica si un documento coincide con un filtro"""
        if field not in doc:
            return False
        
        doc_value = doc[field]
        
        if isinstance(value, dict):
            # Operadores: gt, gte, lt, lte, ne
            if "gt" in value and not (doc_value > value["gt"]):
                return False
            if "gte" in value and not (doc_value >= value["gte"]):
                return False
            if "lt" in value and not (doc_value < value["lt"]):
                return False
            if "lte" in value and not (doc_value <= value["lte"]):
                return False
            if "ne" in value and doc_value == value["ne"]:
                return False
            return True
        
        if isinstance(value, list):
            return doc_value in value
        
        return doc_value == value
    
    def _calculate_score(
        self,
        query_tokens: List[str],
        doc: Dict[str, Any],
        original_query: str
    ) -> float:
        """Calcula el score de relevancia"""
        score = 0.0
        
        # Contar matches de tokens
        text_content = " ".join([
            str(doc.get(field, ""))
            for field in ["title", "description", "prompt", "tags"]
        ]).lower()
        
        for token in query_tokens:
            # Frecuencia del token en el documento
            count = text_content.count(token)
            score += count * 1.0
            
            # Bonus si está en el título
            if "title" in doc and token in str(doc["title"]).lower():
                score += 2.0
        
        # Bonus por match exacto de la query
        if original_query.lower() in text_content:
            score += 5.0
        
        return score
    
    def _extract_highlights(
        self,
        query_tokens: List[str],
        doc: Dict[str, Any]
    ) -> List[str]:
        """Extrae fragmentos destacados"""
        highlights = []
        
        for field in ["title", "description", "prompt"]:
            if field in doc:
                text = str(doc[field])
                for token in query_tokens:
                    if token in text.lower():
                        # Encontrar contexto alrededor del token
                        idx = text.lower().find(token)
                        start = max(0, idx - 30)
                        end = min(len(text), idx + len(token) + 30)
                        highlight = text[start:end].strip()
                        if highlight not in highlights:
                            highlights.append(highlight)
        
        return highlights[:3]  # Máximo 3 highlights
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice"""
        return {
            "total_documents": len(self.index.documents),
            "total_tokens": len(self.index.inverted_index),
            "indexed_fields": list(self.index.metadata_index.keys())
        }


# Instancia global
_search_engine: Optional[AdvancedSearchEngine] = None


def get_search_engine() -> AdvancedSearchEngine:
    """Obtiene la instancia global del motor de búsqueda"""
    global _search_engine
    if _search_engine is None:
        _search_engine = AdvancedSearchEngine()
    return _search_engine

