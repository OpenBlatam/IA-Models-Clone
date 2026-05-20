"""
Semantic Search - Sistema de búsqueda semántica
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
import math

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Motor de búsqueda semántica"""

    def __init__(self):
        """Inicializar motor de búsqueda"""
        self.index: Dict[str, Dict[str, float]] = {}  # Índice invertido
        self.documents: Dict[str, str] = {}  # Almacenamiento de documentos

    def index_document(self, doc_id: str, content: str):
        """
        Indexar un documento.

        Args:
            doc_id: ID del documento
            content: Contenido del documento
        """
        self.documents[doc_id] = content
        
        # Tokenizar y normalizar
        tokens = self._tokenize(content)
        
        # Calcular TF (Term Frequency)
        tf = Counter(tokens)
        total_terms = len(tokens)
        
        # Almacenar en índice
        for term, count in tf.items():
            if term not in self.index:
                self.index[term] = {}
            self.index[term][doc_id] = count / total_terms
        
        logger.debug(f"Documento indexado: {doc_id}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos relevantes.

        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados
            threshold: Umbral de relevancia

        Returns:
            Lista de resultados ordenados por relevancia
        """
        # Tokenizar consulta
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calcular TF-IDF para la consulta
        query_tf = Counter(query_tokens)
        query_terms = len(query_tokens)
        
        # Calcular scores de relevancia
        scores: Dict[str, float] = {}
        
        for term, count in query_tf.items():
            if term in self.index:
                # TF para la consulta
                query_tf_score = count / query_terms
                
                # IDF para el término
                idf = math.log(len(self.documents) / len(self.index[term]))
                
                # Peso del término en la consulta
                query_weight = query_tf_score * idf
                
                # Calcular similitud con cada documento
                for doc_id, doc_tf in self.index[term].items():
                    if doc_id not in scores:
                        scores[doc_id] = 0.0
                    
                    # Similitud coseno simplificada
                    scores[doc_id] += query_weight * doc_tf
        
        # Normalizar scores
        for doc_id in scores:
            doc_length = len(self.documents[doc_id].split())
            scores[doc_id] = scores[doc_id] / math.sqrt(doc_length) if doc_length > 0 else 0.0
        
        # Filtrar y ordenar
        results = [
            {
                "doc_id": doc_id,
                "score": score,
                "content": self.documents[doc_id],
                "snippet": self._generate_snippet(self.documents[doc_id], query)
            }
            for doc_id, score in scores.items()
            if score >= threshold
        ]
        
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizar texto.

        Args:
            text: Texto a tokenizar

        Returns:
            Lista de tokens
        """
        # Convertir a minúsculas y tokenizar
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filtrar stop words básicas
        stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'en', 'a', 'y', 'o', 'pero', 'si', 'no',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        return tokens

    def _generate_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """
        Generar snippet del contenido.

        Args:
            content: Contenido
            query: Consulta
            max_length: Longitud máxima

        Returns:
            Snippet
        """
        query_terms = self._tokenize(query)
        
        # Buscar primera ocurrencia de términos de la consulta
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                snippet = sentence.strip()
                if len(snippet) > max_length:
                    snippet = snippet[:max_length] + "..."
                return snippet
        
        # Si no se encuentra, devolver inicio
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content

    def find_similar(
        self,
        doc_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Encontrar documentos similares.

        Args:
            doc_id: ID del documento
            top_k: Número de resultados

        Returns:
            Lista de documentos similares
        """
        if doc_id not in self.documents:
            return []
        
        # Usar el contenido como consulta
        content = self.documents[doc_id]
        results = self.search(content, top_k=top_k + 1, threshold=0.05)
        
        # Filtrar el documento mismo
        results = [r for r in results if r["doc_id"] != doc_id]
        
        return results[:top_k]

    def clear_index(self):
        """Limpiar índice"""
        self.index.clear()
        self.documents.clear()
        logger.info("Índice limpiado")






