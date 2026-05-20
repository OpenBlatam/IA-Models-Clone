"""
Comparador de Documentos
=========================

Sistema para comparar documentos y detectar similitud semántica.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .embedding_generator import EmbeddingGenerator
from .document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class DocumentSimilarity:
    """Resultado de comparación de documentos"""
    document1_id: str
    document2_id: str
    similarity_score: float
    common_keywords: List[str]
    common_entities: List[Dict[str, Any]]
    differences: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class DocumentComparator:
    """
    Comparador de documentos con análisis semántico
    
    Proporciona:
    - Comparación semántica usando embeddings
    - Detección de keywords y entidades comunes
    - Análisis de diferencias
    - Búsqueda de documentos similares
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar comparador
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        self.embedding_generator = analyzer.embedding_generator
        logger.info("DocumentComparator inicializado")
    
    async def compare_documents(
        self,
        doc1_content: str,
        doc2_content: str,
        doc1_id: Optional[str] = None,
        doc2_id: Optional[str] = None,
        include_analysis: bool = True
    ) -> DocumentSimilarity:
        """
        Comparar dos documentos
        
        Args:
            doc1_content: Contenido del primer documento
            doc2_content: Contenido del segundo documento
            doc1_id: ID del primer documento
            doc2_id: ID del segundo documento
            include_analysis: Si True, incluye análisis detallado
        
        Returns:
            DocumentSimilarity con resultados de comparación
        """
        doc1_id = doc1_id or "doc1"
        doc2_id = doc2_id or "doc2"
        
        # Generar embeddings
        embeddings = await self.embedding_generator.generate_embeddings(
            [doc1_content, doc2_content]
        )
        emb1, emb2 = embeddings[0], embeddings[1]
        
        # Calcular similitud coseno
        similarity = self.embedding_generator.compute_similarity(
            doc1_content, doc2_content
        )
        
        # Análisis adicional si se solicita
        common_keywords = []
        common_entities = []
        differences = {}
        
        if include_analysis:
            # Extraer keywords de ambos documentos
            keywords1 = await self.analyzer.extract_keywords(doc1_content)
            keywords2 = await self.analyzer.extract_keywords(doc2_content)
            
            # Keywords comunes
            common_keywords = list(set(keywords1) & set(keywords2))
            
            # Extraer entidades
            entities1 = await self.analyzer.extract_entities(doc1_content)
            entities2 = await self.analyzer.extract_entities(doc2_content)
            
            # Entidades comunes
            entities1_text = {e["text"].lower() for e in entities1}
            entities2_text = {e["text"].lower() for e in entities2}
            common_entity_texts = entities1_text & entities2_text
            
            common_entities = [
                e for e in entities1
                if e["text"].lower() in common_entity_texts
            ]
            
            # Análisis de diferencias
            differences = {
                "keywords_only_in_doc1": list(set(keywords1) - set(keywords2)),
                "keywords_only_in_doc2": list(set(keywords2) - set(keywords1)),
                "entities_only_in_doc1": [
                    e for e in entities1
                    if e["text"].lower() not in entities2_text
                ],
                "entities_only_in_doc2": [
                    e for e in entities2
                    if e["text"].lower() not in entities1_text
                ],
                "length_difference": abs(len(doc1_content) - len(doc2_content)),
                "length_ratio": min(len(doc1_content), len(doc2_content)) / max(len(doc1_content), len(doc2_content)) if max(len(doc1_content), len(doc2_content)) > 0 else 0
            }
        
        return DocumentSimilarity(
            document1_id=doc1_id,
            document2_id=doc2_id,
            similarity_score=similarity,
            common_keywords=common_keywords,
            common_entities=common_entities,
            differences=differences
        )
    
    async def find_similar_documents(
        self,
        target_doc: str,
        document_corpus: List[Tuple[str, str]],
        threshold: float = 0.7,
        top_k: int = 5
    ) -> List[DocumentSimilarity]:
        """
        Encontrar documentos similares en un corpus
        
        Args:
            target_doc: Documento objetivo
            document_corpus: Lista de tuplas (doc_id, doc_content)
            threshold: Umbral mínimo de similitud
            top_k: Número máximo de resultados
        
        Returns:
            Lista de DocumentSimilarity ordenada por similitud
        """
        # Generar embedding del documento objetivo
        target_embedding = await self.embedding_generator.generate_embeddings(
            [target_doc]
        )
        target_embedding = target_embedding[0] if isinstance(target_embedding, list) else target_embedding
        
        # Generar embeddings del corpus
        corpus_contents = [content for _, content in document_corpus]
        corpus_embeddings = await self.embedding_generator.generate_embeddings(
            corpus_contents
        )
        
        # Calcular similitudes
        similarities = []
        for i, (doc_id, doc_content) in enumerate(document_corpus):
            corpus_emb = corpus_embeddings[i] if isinstance(corpus_embeddings, list) else corpus_embeddings[i]
            
            # Calcular similitud coseno
            if isinstance(target_embedding, np.ndarray) and isinstance(corpus_emb, np.ndarray):
                similarity = np.dot(target_embedding, corpus_emb) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(corpus_emb)
                )
            else:
                similarity = self.embedding_generator.compute_similarity(
                    target_doc, doc_content
                )
            
            if similarity >= threshold:
                # Comparación rápida sin análisis detallado
                similarity_obj = DocumentSimilarity(
                    document1_id="target",
                    document2_id=doc_id,
                    similarity_score=float(similarity),
                    common_keywords=[],
                    common_entities=[],
                    differences={}
                )
                similarities.append(similarity_obj)
        
        # Ordenar por similitud y retornar top_k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]
    
    async def detect_plagiarism(
        self,
        suspicious_doc: str,
        reference_corpus: List[Tuple[str, str]],
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Detectar posible plagio comparando con un corpus de referencia
        
        Args:
            suspicious_doc: Documento sospechoso
            reference_corpus: Corpus de documentos de referencia
            threshold: Umbral de similitud para considerar plagio
        
        Returns:
            Lista de documentos que podrían ser plagio
        """
        similar_docs = await self.find_similar_documents(
            suspicious_doc,
            reference_corpus,
            threshold=threshold,
            top_k=10
        )
        
        plagiarism_results = []
        for sim in similar_docs:
            if sim.similarity_score >= threshold:
                plagiarism_results.append({
                    "reference_document_id": sim.document2_id,
                    "similarity_score": sim.similarity_score,
                    "risk_level": "high" if sim.similarity_score >= 0.95 else "medium",
                    "timestamp": sim.timestamp
                })
        
        return plagiarism_results
















