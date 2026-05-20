"""
Plagiarism Detector - Sistema de detección de plagio
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


class PlagiarismDetector:
    """Detector de plagio"""

    def __init__(self):
        """Inicializar detector"""
        self.reference_documents: Dict[str, str] = {}
        self.fingerprints: Dict[str, List[str]] = {}

    def add_reference(self, doc_id: str, content: str):
        """
        Agregar documento de referencia.

        Args:
            doc_id: ID del documento
            content: Contenido
        """
        self.reference_documents[doc_id] = content
        self.fingerprints[doc_id] = self._generate_fingerprints(content)
        logger.debug(f"Documento de referencia agregado: {doc_id}")

    def check(
        self,
        content: str,
        min_similarity: float = 0.5,
        min_phrase_length: int = 5
    ) -> Dict[str, Any]:
        """
        Verificar plagio.

        Args:
            content: Contenido a verificar
            min_similarity: Similitud mínima
            min_phrase_length: Longitud mínima de frase

        Returns:
            Resultado de verificación
        """
        content_fingerprints = self._generate_fingerprints(content)
        
        matches = []
        max_similarity = 0.0
        most_similar_doc = None
        
        for doc_id, ref_fingerprints in self.fingerprints.items():
            similarity = self._calculate_similarity(
                content_fingerprints,
                ref_fingerprints
            )
            
            if similarity >= min_similarity:
                # Encontrar frases similares
                similar_phrases = self._find_similar_phrases(
                    content,
                    self.reference_documents[doc_id],
                    min_phrase_length
                )
                
                matches.append({
                    "doc_id": doc_id,
                    "similarity": similarity,
                    "similar_phrases": similar_phrases
                })
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_doc = doc_id
        
        return {
            "is_plagiarized": len(matches) > 0,
            "max_similarity": max_similarity,
            "most_similar_doc": most_similar_doc,
            "matches": matches,
            "match_count": len(matches)
        }

    def _generate_fingerprints(self, content: str, n: int = 5) -> List[str]:
        """
        Generar fingerprints (n-gramas).

        Args:
            content: Contenido
            n: Tamaño de n-grama

        Returns:
            Lista de fingerprints
        """
        # Normalizar texto
        text = re.sub(r'[^\w\s]', '', content.lower())
        words = text.split()
        
        if len(words) < n:
            return []
        
        # Generar n-gramas
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            # Crear hash del n-grama
            fingerprint = hashlib.md5(ngram.encode()).hexdigest()
            ngrams.append(fingerprint)
        
        return ngrams

    def _calculate_similarity(
        self,
        fingerprints1: List[str],
        fingerprints2: List[str]
    ) -> float:
        """
        Calcular similitud entre fingerprints.

        Args:
            fingerprints1: Fingerprints 1
            fingerprints2: Fingerprints 2

        Returns:
            Similitud (0-1)
        """
        if not fingerprints1 or not fingerprints2:
            return 0.0
        
        set1 = set(fingerprints1)
        set2 = set(fingerprints2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _find_similar_phrases(
        self,
        text1: str,
        text2: str,
        min_length: int
    ) -> List[Dict[str, Any]]:
        """
        Encontrar frases similares.

        Args:
            text1: Texto 1
            text2: Texto 2
            min_length: Longitud mínima

        Returns:
            Lista de frases similares
        """
        # Normalizar textos
        text1_clean = re.sub(r'[^\w\s]', ' ', text1.lower())
        text2_clean = re.sub(r'[^\w\s]', ' ', text2.lower())
        
        words1 = text1_clean.split()
        words2 = text2_clean.split()
        
        similar_phrases = []
        
        # Buscar secuencias comunes
        for i in range(len(words1) - min_length + 1):
            phrase1 = words1[i:i+min_length]
            phrase1_str = ' '.join(phrase1)
            
            # Buscar en texto 2
            text2_str = ' '.join(words2)
            if phrase1_str in text2_str:
                # Encontrar posición en texto original
                original_phrase = self._find_original_phrase(
                    text1,
                    phrase1_str
                )
                
                similar_phrases.append({
                    "phrase": original_phrase,
                    "length": len(phrase1),
                    "position": i
                })
        
        return similar_phrases

    def _find_original_phrase(self, text: str, phrase_lower: str) -> str:
        """Encontrar frase original con formato"""
        text_lower = text.lower()
        phrase_lower_clean = phrase_lower.lower()
        
        # Buscar posición
        pos = text_lower.find(phrase_lower_clean)
        if pos != -1:
            # Extraer frase original
            end_pos = pos + len(phrase_lower_clean)
            return text[pos:end_pos]
        
        return phrase_lower

    def compare_documents(
        self,
        doc1_id: str,
        doc2_id: str
    ) -> Dict[str, Any]:
        """
        Comparar dos documentos.

        Args:
            doc1_id: ID documento 1
            doc2_id: ID documento 2

        Returns:
            Comparación
        """
        if doc1_id not in self.fingerprints or doc2_id not in self.fingerprints:
            return {"error": "Documento no encontrado"}
        
        similarity = self._calculate_similarity(
            self.fingerprints[doc1_id],
            self.fingerprints[doc2_id]
        )
        
        return {
            "doc1_id": doc1_id,
            "doc2_id": doc2_id,
            "similarity": similarity,
            "is_similar": similarity > 0.5
        }






