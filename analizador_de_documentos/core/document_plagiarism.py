"""
Document Plagiarism - Detección de Plagio
==========================================

Sistema avanzado de detección de plagio y similitud entre documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PlagiarismMatch:
    """Coincidencia de plagio."""
    source_document_id: str
    source_content: str
    matched_content: str
    similarity_score: float
    match_type: str  # 'exact', 'near', 'paraphrased'
    position_start: int
    position_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlagiarismReport:
    """Reporte de plagio."""
    document_id: str
    overall_similarity: float
    matches: List[PlagiarismMatch]
    plagiarism_percentage: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    generated_at: datetime = field(default_factory=datetime.now)


class PlagiarismDetector:
    """Detector de plagio."""
    
    def __init__(self, analyzer):
        """Inicializar detector."""
        self.analyzer = analyzer
        self.document_fingerprints: Dict[str, List[str]] = {}
        self.reference_documents: Dict[str, str] = {}
    
    def add_reference_document(self, document_id: str, content: str):
        """Agregar documento de referencia."""
        self.reference_documents[document_id] = content
        self.document_fingerprints[document_id] = self._generate_fingerprints(content)
    
    def _generate_fingerprints(self, content: str, ngram_size: int = 5) -> List[str]:
        """Generar fingerprints del documento."""
        words = content.split()
        fingerprints = []
        
        for i in range(len(words) - ngram_size + 1):
            ngram = ' '.join(words[i:i + ngram_size])
            fingerprint = hashlib.md5(ngram.encode()).hexdigest()
            fingerprints.append(fingerprint)
        
        return fingerprints
    
    async def detect_plagiarism(
        self,
        document_id: str,
        content: str,
        threshold: float = 0.7,
        check_references: bool = True
    ) -> PlagiarismReport:
        """
        Detectar plagio en un documento.
        
        Args:
            document_id: ID del documento a verificar
            content: Contenido del documento
            threshold: Umbral de similitud (0-1)
            check_references: Verificar contra documentos de referencia
        
        Returns:
            PlagiarismReport con resultados
        """
        matches = []
        content_fingerprints = self._generate_fingerprints(content)
        
        if check_references:
            for ref_id, ref_content in self.reference_documents.items():
                ref_fingerprints = self.document_fingerprints[ref_id]
                
                # Calcular similitud de fingerprints
                similarity = self._calculate_fingerprint_similarity(
                    content_fingerprints,
                    ref_fingerprints
                )
                
                if similarity >= threshold:
                    # Encontrar coincidencias específicas
                    specific_matches = self._find_specific_matches(content, ref_content, threshold)
                    
                    for match in specific_matches:
                        match.source_document_id = ref_id
                        matches.append(match)
        
        # Calcular similitud general
        if matches:
            overall_similarity = sum(m.score for m in matches) / len(matches)
            plagiarism_percentage = (len(matches) / max(len(content.split()), 1)) * 100
        else:
            overall_similarity = 0.0
            plagiarism_percentage = 0.0
        
        # Determinar nivel de riesgo
        risk_level = self._determine_risk_level(overall_similarity, plagiarism_percentage)
        
        return PlagiarismReport(
            document_id=document_id,
            overall_similarity=overall_similarity,
            matches=matches,
            plagiarism_percentage=plagiarism_percentage,
            risk_level=risk_level
        )
    
    def _calculate_fingerprint_similarity(
        self,
        fingerprints1: List[str],
        fingerprints2: List[str]
    ) -> float:
        """Calcular similitud entre dos conjuntos de fingerprints."""
        if not fingerprints1 or not fingerprints2:
            return 0.0
        
        set1 = set(fingerprints1)
        set2 = set(fingerprints2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_specific_matches(
        self,
        content: str,
        reference_content: str,
        threshold: float
    ) -> List[PlagiarismMatch]:
        """Encontrar coincidencias específicas."""
        matches = []
        content_sentences = content.split('.')
        ref_sentences = reference_content.split('.')
        
        for i, content_sent in enumerate(content_sentences):
            if not content_sent.strip():
                continue
            
            for j, ref_sent in enumerate(ref_sentences):
                if not ref_sent.strip():
                    continue
                
                similarity = self._calculate_text_similarity(content_sent, ref_sent)
                
                if similarity >= threshold:
                    matches.append(PlagiarismMatch(
                        source_document_id="",  # Se asignará después
                        source_content=ref_sent,
                        matched_content=content_sent,
                        similarity_score=similarity,
                        match_type="near" if similarity < 0.95 else "exact",
                        position_start=0,  # Simplificado
                        position_end=len(content_sent)
                    ))
        
        return matches
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud entre dos textos."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_risk_level(
        self,
        similarity: float,
        percentage: float
    ) -> str:
        """Determinar nivel de riesgo."""
        if similarity >= 0.9 or percentage >= 50:
            return "critical"
        elif similarity >= 0.7 or percentage >= 30:
            return "high"
        elif similarity >= 0.5 or percentage >= 15:
            return "medium"
        else:
            return "low"
    
    def get_reference_documents(self) -> List[str]:
        """Obtener lista de documentos de referencia."""
        return list(self.reference_documents.keys())


__all__ = [
    "PlagiarismDetector",
    "PlagiarismReport",
    "PlagiarismMatch"
]


