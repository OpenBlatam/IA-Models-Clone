"""
Document Multi-Language - Análisis Multi-idioma Avanzado
========================================================

Análisis de documentos en múltiples idiomas simultáneamente.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetection:
    """Detección de idioma."""
    language: str
    confidence: float
    alternatives: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class MultiLanguageAnalysis:
    """Análisis multi-idioma."""
    detected_languages: List[LanguageDetection]
    primary_language: str
    language_distribution: Dict[str, float]
    analysis_by_language: Dict[str, Any] = field(default_factory=dict)


class MultiLanguageAnalyzer:
    """Analizador multi-idioma."""
    
    def __init__(self, analyzer):
        """Inicializar analizador."""
        self.analyzer = analyzer
        self.supported_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ru', 'ar'
        ]
    
    async def detect_languages(self, content: str) -> List[LanguageDetection]:
        """
        Detectar todos los idiomas en el documento.
        
        Args:
            content: Contenido del documento
        
        Returns:
            Lista de LanguageDetection
        """
        # Dividir en segmentos
        segments = self._split_into_segments(content)
        
        # Detectar idioma por segmento
        language_scores = defaultdict(float)
        language_counts = defaultdict(int)
        
        for segment in segments:
            if len(segment.strip()) < 10:
                continue
            
            # Detectar idioma del segmento (simplificado)
            lang_detection = await self._detect_segment_language(segment)
            
            if lang_detection:
                language_scores[lang_detection["language"]] += lang_detection["confidence"]
                language_counts[lang_detection["language"]] += 1
        
        # Normalizar scores
        total_segments = len([s for s in segments if len(s.strip()) >= 10])
        if total_segments > 0:
            for lang in language_scores:
                language_scores[lang] /= language_counts[lang]
        
        # Crear detecciones
        detections = []
        for lang, score in sorted(language_scores.items(), key=lambda x: x[1], reverse=True):
            detections.append(LanguageDetection(
                language=lang,
                confidence=score / total_segments if total_segments > 0 else 0,
                alternatives=[]
            ))
        
        return detections
    
    async def analyze_multilanguage(
        self,
        content: str,
        languages: Optional[List[str]] = None
    ) -> MultiLanguageAnalysis:
        """
        Analizar documento multi-idioma.
        
        Args:
            content: Contenido del documento
            languages: Idiomas específicos a analizar (opcional)
        
        Returns:
            MultiLanguageAnalysis con resultados
        """
        # Detectar idiomas
        detections = await self.detect_languages(content)
        
        if not detections:
            # Fallback a detección simple
            if hasattr(self.analyzer, 'detect_language'):
                simple_detection = await self.analyzer.detect_language(content)
                detections = [LanguageDetection(
                    language=simple_detection.get("language", "unknown"),
                    confidence=simple_detection.get("confidence", 0.0)
                )]
        
        primary_language = detections[0].language if detections else "unknown"
        
        # Distribución de idiomas
        language_distribution = {
            det.language: det.confidence
            for det in detections
        }
        
        # Analizar por idioma principal
        analysis_by_language = {}
        if primary_language != "unknown":
            try:
                result = await self.analyzer.analyze_document(document_content=content)
                analysis_by_language[primary_language] = {
                    "classification": result.classification if hasattr(result, 'classification') else None,
                    "summary": result.summary if hasattr(result, 'summary') else None,
                    "keywords": result.keywords if hasattr(result, 'keywords') else []
                }
            except Exception as e:
                logger.error(f"Error analizando por idioma {primary_language}: {e}")
        
        return MultiLanguageAnalysis(
            detected_languages=detections,
            primary_language=primary_language,
            language_distribution=language_distribution,
            analysis_by_language=analysis_by_language
        )
    
    def _split_into_segments(self, content: str, segment_length: int = 500) -> List[str]:
        """Dividir contenido en segmentos."""
        segments = []
        words = content.split()
        
        current_segment = []
        current_length = 0
        
        for word in words:
            current_segment.append(word)
            current_length += len(word) + 1
            
            if current_length >= segment_length:
                segments.append(' '.join(current_segment))
                current_segment = []
                current_length = 0
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    async def _detect_segment_language(self, segment: str) -> Optional[Dict[str, Any]]:
        """Detectar idioma de un segmento (simplificado)."""
        # En producción usar bibliotecas como langdetect
        # Por ahora implementación básica
        
        segment_lower = segment.lower()
        
        # Palabras comunes por idioma
        language_patterns = {
            'en': ['the', 'and', 'is', 'are', 'was', 'were', 'this', 'that'],
            'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'una', 'es', 'son'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'être', 'pour', 'dans'],
            'de': ['der', 'die', 'das', 'und', 'ist', 'sind', 'für', 'mit'],
            'pt': ['o', 'a', 'de', 'que', 'e', 'em', 'um', 'uma', 'é', 'são'],
            'it': ['il', 'la', 'di', 'che', 'e', 'in', 'un', 'una', 'è', 'sono']
        }
        
        scores = {}
        for lang, patterns in language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in segment_lower)
            if score > 0:
                scores[lang] = score / len(patterns)
        
        if scores:
            detected_lang = max(scores.items(), key=lambda x: x[1])[0]
            return {
                "language": detected_lang,
                "confidence": scores[detected_lang]
            }
        
        return None


__all__ = [
    "MultiLanguageAnalyzer",
    "MultiLanguageAnalysis",
    "LanguageDetection"
]
















