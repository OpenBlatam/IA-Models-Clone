"""
Detección de Anomalías en Documentos
=====================================

Sistema para detectar anomalías, inconsistencias y patrones inusuales.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import Counter

from .document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Anomalía detectada"""
    type: str
    severity: str  # low, medium, high, critical
    description: str
    location: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyReport:
    """Reporte de anomalías"""
    document_id: str
    anomalies: List[Anomaly]
    total_anomalies: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    risk_score: float  # 0-100
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Contar por severidad
        self.critical_count = sum(1 for a in self.anomalies if a.severity == "critical")
        self.high_count = sum(1 for a in self.anomalies if a.severity == "high")
        self.medium_count = sum(1 for a in self.anomalies if a.severity == "medium")
        self.low_count = sum(1 for a in self.anomalies if a.severity == "low")
        
        # Calcular risk score
        self.risk_score = (
            self.critical_count * 25 +
            self.high_count * 15 +
            self.medium_count * 8 +
            self.low_count * 3
        )
        self.risk_score = min(100, self.risk_score)


class AnomalyDetector:
    """
    Detector de anomalías en documentos
    
    Detecta:
    - Anomalías de contenido
    - Inconsistencias estructurales
    - Patrones inusuales
    - Valores atípicos
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar detector
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        logger.info("AnomalyDetector inicializado")
    
    async def detect_anomalies(
        self,
        content: str,
        document_id: Optional[str] = None,
        baseline: Optional[Dict[str, Any]] = None
    ) -> AnomalyReport:
        """
        Detectar anomalías en documento
        
        Args:
            content: Contenido del documento
            document_id: ID del documento
            baseline: Baseline de comparación (opcional)
        
        Returns:
            AnomalyReport con anomalías detectadas
        """
        document_id = document_id or "unknown"
        anomalies = []
        
        # Análisis básico
        analysis = await self.analyzer.analyze_document(content)
        
        # Detectar anomalías de longitud
        anomalies.extend(self._detect_length_anomalies(content, baseline))
        
        # Detectar anomalías de sentimiento
        if analysis.sentiment:
            anomalies.extend(self._detect_sentiment_anomalies(analysis.sentiment, baseline))
        
        # Detectar anomalías de estructura
        anomalies.extend(self._detect_structure_anomalies(content))
        
        # Detectar anomalías de contenido
        anomalies.extend(self._detect_content_anomalies(content, analysis))
        
        # Detectar anomalías de keywords
        if analysis.keywords:
            anomalies.extend(self._detect_keyword_anomalies(analysis.keywords, baseline))
        
        return AnomalyReport(
            document_id=document_id,
            anomalies=anomalies,
            total_anomalies=len(anomalies)
        )
    
    def _detect_length_anomalies(
        self,
        content: str,
        baseline: Optional[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detectar anomalías de longitud"""
        anomalies = []
        length = len(content)
        
        if baseline:
            avg_length = baseline.get("avg_length", 0)
            std_length = baseline.get("std_length", 1000)
            
            if length < avg_length - 3 * std_length:
                anomalies.append(Anomaly(
                    type="length",
                    severity="high",
                    description=f"Documento extremadamente corto ({length} caracteres)",
                    confidence=0.9
                ))
            elif length > avg_length + 3 * std_length:
                anomalies.append(Anomaly(
                    type="length",
                    severity="medium",
                    description=f"Documento extremadamente largo ({length} caracteres)",
                    confidence=0.8
                ))
        else:
            # Detección sin baseline
            if length < 50:
                anomalies.append(Anomaly(
                    type="length",
                    severity="medium",
                    description="Documento muy corto",
                    confidence=0.7
                ))
            elif length > 100000:
                anomalies.append(Anomaly(
                    type="length",
                    severity="low",
                    description="Documento muy largo",
                    confidence=0.6
                ))
        
        return anomalies
    
    def _detect_sentiment_anomalies(
        self,
        sentiment: Dict[str, float],
        baseline: Optional[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detectar anomalías de sentimiento"""
        anomalies = []
        
        # Sentimiento muy extremo
        if sentiment.get("positive", 0) > 0.9:
            anomalies.append(Anomaly(
                type="sentiment",
                severity="low",
                description="Sentimiento extremadamente positivo",
                confidence=0.7
            ))
        elif sentiment.get("negative", 0) > 0.9:
            anomalies.append(Anomaly(
                type="sentiment",
                severity="medium",
                description="Sentimiento extremadamente negativo",
                confidence=0.8
            ))
        
        # Comparar con baseline
        if baseline:
            baseline_sentiment = baseline.get("avg_sentiment", 0.5)
            current_sentiment = sentiment.get("positive", 0) - sentiment.get("negative", 0)
            
            if abs(current_sentiment - baseline_sentiment) > 0.5:
                anomalies.append(Anomaly(
                    type="sentiment",
                    severity="medium",
                    description="Sentimiento muy diferente al baseline",
                    confidence=0.75
                ))
        
        return anomalies
    
    def _detect_structure_anomalies(self, content: str) -> List[Anomaly]:
        """Detectar anomalías de estructura"""
        anomalies = []
        
        # Sin párrafos
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 2 and len(content) > 500:
            anomalies.append(Anomaly(
                type="structure",
                severity="low",
                description="Documento largo sin párrafos claros",
                confidence=0.6
            ))
        
        # Demasiados espacios
        if '  ' * 10 in content:
            anomalies.append(Anomaly(
                type="structure",
                severity="low",
                description="Espacios excesivos detectados",
                confidence=0.7
            ))
        
        # Sin puntuación
        punctuation_count = sum(1 for c in content if c in '.,!?;:')
        if len(content) > 100 and punctuation_count < len(content) / 100:
            anomalies.append(Anomaly(
                type="structure",
                severity="medium",
                description="Documento con muy poca puntuación",
                confidence=0.8
            ))
        
        return anomalies
    
    def _detect_content_anomalies(
        self,
        content: str,
        analysis: Any
    ) -> List[Anomaly]:
        """Detectar anomalías de contenido"""
        anomalies = []
        
        # Contenido repetitivo
        words = content.split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.3:
                anomalies.append(Anomaly(
                    type="content",
                    severity="medium",
                    description="Alto nivel de repetición en el contenido",
                    confidence=0.75
                ))
        
        # Contenido muy corto para análisis
        if analysis and hasattr(analysis, 'confidence'):
            if analysis.confidence < 0.3 and len(content) > 200:
                anomalies.append(Anomaly(
                    type="content",
                    severity="high",
                    description="Baja confianza en el análisis (posible contenido inusual)",
                    confidence=0.8
                ))
        
        return anomalies
    
    def _detect_keyword_anomalies(
        self,
        keywords: List[str],
        baseline: Optional[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detectar anomalías en keywords"""
        anomalies = []
        
        # Muy pocos keywords
        if len(keywords) < 3:
            anomalies.append(Anomaly(
                type="keywords",
                severity="low",
                description="Muy pocas palabras clave extraídas",
                confidence=0.6
            ))
        
        # Comparar con baseline
        if baseline:
            baseline_keywords = set(baseline.get("common_keywords", []))
            current_keywords = set(keywords)
            
            overlap = len(baseline_keywords & current_keywords)
            if overlap == 0 and len(baseline_keywords) > 0:
                anomalies.append(Anomaly(
                    type="keywords",
                    severity="medium",
                    description="Keywords completamente diferentes al baseline",
                    confidence=0.7
                ))
        
        return anomalies
    
    async def compare_with_baseline(
        self,
        document: Dict[str, Any],
        baseline_documents: List[Dict[str, Any]]
    ) -> AnomalyReport:
        """
        Comparar documento con baseline de documentos similares
        
        Args:
            document: Documento a analizar
            baseline_documents: Lista de documentos de referencia
        
        Returns:
            AnomalyReport con comparaciones
        """
        # Calcular baseline
        baseline = self._calculate_baseline(baseline_documents)
        
        # Detectar anomalías
        return await self.detect_anomalies(
            document["content"],
            document.get("id"),
            baseline
        )
    
    def _calculate_baseline(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular baseline de documentos"""
        if not documents:
            return {}
        
        lengths = [len(doc.get("content", "")) for doc in documents]
        avg_length = np.mean(lengths) if lengths else 0
        std_length = np.std(lengths) if lengths else 0
        
        # Keywords comunes
        all_keywords = []
        for doc in documents:
            # Asumir que tienen keywords en metadata
            keywords = doc.get("metadata", {}).get("keywords", [])
            all_keywords.extend(keywords)
        
        keyword_counts = Counter(all_keywords)
        common_keywords = [kw for kw, count in keyword_counts.most_common(10)]
        
        # Sentimiento promedio
        sentiments = []
        for doc in documents:
            sentiment = doc.get("metadata", {}).get("sentiment", {})
            if sentiment:
                pos = sentiment.get("positive", 0)
                neg = sentiment.get("negative", 0)
                sentiments.append(pos - neg)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0.5
        
        return {
            "avg_length": avg_length,
            "std_length": std_length,
            "common_keywords": common_keywords,
            "avg_sentiment": avg_sentiment
        }
















