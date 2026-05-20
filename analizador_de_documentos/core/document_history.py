"""
Document History - Análisis de Documentos Históricos
=====================================================

Análisis de documentos históricos y archivos antiguos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class HistoricalDocument:
    """Documento histórico."""
    document_id: str
    content: str
    date: datetime
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalAnalysis:
    """Análisis histórico."""
    period: str
    start_date: datetime
    end_date: datetime
    total_documents: int
    trends: Dict[str, Any] = field(default_factory=dict)
    evolution: Dict[str, List[float]] = field(default_factory=dict)
    key_events: List[Dict[str, Any]] = field(default_factory=list)


class HistoricalAnalyzer:
    """Analizador de documentos históricos."""
    
    def __init__(self, analyzer):
        """Inicializar analizador."""
        self.analyzer = analyzer
        self.historical_documents: Dict[str, HistoricalDocument] = {}
    
    def add_historical_document(
        self,
        document_id: str,
        content: str,
        date: datetime,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HistoricalDocument:
        """Agregar documento histórico."""
        doc = HistoricalDocument(
            document_id=document_id,
            content=content,
            date=date,
            source=source,
            metadata=metadata or {}
        )
        
        self.historical_documents[document_id] = doc
        
        return doc
    
    async def analyze_historical_trends(
        self,
        start_date: datetime,
        end_date: datetime,
        metric: str = "quality"
    ) -> HistoricalAnalysis:
        """
        Analizar tendencias históricas.
        
        Args:
            start_date: Fecha inicio
            end_date: Fecha fin
            metric: Métrica a analizar
        
        Returns:
            HistoricalAnalysis con resultados
        """
        # Filtrar documentos del período
        period_docs = [
            doc for doc in self.historical_documents.values()
            if start_date <= doc.date <= end_date
        ]
        
        if not period_docs:
            return HistoricalAnalysis(
                period=f"{start_date.date()} to {end_date.date()}",
                start_date=start_date,
                end_date=end_date,
                total_documents=0
            )
        
        # Agrupar por período (mensual)
        monthly_data = defaultdict(list)
        
        for doc in period_docs:
            month_key = doc.date.strftime("%Y-%m")
            monthly_data[month_key].append(doc)
        
        # Analizar cada documento
        trends = {}
        evolution = defaultdict(list)
        
        for month_key, docs in sorted(monthly_data.items()):
            month_metrics = []
            
            for doc in docs:
                try:
                    # Analizar documento
                    result = await self.analyzer.analyze_document(document_content=doc.content)
                    
                    if metric == "quality":
                        quality = await self.analyzer.analyze_quality(result.content)
                        month_metrics.append(quality.overall_score)
                    elif metric == "grammar":
                        grammar = await self.analyzer.analyze_grammar(result.content)
                        month_metrics.append(grammar.overall_score)
                    else:
                        # Métrica genérica
                        month_metrics.append(100.0)
                except Exception as e:
                    logger.error(f"Error analizando documento histórico {doc.document_id}: {e}")
            
            if month_metrics:
                avg_metric = sum(month_metrics) / len(month_metrics)
                trends[month_key] = avg_metric
                evolution[metric].append(avg_metric)
        
        # Detectar eventos clave (cambios significativos)
        key_events = self._detect_key_events(evolution.get(metric, []), sorted(monthly_data.keys()))
        
        return HistoricalAnalysis(
            period=f"{start_date.date()} to {end_date.date()}",
            start_date=start_date,
            end_date=end_date,
            total_documents=len(period_docs),
            trends=trends,
            evolution=dict(evolution),
            key_events=key_events
        )
    
    def _detect_key_events(
        self,
        metrics: List[float],
        dates: List[str]
    ) -> List[Dict[str, Any]]:
        """Detectar eventos clave."""
        events = []
        
        if len(metrics) < 2:
            return events
        
        # Detectar cambios significativos
        for i in range(1, len(metrics)):
            change = metrics[i] - metrics[i - 1]
            
            if abs(change) > 15:  # Cambio significativo
                events.append({
                    "date": dates[i],
                    "type": "significant_change",
                    "change": change,
                    "direction": "increase" if change > 0 else "decrease",
                    "from_value": metrics[i - 1],
                    "to_value": metrics[i]
                })
        
        return events
    
    def get_documents_by_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[HistoricalDocument]:
        """Obtener documentos por período."""
        return [
            doc for doc in self.historical_documents.values()
            if start_date <= doc.date <= end_date
        ]
    
    def get_oldest_document(self) -> Optional[HistoricalDocument]:
        """Obtener documento más antiguo."""
        if not self.historical_documents:
            return None
        
        return min(self.historical_documents.values(), key=lambda d: d.date)
    
    def get_newest_document(self) -> Optional[HistoricalDocument]:
        """Obtener documento más reciente."""
        if not self.historical_documents:
            return None
        
        return max(self.historical_documents.values(), key=lambda d: d.date)


__all__ = [
    "HistoricalAnalyzer",
    "HistoricalDocument",
    "HistoricalAnalysis"
]


