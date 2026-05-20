"""
Document Reporting - Sistema de Reportes Avanzado
==============================================

Sistema avanzado de generación de reportes.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """Sección de reporte."""
    section_id: str
    title: str
    content: Any
    section_type: str = "text"  # 'text', 'table', 'chart', 'list'
    order: int = 0


@dataclass
class DocumentReport:
    """Reporte de documento."""
    report_id: str
    report_type: str
    title: str
    sections: List[ReportSection]
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Generador de reportes."""
    
    def __init__(self, analyzer):
        """Inicializar generador."""
        self.analyzer = analyzer
        self.reports: Dict[str, DocumentReport] = {}
    
    async def generate_comprehensive_report(
        self,
        document_id: str,
        analysis_result: Any,
        quality_analysis: Optional[Any] = None,
        grammar_analysis: Optional[Any] = None,
        metadata: Optional[Any] = None
    ) -> DocumentReport:
        """
        Generar reporte comprehensivo.
        
        Args:
            document_id: ID del documento
            analysis_result: Resultado de análisis
            quality_analysis: Análisis de calidad
            grammar_analysis: Análisis gramatical
            metadata: Metadatos
        
        Returns:
            DocumentReport completo
        """
        sections = []
        order = 0
        
        # Sección: Resumen Ejecutivo
        sections.append(ReportSection(
            section_id="executive_summary",
            title="Resumen Ejecutivo",
            content=self._generate_executive_summary(analysis_result, quality_analysis),
            section_type="text",
            order=order
        ))
        order += 1
        
        # Sección: Análisis Básico
        sections.append(ReportSection(
            section_id="basic_analysis",
            title="Análisis Básico",
            content={
                "classification": analysis_result.classification if hasattr(analysis_result, 'classification') else {},
                "summary": analysis_result.summary if hasattr(analysis_result, 'summary') else None,
                "keywords": analysis_result.keywords if hasattr(analysis_result, 'keywords') else []
            },
            section_type="text",
            order=order
        ))
        order += 1
        
        # Sección: Análisis de Calidad
        if quality_analysis:
            sections.append(ReportSection(
                section_id="quality_analysis",
                title="Análisis de Calidad",
                content={
                    "overall_score": quality_analysis.overall_score,
                    "readability": quality_analysis.readability_score,
                    "completeness": quality_analysis.completeness_score,
                    "structure": quality_analysis.structure_score,
                    "issues": quality_analysis.issues,
                    "recommendations": quality_analysis.recommendations
                },
                section_type="table",
                order=order
            ))
            order += 1
        
        # Sección: Análisis Gramatical
        if grammar_analysis:
            sections.append(ReportSection(
                section_id="grammar_analysis",
                title="Análisis Gramatical",
                content={
                    "overall_score": grammar_analysis.overall_score,
                    "spelling": grammar_analysis.spelling_score,
                    "grammar": grammar_analysis.grammar_score,
                    "punctuation": grammar_analysis.punctuation_score,
                    "readability_index": grammar_analysis.readability_index,
                    "issues_count": len(grammar_analysis.issues)
                },
                section_type="table",
                order=order
            ))
            order += 1
        
        # Sección: Metadatos
        if metadata:
            sections.append(ReportSection(
                section_id="metadata",
                title="Metadatos",
                content=metadata.__dict__ if hasattr(metadata, '__dict__') else metadata,
                section_type="list",
                order=order
            ))
            order += 1
        
        # Sección: Recomendaciones
        if hasattr(self.analyzer, 'recommendation_engine'):
            try:
                recommendations = await self.analyzer.generate_recommendations(
                    analysis_result, quality_analysis, grammar_analysis
                )
                sections.append(ReportSection(
                    section_id="recommendations",
                    title="Recomendaciones",
                    content=[
                        {
                            "title": r.title,
                            "description": r.description,
                            "priority": r.priority,
                            "actions": r.action_items
                        }
                        for r in recommendations[:10]
                    ],
                    section_type="list",
                    order=order
                ))
            except:
                pass
        
        report = DocumentReport(
            report_id=f"report_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="comprehensive",
            title=f"Reporte Completo - Documento {document_id}",
            sections=sections,
            metadata={
                "document_id": document_id,
                "sections_count": len(sections)
            }
        )
        
        self.reports[report.report_id] = report
        
        return report
    
    def _generate_executive_summary(
        self,
        analysis_result: Any,
        quality_analysis: Optional[Any]
    ) -> str:
        """Generar resumen ejecutivo."""
        summary_parts = []
        
        if hasattr(analysis_result, 'summary') and analysis_result.summary:
            summary_parts.append(f"Resumen: {analysis_result.summary[:200]}...")
        
        if quality_analysis:
            summary_parts.append(
                f"Calidad General: {quality_analysis.overall_score:.1f}/100"
            )
        
        if hasattr(analysis_result, 'classification') and analysis_result.classification:
            top_class = max(analysis_result.classification.items(), key=lambda x: x[1])[0]
            summary_parts.append(f"Clasificación Principal: {top_class}")
        
        return "\n".join(summary_parts) if summary_parts else "Resumen ejecutivo no disponible"
    
    async def generate_statistics_report(
        self,
        period_days: int = 30
    ) -> DocumentReport:
        """Generar reporte de estadísticas."""
        sections = []
        
        # Obtener métricas
        if hasattr(self.analyzer, 'metrics_collector'):
            dashboard = await self.analyzer.generate_metrics_dashboard("daily", period_days)
            
            sections.append(ReportSection(
                section_id="overview",
                title="Resumen General",
                content={
                    "total_documents": dashboard.total_documents,
                    "total_analyses": dashboard.total_analyses,
                    "average_quality": dashboard.average_quality_score,
                    "average_grammar": dashboard.average_grammar_score
                },
                section_type="table",
                order=0
            ))
            
            sections.append(ReportSection(
                section_id="trends",
                title="Tendencias",
                content=dashboard.trends,
                section_type="chart",
                order=1
            ))
        
        report = DocumentReport(
            report_id=f"stats_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="statistics",
            title=f"Reporte de Estadísticas - {period_days} días",
            sections=sections
        )
        
        return report
    
    def get_report(self, report_id: str) -> Optional[DocumentReport]:
        """Obtener reporte."""
        return self.reports.get(report_id)
    
    def list_reports(
        self,
        report_type: Optional[str] = None
    ) -> List[DocumentReport]:
        """Listar reportes."""
        reports = list(self.reports.values())
        
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        
        return sorted(reports, key=lambda r: r.generated_at, reverse=True)


__all__ = [
    "ReportGenerator",
    "DocumentReport",
    "ReportSection"
]



