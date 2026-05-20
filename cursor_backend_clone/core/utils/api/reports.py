"""
Reports - Sistema de Reportes
==============================

Sistema para generar reportes y estadísticas del sistema.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Report:
    """Reporte"""
    title: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    data: Dict[str, Any]
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "data": self.data,
            "summary": self.summary,
            "metadata": self.metadata
        }


class ReportGenerator:
    """
    Generador de reportes.
    
    Genera reportes de estadísticas y métricas del sistema.
    """
    
    def __init__(self):
        self.reports: List[Report] = []
        self.max_reports = 1000
    
    def generate_system_report(
        self,
        agent,
        period_hours: int = 24,
        include_metrics: bool = True,
        include_tasks: bool = True,
        include_health: bool = True
    ) -> Report:
        """
        Generar reporte del sistema.
        
        Args:
            agent: Instancia del agente
            period_hours: Horas del período a reportar
            include_metrics: Incluir métricas
            include_tasks: Incluir estadísticas de tareas
            include_health: Incluir health checks
            
        Returns:
            Reporte generado
        """
        now = datetime.now()
        period_start = now - timedelta(hours=period_hours)
        
        data: Dict[str, Any] = {}
        summary: Dict[str, Any] = {}
        
        # Métricas
        if include_metrics and hasattr(agent, 'metrics'):
            metrics_summary = agent.metrics.get_summary() if hasattr(agent.metrics, 'get_summary') else {}
            data["metrics"] = metrics_summary
            summary["total_operations"] = metrics_summary.get("total_operations", 0)
        
        # Tareas
        if include_tasks:
            status = agent.get_status() if hasattr(agent, 'get_status') else {}
            data["tasks"] = {
                "total": status.get("tasks_total", 0),
                "pending": status.get("tasks_pending", 0),
                "completed": status.get("tasks_completed", 0),
                "failed": status.get("tasks_failed", 0)
            }
            summary["tasks_total"] = status.get("tasks_total", 0)
            summary["success_rate"] = (
                status.get("tasks_completed", 0) / status.get("tasks_total", 1) * 100
                if status.get("tasks_total", 0) > 0 else 0
            )
        
        # Health
        if include_health and hasattr(agent, 'health_checker'):
            health = agent.health_checker.get_health() if hasattr(agent.health_checker, 'get_health') else {}
            data["health"] = health
            summary["health_status"] = health.get("status", "unknown")
        
        report = Report(
            title="System Report",
            generated_at=now,
            period_start=period_start,
            period_end=now,
            data=data,
            summary=summary
        )
        
        self.reports.append(report)
        if len(self.reports) > self.max_reports:
            self.reports = self.reports[-self.max_reports:]
        
        logger.info(f"📊 System report generated: {report.title}")
        return report
    
    def generate_performance_report(
        self,
        performance_analyzer,
        period_hours: int = 24
    ) -> Report:
        """
        Generar reporte de rendimiento.
        
        Args:
            performance_analyzer: Analizador de rendimiento
            period_hours: Horas del período
            
        Returns:
            Reporte generado
        """
        now = datetime.now()
        period_start = now - timedelta(hours=period_hours)
        
        summary = performance_analyzer.get_summary() if hasattr(performance_analyzer, 'get_summary') else {}
        
        report = Report(
            title="Performance Report",
            generated_at=now,
            period_start=period_start,
            period_end=now,
            data={
                "profiles": {
                    name: profile.to_dict()
                    for name, profile in performance_analyzer.get_all_profiles().items()
                }
            },
            summary=summary
        )
        
        self.reports.append(report)
        if len(self.reports) > self.max_reports:
            self.reports = self.reports[-self.max_reports:]
        
        logger.info(f"📊 Performance report generated")
        return report
    
    def get_reports(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Report]:
        """
        Obtener reportes con filtros.
        
        Args:
            since: Filtrar desde fecha
            limit: Límite de resultados
            
        Returns:
            Lista de reportes
        """
        reports = self.reports
        
        if since:
            reports = [r for r in reports if r.generated_at >= since]
        
        # Ordenar por fecha (más recientes primero)
        reports.sort(key=lambda x: x.generated_at, reverse=True)
        
        return reports[:limit]
    
    def export_report(
        self,
        report: Report,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Exportar reporte a archivo.
        
        Args:
            report: Reporte a exportar
            format: Formato (json, markdown)
            output_path: Ruta de salida (opcional)
            
        Returns:
            Contenido del reporte
        """
        if format == "json":
            import json
            content = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str)
        elif format == "markdown":
            content = self._format_report_markdown(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            from pathlib import Path
            Path(output_path).write_text(content, encoding="utf-8")
            logger.info(f"📊 Report exported to {output_path}")
        
        return content
    
    def _format_report_markdown(self, report: Report) -> str:
        """Formatear reporte en Markdown"""
        lines = [
            f"# {report.title}",
            "",
            f"**Generado:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Período:** {report.period_start.strftime('%Y-%m-%d %H:%M:%S')} - {report.period_end.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Resumen",
            ""
        ]
        
        for key, value in report.summary.items():
            lines.append(f"- **{key}:** {value}")
        
        lines.append("")
        lines.append("## Datos")
        lines.append("")
        lines.append("```json")
        import json
        lines.append(json.dumps(report.data, indent=2, ensure_ascii=False, default=str))
        lines.append("```")
        
        return "\n".join(lines)




