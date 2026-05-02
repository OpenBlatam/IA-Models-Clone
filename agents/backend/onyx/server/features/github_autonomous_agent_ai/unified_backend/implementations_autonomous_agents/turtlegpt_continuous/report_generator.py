"""
Report Generator Module
=======================

Generación de reportes y análisis del agente.
Proporciona funcionalidades para generar reportes comprehensivos del estado y rendimiento.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Tipo de reporte."""
    STATUS = "status"
    PERFORMANCE = "performance"
    METRICS = "metrics"
    TASKS = "tasks"
    MEMORY = "memory"
    COMPREHENSIVE = "comprehensive"


class ReportFormat(Enum):
    """Formato de reporte."""
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


@dataclass
class ReportSection:
    """Sección de un reporte."""
    title: str
    content: Dict[str, Any]
    order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "title": self.title,
            "content": self.content,
            "order": self.order
        }


@dataclass
class Report:
    """Reporte completo."""
    report_type: ReportType
    timestamp: datetime
    agent_name: str
    sections: List[ReportSection]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "report_type": self.report_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "sections": [s.to_dict() for s in sorted(self.sections, key=lambda x: x.order)],
            "metadata": self.metadata
        }


class ReportGenerator:
    """
    Generador de reportes del agente.
    
    Proporciona funcionalidades para:
    - Generar reportes de estado
    - Generar reportes de rendimiento
    - Generar reportes de métricas
    - Exportar reportes en múltiples formatos
    - Análisis y estadísticas
    """
    
    def __init__(self, agent: Any):
        """
        Inicializar generador de reportes.
        
        Args:
            agent: Referencia al agente
        """
        self.agent = agent
        self.reports: List[Report] = []
        self.max_reports = 100
    
    def generate_status_report(self) -> Report:
        """
        Generar reporte de estado.
        
        Returns:
            Reporte de estado
        """
        sections = []
        
        # Estado básico
        if hasattr(self.agent, 'state'):
            sections.append(ReportSection(
                title="Agent State",
                content={
                    "status": str(self.agent.state.status) if hasattr(self.agent.state, 'status') else "unknown",
                    "is_running": getattr(self.agent, 'is_running', False),
                    "should_stop": getattr(self.agent, 'should_stop', False)
                },
                order=1
            ))
        
        # Tareas
        if hasattr(self.agent, 'task_manager'):
            task_info = self._get_task_info()
            sections.append(ReportSection(
                title="Tasks",
                content=task_info,
                order=2
            ))
        
        # Memoria
        if hasattr(self.agent, 'episodic_memory') and hasattr(self.agent, 'semantic_memory'):
            memory_info = self._get_memory_info()
            sections.append(ReportSection(
                title="Memory",
                content=memory_info,
                order=3
            ))
        
        # Componentes
        if hasattr(self.agent, 'component_registry'):
            components_info = self._get_components_info()
            sections.append(ReportSection(
                title="Components",
                content=components_info,
                order=4
            ))
        
        report = Report(
            report_type=ReportType.STATUS,
            timestamp=datetime.now(),
            agent_name=self.agent.name,
            sections=sections
        )
        
        self._add_report(report)
        return report
    
    def generate_performance_report(self) -> Report:
        """
        Generar reporte de rendimiento.
        
        Returns:
            Reporte de rendimiento
        """
        sections = []
        
        # Métricas generales
        if hasattr(self.agent, 'metrics_manager'):
            metrics = self.agent.metrics_manager.get_metrics()
            sections.append(ReportSection(
                title="Metrics",
                content=metrics,
                order=1
            ))
        
        # Rendimiento de tareas
        if hasattr(self.agent, 'task_manager'):
            performance_info = self._get_performance_info()
            sections.append(ReportSection(
                title="Task Performance",
                content=performance_info,
                order=2
            ))
        
        # LLM Calls
        if hasattr(self.agent, 'metrics_manager'):
            llm_info = self._get_llm_info()
            sections.append(ReportSection(
                title="LLM Calls",
                content=llm_info,
                order=3
            ))
        
        report = Report(
            report_type=ReportType.PERFORMANCE,
            timestamp=datetime.now(),
            agent_name=self.agent.name,
            sections=sections
        )
        
        self._add_report(report)
        return report
    
    def generate_comprehensive_report(self) -> Report:
        """
        Generar reporte comprehensivo.
        
        Returns:
            Reporte comprehensivo
        """
        sections = []
        
        # Estado
        status_report = self.generate_status_report()
        sections.extend(status_report.sections)
        
        # Rendimiento
        performance_report = self.generate_performance_report()
        sections.extend(performance_report.sections)
        
        # Análisis
        analysis = self._generate_analysis()
        sections.append(ReportSection(
            title="Analysis",
            content=analysis,
            order=100
        ))
        
        report = Report(
            report_type=ReportType.COMPREHENSIVE,
            timestamp=datetime.now(),
            agent_name=self.agent.name,
            sections=sections,
            metadata={
                "generated_by": "ReportGenerator",
                "version": "1.0"
            }
        )
        
        self._add_report(report)
        return report
    
    def export_report(
        self,
        report: Report,
        file_path: Path,
        format: ReportFormat = ReportFormat.JSON
    ) -> bool:
        """
        Exportar reporte a archivo.
        
        Args:
            report: Reporte a exportar
            file_path: Ruta del archivo
            format: Formato de exportación
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            if format == ReportFormat.JSON:
                return self._export_json(report, file_path)
            elif format == ReportFormat.TEXT:
                return self._export_text(report, file_path)
            elif format == ReportFormat.MARKDOWN:
                return self._export_markdown(report, file_path)
            elif format == ReportFormat.HTML:
                return self._export_html(report, file_path)
            elif format == ReportFormat.CSV:
                return self._export_csv(report, file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting report: {e}", exc_info=True)
            return False
    
    def get_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: Optional[int] = None
    ) -> List[Report]:
        """
        Obtener reportes generados.
        
        Args:
            report_type: Filtrar por tipo
            limit: Límite de resultados
            
        Returns:
            Lista de reportes
        """
        filtered = self.reports
        
        if report_type:
            filtered = [r for r in filtered if r.report_type == report_type]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def _get_task_info(self) -> Dict[str, Any]:
        """Obtener información de tareas."""
        if not hasattr(self.agent, 'task_manager'):
            return {}
        
        try:
            pending = self.agent.task_manager.get_pending_tasks()
            completed = self.agent.task_manager.get_completed_tasks()
            
            return {
                "pending_count": len(pending),
                "completed_count": len(completed),
                "total_tasks": len(pending) + len(completed),
                "recent_tasks": [
                    {
                        "task_id": t.task_id,
                        "description": t.description,
                        "priority": t.priority,
                        "status": t.status.value if hasattr(t.status, 'value') else str(t.status)
                    }
                    for t in (pending + completed)[-10:]
                ]
            }
        except Exception as e:
            logger.warning(f"Error getting task info: {e}")
            return {}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Obtener información de memoria."""
        info = {}
        
        try:
            if hasattr(self.agent, 'episodic_memory'):
                episodes = self.agent.episodic_memory.get_all()
                info["episodic"] = {
                    "count": len(episodes),
                    "recent": [
                        {
                            "content": e.get("content", "")[:100],
                            "timestamp": e.get("timestamp", "")
                        }
                        for e in episodes[-5:]
                    ]
                }
            
            if hasattr(self.agent, 'semantic_memory'):
                # Asumir que semantic_memory tiene método similar
                info["semantic"] = {
                    "available": True
                }
        except Exception as e:
            logger.warning(f"Error getting memory info: {e}")
        
        return info
    
    def _get_components_info(self) -> Dict[str, Any]:
        """Obtener información de componentes."""
        if not hasattr(self.agent, 'component_registry'):
            return {}
        
        try:
            components = self.agent.component_registry.get_all()
            return {
                "total_components": len(components),
                "component_names": list(components.keys())
            }
        except Exception as e:
            logger.warning(f"Error getting components info: {e}")
            return {}
    
    def _get_performance_info(self) -> Dict[str, Any]:
        """Obtener información de rendimiento."""
        info = {}
        
        try:
            if hasattr(self.agent, 'metrics_manager'):
                metrics = self.agent.metrics_manager.get_metrics()
                info.update({
                    "total_tasks_processed": metrics.get("total_tasks_processed", 0),
                    "total_llm_calls": metrics.get("total_llm_calls", 0),
                    "average_response_time": metrics.get("average_response_time", 0.0)
                })
        except Exception as e:
            logger.warning(f"Error getting performance info: {e}")
        
        return info
    
    def _get_llm_info(self) -> Dict[str, Any]:
        """Obtener información de llamadas LLM."""
        info = {}
        
        try:
            if hasattr(self.agent, 'metrics_manager'):
                metrics = self.agent.metrics_manager.get_metrics()
                info = {
                    "total_calls": metrics.get("total_llm_calls", 0),
                    "total_tokens": metrics.get("total_tokens_used", 0),
                    "average_response_time": metrics.get("average_response_time", 0.0),
                    "error_count": metrics.get("error_count", 0)
                }
        except Exception as e:
            logger.warning(f"Error getting LLM info: {e}")
        
        return info
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generar análisis del agente."""
        analysis = {
            "health_status": "healthy",
            "recommendations": [],
            "warnings": []
        }
        
        try:
            # Análisis de tareas
            if hasattr(self.agent, 'task_manager'):
                pending = self.agent.task_manager.get_pending_tasks()
                if len(pending) > 50:
                    analysis["warnings"].append("High number of pending tasks")
                    analysis["recommendations"].append("Consider processing tasks more frequently")
            
            # Análisis de métricas
            if hasattr(self.agent, 'metrics_manager'):
                metrics = self.agent.metrics_manager.get_metrics()
                error_rate = metrics.get("error_count", 0) / max(metrics.get("total_tasks_processed", 1), 1)
                if error_rate > 0.1:
                    analysis["warnings"].append("High error rate detected")
                    analysis["health_status"] = "degraded"
                    analysis["recommendations"].append("Review error logs and improve error handling")
        except Exception as e:
            logger.warning(f"Error generating analysis: {e}")
        
        return analysis
    
    def _export_json(self, report: Report, file_path: Path) -> bool:
        """Exportar a JSON."""
        import json
        data = report.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    
    def _export_text(self, report: Report, file_path: Path) -> bool:
        """Exportar a texto."""
        lines = [
            f"Report: {report.report_type.value.upper()}",
            f"Agent: {report.agent_name}",
            f"Timestamp: {report.timestamp.isoformat()}",
            "=" * 60,
            ""
        ]
        
        for section in sorted(report.sections, key=lambda x: x.order):
            lines.append(f"## {section.title}")
            lines.append("")
            for key, value in section.content.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return True
    
    def _export_markdown(self, report: Report, file_path: Path) -> bool:
        """Exportar a Markdown."""
        lines = [
            f"# {report.report_type.value.upper()} Report",
            "",
            f"**Agent:** {report.agent_name}",
            f"**Timestamp:** {report.timestamp.isoformat()}",
            "",
            "---",
            ""
        ]
        
        for section in sorted(report.sections, key=lambda x: x.order):
            lines.append(f"## {section.title}")
            lines.append("")
            
            # Formatear contenido
            if isinstance(section.content, dict):
                for key, value in section.content.items():
                    if isinstance(value, (dict, list)):
                        lines.append(f"### {key}")
                        lines.append(f"```json")
                        import json
                        lines.append(json.dumps(value, indent=2, default=str))
                        lines.append("```")
                    else:
                        lines.append(f"- **{key}:** {value}")
            else:
                lines.append(str(section.content))
            
            lines.append("")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return True
    
    def _export_html(self, report: Report, file_path: Path) -> bool:
        """Exportar a HTML."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{report.report_type.value.upper()} Report - {report.agent_name}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #333; }",
            "h2 { color: #666; border-bottom: 2px solid #ccc; padding-bottom: 5px; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{report.report_type.value.upper()} Report</h1>",
            f"<p><strong>Agent:</strong> {report.agent_name}</p>",
            f"<p><strong>Timestamp:</strong> {report.timestamp.isoformat()}</p>",
            "<hr>"
        ]
        
        for section in sorted(report.sections, key=lambda x: x.order):
            html.append(f"<h2>{section.title}</h2>")
            
            if isinstance(section.content, dict):
                html.append("<table>")
                for key, value in section.content.items():
                    html.append(f"<tr><th>{key}</th><td>{value}</td></tr>")
                html.append("</table>")
            else:
                html.append(f"<p>{section.content}</p>")
            
            html.append("<br>")
        
        html.extend(["</body>", "</html>"])
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))
        
        return True
    
    def _export_csv(self, report: Report, file_path: Path) -> bool:
        """Exportar a CSV."""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Section", "Key", "Value"])
            
            for section in sorted(report.sections, key=lambda x: x.order):
                if isinstance(section.content, dict):
                    for key, value in section.content.items():
                        writer.writerow([section.title, key, value])
                else:
                    writer.writerow([section.title, "content", section.content])
        
        return True
    
    def _add_report(self, report: Report) -> None:
        """Agregar reporte al historial."""
        self.reports.append(report)
        
        if len(self.reports) > self.max_reports:
            self.reports = self.reports[-self.max_reports:]


def create_report_generator(agent: Any) -> ReportGenerator:
    """
    Factory function para crear ReportGenerator.
    
    Args:
        agent: Referencia al agente
        
    Returns:
        Instancia de ReportGenerator
    """
    return ReportGenerator(agent)
