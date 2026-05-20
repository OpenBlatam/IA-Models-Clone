"""
Reports - Sistema de Reportes Automatizados
==========================================

Sistema de generación automática de reportes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Tipos de reporte."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class Report:
    """Reporte."""
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    format: str = "json"  # "json", "csv", "pdf", "html"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Generador de reportes."""
    
    def __init__(self):
        self.reports: Dict[str, Report] = {}
        self.scheduled_reports: Dict[str, Dict[str, Any]] = {}
    
    async def generate_report(
        self,
        report_id: str,
        report_type: ReportType,
        title: str,
        period_start: datetime,
        period_end: datetime,
        data_source: Callable,
        format: str = "json",
    ) -> Report:
        """
        Generar reporte.
        
        Args:
            report_id: ID único del reporte
            report_type: Tipo de reporte
            title: Título del reporte
            period_start: Inicio del período
            period_end: Fin del período
            data_source: Función para obtener datos
            format: Formato del reporte
        
        Returns:
            Reporte generado
        """
        logger.info(f"Generating report: {title}")
        
        # Obtener datos
        if asyncio.iscoroutinefunction(data_source):
            data = await data_source(period_start, period_end)
        else:
            data = data_source(period_start, period_end)
        
        report = Report(
            report_id=report_id,
            report_type=report_type,
            title=title,
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            data=data,
            format=format,
        )
        
        self.reports[report_id] = report
        
        logger.info(f"Report {report_id} generated successfully")
        
        return report
    
    async def schedule_report(
        self,
        schedule_id: str,
        report_type: ReportType,
        title: str,
        data_source: Callable,
        schedule_time: Optional[datetime] = None,
    ):
        """Programar reporte."""
        schedule = {
            "schedule_id": schedule_id,
            "report_type": report_type,
            "title": title,
            "data_source": data_source,
            "schedule_time": schedule_time or datetime.now(),
            "next_run": schedule_time or datetime.now(),
        }
        
        self.scheduled_reports[schedule_id] = schedule
        
        logger.info(f"Scheduled report: {schedule_id}")
        
        # Iniciar tarea de ejecución
        asyncio.create_task(self._run_scheduled_reports())
    
    async def _run_scheduled_reports(self):
        """Ejecutar reportes programados."""
        while True:
            try:
                now = datetime.now()
                
                for schedule_id, schedule in self.scheduled_reports.items():
                    if now >= schedule["next_run"]:
                        # Calcular período según tipo
                        period_end = now
                        
                        if schedule["report_type"] == ReportType.DAILY:
                            period_start = period_end - timedelta(days=1)
                        elif schedule["report_type"] == ReportType.WEEKLY:
                            period_start = period_end - timedelta(weeks=1)
                        elif schedule["report_type"] == ReportType.MONTHLY:
                            period_start = period_end - timedelta(days=30)
                        else:
                            period_start = period_end - timedelta(days=1)
                        
                        # Generar reporte
                        report_id = f"{schedule_id}_{now.isoformat()}"
                        await self.generate_report(
                            report_id,
                            schedule["report_type"],
                            schedule["title"],
                            period_start,
                            period_end,
                            schedule["data_source"],
                        )
                        
                        # Calcular próxima ejecución
                        if schedule["report_type"] == ReportType.DAILY:
                            schedule["next_run"] = now + timedelta(days=1)
                        elif schedule["report_type"] == ReportType.WEEKLY:
                            schedule["next_run"] = now + timedelta(weeks=1)
                        elif schedule["report_type"] == ReportType.MONTHLY:
                            schedule["next_run"] = now + timedelta(days=30)
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error in scheduled reports: {e}")
                await asyncio.sleep(60)
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """Obtener reporte."""
        return self.reports.get(report_id)
    
    def list_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Listar reportes."""
        reports = list(self.reports.values())
        
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        
        reports.sort(key=lambda r: r.generated_at, reverse=True)
        
        return [
            {
                "report_id": r.report_id,
                "title": r.title,
                "report_type": r.report_type.value,
                "generated_at": r.generated_at.isoformat(),
                "period_start": r.period_start.isoformat(),
                "period_end": r.period_end.isoformat(),
                "format": r.format,
            }
            for r in reports[:limit]
        ]
































