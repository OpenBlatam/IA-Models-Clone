"""
Advanced Reporting - Sistema de Reportes Avanzado
=================================================

Genera reportes avanzados y personalizados.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class AdvancedReporting:
    """Sistema de reportes avanzado"""

    def __init__(self, reports_dir: Path = None):
        """
        Inicializa el sistema de reportes.

        Args:
            reports_dir: Directorio para almacenar reportes
        """
        if reports_dir is None:
            reports_dir = Path("reports")
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_project_report(
        self,
        project_id: str,
        project_info: Dict[str, Any],
        include_stats: bool = True,
        include_timeline: bool = True,
    ) -> Dict[str, Any]:
        """
        Genera un reporte completo de proyecto.

        Args:
            project_id: ID del proyecto
            project_info: Información del proyecto
            include_stats: Incluir estadísticas
            include_timeline: Incluir timeline

        Returns:
            Reporte completo
        """
        report = {
            "project_id": project_id,
            "generated_at": datetime.now().isoformat(),
            "project_info": project_info,
        }

        if include_stats:
            report["statistics"] = {
                "features_count": len(project_info.get("features", [])),
                "backend_framework": project_info.get("backend_framework"),
                "frontend_framework": project_info.get("frontend_framework"),
                "ai_type": project_info.get("ai_type"),
            }

        if include_timeline:
            report["timeline"] = {
                "created_at": project_info.get("created_at", datetime.now().isoformat()),
                "generated_at": datetime.now().isoformat(),
            }

        # Guardar reporte
        report_file = self.reports_dir / f"project_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        logger.info(f"Reporte generado para proyecto {project_id}")
        return report

    def generate_system_report(
        self,
        time_period: str = "daily",
        include_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Genera un reporte del sistema.

        Args:
            time_period: Período (daily, weekly, monthly)
            include_metrics: Incluir métricas

        Returns:
            Reporte del sistema
        """
        report = {
            "report_type": "system",
            "time_period": time_period,
            "generated_at": datetime.now().isoformat(),
        }

        if include_metrics:
            report["metrics"] = {
                "total_projects": 0,  # Se llenaría con datos reales
                "success_rate": 0.0,
                "average_generation_time": 0.0,
            }

        # Guardar reporte
        report_file = self.reports_dir / f"system_{time_period}_{datetime.now().strftime('%Y%m%d')}.json"
        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return report

    def list_reports(
        self,
        report_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Lista reportes generados.

        Args:
            report_type: Tipo de reporte (opcional)
            limit: Límite de resultados

        Returns:
            Lista de reportes
        """
        reports = []
        for report_file in sorted(self.reports_dir.glob("*.json"), reverse=True):
            try:
                report_data = json.loads(report_file.read_text(encoding="utf-8"))
                if not report_type or report_data.get("report_type") == report_type:
                    reports.append({
                        "file": report_file.name,
                        "generated_at": report_data.get("generated_at"),
                        "report_type": report_data.get("report_type", "project"),
                    })
            except Exception as e:
                logger.error(f"Error leyendo reporte {report_file}: {e}")

        return reports[:limit]


