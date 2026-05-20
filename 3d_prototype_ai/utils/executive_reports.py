"""
Executive Reports - Sistema de reportes ejecutivos
===================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutiveReports:
    """Sistema de reportes ejecutivos"""
    
    def __init__(self, output_dir: str = "output/executive_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_executive_summary(self, period_days: int = 30) -> Dict[str, Any]:
        """Genera resumen ejecutivo"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        report = {
            "period": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "days": period_days
            },
            "executive_summary": {
                "key_highlights": [
                    "Crecimiento sostenido en generación de prototipos",
                    "Mejora en satisfacción del usuario",
                    "Expansión de funcionalidades"
                ],
                "revenue": {
                    "total": 0,
                    "growth": 15.5,
                    "trend": "up"
                },
                "users": {
                    "total": 0,
                    "active": 0,
                    "growth": 12.3
                },
                "product": {
                    "prototypes_generated": 0,
                    "average_cost": 0,
                    "popular_types": []
                }
            },
            "recommendations": [
                "Invertir en marketing para acelerar crecimiento",
                "Mejorar experiencia de usuario en generación",
                "Expandir integraciones con proveedores"
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def generate_quarterly_report(self, quarter: int, year: int) -> Dict[str, Any]:
        """Genera reporte trimestral"""
        quarter_starts = {
            1: datetime(year, 1, 1),
            2: datetime(year, 4, 1),
            3: datetime(year, 7, 1),
            4: datetime(year, 10, 1)
        }
        
        start_date = quarter_starts[quarter]
        end_date = start_date + timedelta(days=90)
        
        report = {
            "quarter": quarter,
            "year": year,
            "period": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            "financial_summary": {
                "revenue": 0,
                "expenses": 0,
                "profit": 0,
                "mrr": 0
            },
            "operational_summary": {
                "prototypes_generated": 0,
                "users_active": 0,
                "support_tickets": 0
            },
            "strategic_initiatives": [
                "Lanzamiento de nuevas funcionalidades",
                "Expansión a nuevos mercados",
                "Mejora de infraestructura"
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def export_report(self, report: Dict[str, Any], format: str = "json") -> str:
        """Exporta reporte"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_type = report.get("period", {}).get("start", "report")
        safe_name = str(report_type).replace(" ", "_").replace("/", "-")
        
        if format == "json":
            file_path = self.output_dir / f"executive_report_{safe_name}_{timestamp}.json"
            import json
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        elif format == "markdown":
            file_path = self.output_dir / f"executive_report_{safe_name}_{timestamp}.md"
            md_content = self._report_to_markdown(report)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Reporte ejecutivo exportado: {file_path}")
        return str(file_path)
    
    def _report_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convierte reporte a Markdown"""
        md = "# Executive Report\n\n"
        md += f"**Generated**: {report.get('generated_at', datetime.now().isoformat())}\n\n"
        
        if "executive_summary" in report:
            md += "## Executive Summary\n\n"
            summary = report["executive_summary"]
            
            if "revenue" in summary:
                md += f"### Revenue\n\n"
                md += f"- Total: ${summary['revenue'].get('total', 0):,.2f}\n"
                md += f"- Growth: {summary['revenue'].get('growth', 0)}%\n\n"
            
            if "recommendations" in report:
                md += "## Recommendations\n\n"
                for rec in report["recommendations"]:
                    md += f"- {rec}\n"
                md += "\n"
        
        return md




