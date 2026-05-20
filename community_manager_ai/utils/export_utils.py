"""
Export Utils - Utilidades de Exportación
=========================================

Utilidades para exportar datos y reportes.
"""

import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExportUtils:
    """Utilidades para exportar datos"""
    
    @staticmethod
    def export_posts_to_csv(
        posts: List[Dict[str, Any]],
        file_path: str
    ) -> bool:
        """
        Exportar posts a CSV
        
        Args:
            posts: Lista de posts
            file_path: Ruta del archivo
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            if not posts:
                logger.warning("No hay posts para exportar")
                return False
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "id", "content", "platforms", "scheduled_time",
                    "status", "created_at", "tags"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for post in posts:
                    row = {
                        "id": post.get("id", ""),
                        "content": post.get("content", "")[:100],  # Truncar
                        "platforms": ",".join(post.get("platforms", [])),
                        "scheduled_time": post.get("scheduled_time", ""),
                        "status": post.get("status", ""),
                        "created_at": post.get("created_at", ""),
                        "tags": ",".join(post.get("tags", []))
                    }
                    writer.writerow(row)
            
            logger.info(f"Posts exportados a {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando a CSV: {e}")
            return False
    
    @staticmethod
    def export_analytics_to_json(
        analytics: Dict[str, Any],
        file_path: str
    ) -> bool:
        """
        Exportar analytics a JSON
        
        Args:
            analytics: Datos de analytics
            file_path: Ruta del archivo
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "analytics": analytics
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analytics exportados a {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando analytics: {e}")
            return False
    
    @staticmethod
    def export_calendar_to_ical(
        events: List[Dict[str, Any]],
        file_path: str
    ) -> bool:
        """
        Exportar calendario a formato iCal
        
        Args:
            events: Lista de eventos
            file_path: Ruta del archivo
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            ical_content = "BEGIN:VCALENDAR\n"
            ical_content += "VERSION:2.0\n"
            ical_content += "PRODID:-//Community Manager AI//EN\n"
            
            for event in events:
                scheduled_time = event.get("scheduled_time")
                if not scheduled_time:
                    continue
                
                if isinstance(scheduled_time, str):
                    scheduled_time = datetime.fromisoformat(scheduled_time)
                
                # Formatear para iCal
                dt_start = scheduled_time.strftime("%Y%m%dT%H%M%S")
                dt_end = (scheduled_time.replace(hour=scheduled_time.hour + 1)).strftime("%Y%m%dT%H%M%S")
                
                ical_content += "BEGIN:VEVENT\n"
                ical_content += f"DTSTART:{dt_start}\n"
                ical_content += f"DTEND:{dt_end}\n"
                ical_content += f"SUMMARY:{event.get('content', 'Post')[:50]}\n"
                ical_content += f"DESCRIPTION:{event.get('content', '')}\n"
                ical_content += "END:VEVENT\n"
            
            ical_content += "END:VCALENDAR\n"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(ical_content)
            
            logger.info(f"Calendario exportado a {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando calendario: {e}")
            return False
    
    @staticmethod
    def generate_report(
        posts: List[Dict[str, Any]],
        analytics: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None
    ) -> str:
        """
        Generar reporte en texto
        
        Args:
            posts: Lista de posts
            analytics: Datos de analytics (opcional)
            file_path: Ruta para guardar (opcional)
            
        Returns:
            Contenido del reporte
        """
        report = []
        report.append("=" * 60)
        report.append("REPORTE DE COMMUNITY MANAGER AI")
        report.append("=" * 60)
        report.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Estadísticas de posts
        report.append("ESTADÍSTICAS DE POSTS")
        report.append("-" * 60)
        report.append(f"Total de posts: {len(posts)}")
        
        status_counts = {}
        platform_counts = {}
        
        for post in posts:
            status = post.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            for platform in post.get("platforms", []):
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        report.append("\nPor estado:")
        for status, count in status_counts.items():
            report.append(f"  {status}: {count}")
        
        report.append("\nPor plataforma:")
        for platform, count in platform_counts.items():
            report.append(f"  {platform}: {count}")
        
        # Analytics si está disponible
        if analytics:
            report.append("\n" + "=" * 60)
            report.append("ANALYTICS")
            report.append("-" * 60)
            report.append(json.dumps(analytics, indent=2, ensure_ascii=False))
        
        report.append("\n" + "=" * 60)
        
        report_content = "\n".join(report)
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Reporte guardado en {file_path}")
        
        return report_content
    
    @staticmethod
    def export_to_excel(
        data: List[Dict[str, Any]],
        file_path: str,
        sheet_name: str = "Data"
    ) -> bool:
        """
        Exportar datos a Excel
        
        Args:
            data: Lista de diccionarios con datos
            file_path: Ruta del archivo
            sheet_name: Nombre de la hoja
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            try:
                import openpyxl
                from openpyxl import Workbook
            except ImportError:
                logger.error("openpyxl no está instalado. Instala con: pip install openpyxl")
                return False
            
            wb = Workbook()
            ws = wb.active
            ws.title = sheet_name
            
            if not data:
                logger.warning("No hay datos para exportar")
                return False
            
            headers = list(data[0].keys())
            ws.append(headers)
            
            for row in data:
                values = [row.get(header, "") for header in headers]
                ws.append(values)
            
            wb.save(file_path)
            logger.info(f"Datos exportados a Excel: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando a Excel: {e}")
            return False
    
    @staticmethod
    def export_to_pdf(
        content: str,
        file_path: str,
        title: str = "Report"
    ) -> bool:
        """
        Exportar contenido a PDF
        
        Args:
            content: Contenido a exportar
            file_path: Ruta del archivo
            title: Título del documento
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet
            except ImportError:
                logger.error("reportlab no está instalado. Instala con: pip install reportlab")
                return False
            
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            story.append(Paragraph(title, styles['Title']))
            story.append(Spacer(1, 12))
            
            for line in content.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            logger.info(f"Contenido exportado a PDF: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando a PDF: {e}")
            return False
    
    @staticmethod
    def export_analytics_to_csv(
        analytics: Dict[str, Any],
        file_path: str
    ) -> bool:
        """
        Exportar analytics a CSV
        
        Args:
            analytics: Datos de analytics
            file_path: Ruta del archivo
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                writer.writerow(["Metric", "Value"])
                
                for key, value in analytics.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            writer.writerow([f"{key}.{sub_key}", sub_value])
                    else:
                        writer.writerow([key, value])
            
            logger.info(f"Analytics exportados a CSV: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando analytics a CSV: {e}")
            return False



