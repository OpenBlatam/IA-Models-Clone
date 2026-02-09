"""
Servicio de generación de reportes en PDF y otros formatos
"""

from typing import Dict, List, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io


class ReportService:
    """Servicio de generación de reportes"""
    
    def __init__(self):
        """Inicializa el servicio de reportes"""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Configura estilos personalizados"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12
        ))
    
    def generate_pdf_report(
        self,
        user_id: str,
        user_data: Dict,
        progress_data: Dict,
        analytics_data: Optional[Dict] = None
    ) -> bytes:
        """
        Genera un reporte PDF completo
        
        Args:
            user_id: ID del usuario
            user_data: Datos del usuario
            progress_data: Datos de progreso
            analytics_data: Datos de análisis (opcional)
        
        Returns:
            Bytes del PDF generado
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Título
        title = Paragraph("Reporte de Recuperación", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Información del usuario
        story.append(Paragraph("Información del Usuario", self.styles['SectionHeader']))
        user_info = [
            ["ID de Usuario:", user_id],
            ["Fecha del Reporte:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]
        if user_data.get("name"):
            user_info.append(["Nombre:", user_data["name"]])
        if user_data.get("email"):
            user_info.append(["Email:", user_data["email"]])
        
        user_table = Table(user_info, colWidths=[2*inch, 4*inch])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(user_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Progreso
        story.append(Paragraph("Progreso de Recuperación", self.styles['SectionHeader']))
        progress_info = [
            ["Días de Sobriedad:", str(progress_data.get("days_sober", 0))],
            ["Tasa de Éxito:", f"{progress_data.get('success_rate', 0):.2f}%"],
            ["Racha Actual:", f"{progress_data.get('streak_days', 0)} días"],
        ]
        
        if progress_data.get("time_sober_formatted"):
            progress_info.append(["Tiempo Total:", progress_data["time_sober_formatted"]])
        
        progress_table = Table(progress_info, colWidths=[2*inch, 4*inch])
        progress_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5e9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(progress_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Análisis (si está disponible)
        if analytics_data:
            story.append(Paragraph("Análisis Detallado", self.styles['SectionHeader']))
            
            trends = analytics_data.get("trends", {})
            if trends:
                trend_text = f"Tendencia de Consumo: {trends.get('consumption_trend', 'N/A')}"
                story.append(Paragraph(trend_text, self.styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            recommendations = analytics_data.get("recommendations", [])
            if recommendations:
                story.append(Paragraph("Recomendaciones:", self.styles['Heading3']))
                for rec in recommendations[:5]:  # Limitar a 5
                    story.append(Paragraph(f"• {rec}", self.styles['Normal']))
                    story.append(Spacer(1, 0.05*inch))
        
        # Mensaje motivacional
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Mensaje Final", self.styles['SectionHeader']))
        motivational_text = (
            "Continúa con tu compromiso de recuperación. Cada día es un paso hacia una vida mejor. "
            "Recuerda que no estás solo en este viaje y que cada momento de resistencia te hace más fuerte."
        )
        story.append(Paragraph(motivational_text, self.styles['Normal']))
        
        # Construir PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_summary_report(self, user_id: str, data: Dict) -> Dict:
        """
        Genera un reporte resumido en formato JSON
        
        Args:
            user_id: ID del usuario
            data: Datos del usuario
        
        Returns:
            Reporte resumido
        """
        return {
            "user_id": user_id,
            "report_type": "summary",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "days_sober": data.get("days_sober", 0),
                "success_rate": data.get("success_rate", 0),
                "current_streak": data.get("streak_days", 0),
                "milestones_achieved": data.get("milestones_achieved", []),
                "key_achievements": data.get("achievements", [])
            },
            "recommendations": data.get("recommendations", [])
        }

