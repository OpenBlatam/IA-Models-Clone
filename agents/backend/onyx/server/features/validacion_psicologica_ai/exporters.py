"""
Exportadores de Reportes para Validación Psicológica AI
========================================================
Exportación a diferentes formatos (PDF, Excel, JSON)
"""

from typing import Dict, Any, Optional
from datetime import datetime
from io import BytesIO
import structlog
import json

from .models import ValidationReport, PsychologicalProfile

logger = structlog.get_logger()


class ReportExporter:
    """Exportador base de reportes"""
    
    @staticmethod
    def export_to_json(
        report: ValidationReport,
        profile: Optional[PsychologicalProfile] = None
    ) -> str:
        """
        Exportar reporte a JSON
        
        Args:
            report: Reporte a exportar
            profile: Perfil psicológico (opcional)
            
        Returns:
            JSON string
        """
        data = {
            "report": report.to_dict(),
            "profile": profile.to_dict() if profile else None,
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_to_text(
        report: ValidationReport,
        profile: Optional[PsychologicalProfile] = None
    ) -> str:
        """
        Exportar reporte a texto plano
        
        Args:
            report: Reporte a exportar
            profile: Perfil psicológico (opcional)
            
        Returns:
            Texto formateado
        """
        lines = []
        lines.append("=" * 80)
        lines.append("REPORTE DE VALIDACIÓN PSICOLÓGICA")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Fecha de generación: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Resumen
        lines.append("RESUMEN EJECUTIVO")
        lines.append("-" * 80)
        lines.append(report.summary)
        lines.append("")
        
        # Perfil psicológico
        if profile:
            lines.append("PERFIL PSICOLÓGICO")
            lines.append("-" * 80)
            
            lines.append("\nRasgos de Personalidad:")
            for trait, score in profile.personality_traits.items():
                lines.append(f"  - {trait.capitalize()}: {score:.2f}")
            
            lines.append("\nEstado Emocional:")
            for key, value in profile.emotional_state.items():
                if isinstance(value, float):
                    lines.append(f"  - {key.capitalize()}: {value:.2f}")
                else:
                    lines.append(f"  - {key.capitalize()}: {value}")
            
            lines.append(f"\nConfianza del Análisis: {profile.confidence_score * 100:.1f}%")
            lines.append("")
        
        # Análisis detallado
        lines.append("ANÁLISIS DETALLADO")
        lines.append("-" * 80)
        
        if report.sentiment_analysis:
            lines.append("\nAnálisis de Sentimientos:")
            sentiment = report.sentiment_analysis
            lines.append(f"  - Sentimiento general: {sentiment.get('overall_sentiment', 'N/A')}")
            if 'sentiment_distribution' in sentiment:
                dist = sentiment['sentiment_distribution']
                lines.append(f"  - Positivo: {dist.get('positive', 0):.1%}")
                lines.append(f"  - Neutral: {dist.get('neutral', 0):.1%}")
                lines.append(f"  - Negativo: {dist.get('negative', 0):.1%}")
        
        # Insights por plataforma
        if report.social_media_insights:
            lines.append("\nInsights por Plataforma:")
            for platform, insights in report.social_media_insights.items():
                lines.append(f"\n  {platform.upper()}:")
                lines.append(f"    - Posts: {insights.get('post_count', 0)}")
                lines.append(f"    - Engagement rate: {insights.get('engagement_rate', 0):.1%}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"Generado el {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def export_to_html(
        report: ValidationReport,
        profile: Optional[PsychologicalProfile] = None
    ) -> str:
        """
        Exportar reporte a HTML
        
        Args:
            report: Reporte a exportar
            profile: Perfil psicológico (opcional)
            
        Returns:
            HTML string
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reporte de Validación Psicológica</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #f9f9f9;
            padding: 20px;
            border-left: 4px solid #4CAF50;
            margin: 20px 0;
        }}
        .trait {{
            display: inline-block;
            margin: 10px;
            padding: 10px 20px;
            background-color: #e3f2fd;
            border-radius: 5px;
        }}
        .metric {{
            margin: 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #888;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Reporte de Validación Psicológica</h1>
        <p><strong>Fecha de generación:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Resumen Ejecutivo</h2>
            <p>{report.summary.replace(chr(10), '<br>')}</p>
        </div>
"""
        
        if profile:
            html += f"""
        <h2>Perfil Psicológico</h2>
        <div class="metric">
            <span class="metric-label">Confianza del Análisis:</span>
            <span>{profile.confidence_score * 100:.1f}%</span>
        </div>
        
        <h3>Rasgos de Personalidad</h3>
        <div>
"""
            for trait, score in profile.personality_traits.items():
                html += f'<div class="trait"><strong>{trait.capitalize()}</strong>: {score:.2f}</div>'
            
            html += """
        </div>
        
        <h3>Estado Emocional</h3>
        <table>
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
            </tr>
"""
            for key, value in profile.emotional_state.items():
                if isinstance(value, float):
                    html += f'<tr><td>{key.capitalize()}</td><td>{value:.2f}</td></tr>'
                else:
                    html += f'<tr><td>{key.capitalize()}</td><td>{value}</td></tr>'
            
            html += """
        </table>
"""
        
        html += f"""
        <h2>Análisis de Sentimientos</h2>
        <table>
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
            </tr>
"""
        
        if report.sentiment_analysis:
            sentiment = report.sentiment_analysis
            html += f'<tr><td>Sentimiento General</td><td>{sentiment.get("overall_sentiment", "N/A")}</td></tr>'
            if 'sentiment_distribution' in sentiment:
                dist = sentiment['sentiment_distribution']
                html += f'<tr><td>Positivo</td><td>{dist.get("positive", 0):.1%}</td></tr>'
                html += f'<tr><td>Neutral</td><td>{dist.get("neutral", 0):.1%}</td></tr>'
                html += f'<tr><td>Negativo</td><td>{dist.get("negative", 0):.1%}</td></tr>'
        
        html += """
        </table>
        
        <h2>Insights por Plataforma</h2>
        <table>
            <tr>
                <th>Plataforma</th>
                <th>Posts</th>
                <th>Engagement Rate</th>
            </tr>
"""
        
        for platform, insights in report.social_media_insights.items():
            html += f"""
            <tr>
                <td>{platform.upper()}</td>
                <td>{insights.get('post_count', 0)}</td>
                <td>{insights.get('engagement_rate', 0):.1%}</td>
            </tr>
"""
        
        html += f"""
        </table>
        
        <div class="footer">
            <p>Generado el {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Validación Psicológica AI v1.1.0</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    @staticmethod
    def export_to_csv_data(
        report: ValidationReport,
        profile: Optional[PsychologicalProfile] = None
    ) -> str:
        """
        Exportar datos a formato CSV
        
        Args:
            report: Reporte a exportar
            profile: Perfil psicológico (opcional)
            
        Returns:
            CSV string
        """
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Encabezados
        writer.writerow(["Métrica", "Valor"])
        writer.writerow(["Fecha de generación", report.generated_at.strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])
        
        if profile:
            writer.writerow(["PERFIL PSICOLÓGICO"])
            writer.writerow(["Rasgo", "Score"])
            for trait, score in profile.personality_traits.items():
                writer.writerow([trait, f"{score:.2f}"])
            writer.writerow([])
            
            writer.writerow(["Estado Emocional", "Valor"])
            for key, value in profile.emotional_state.items():
                writer.writerow([key, value])
            writer.writerow([])
        
        writer.writerow(["ANÁLISIS DE SENTIMIENTOS"])
        if report.sentiment_analysis:
            sentiment = report.sentiment_analysis
            writer.writerow(["Sentimiento General", sentiment.get("overall_sentiment", "N/A")])
            if 'sentiment_distribution' in sentiment:
                dist = sentiment['sentiment_distribution']
                writer.writerow(["Positivo", f"{dist.get('positive', 0):.2%}"])
                writer.writerow(["Neutral", f"{dist.get('neutral', 0):.2%}"])
                writer.writerow(["Negativo", f"{dist.get('negative', 0):.2%}"])
        
        return output.getvalue()


class PDFExporter:
    """Exportador a PDF (requiere reportlab)"""
    
    @staticmethod
    def export_to_pdf(
        report: ValidationReport,
        profile: Optional[PsychologicalProfile] = None,
        output_path: Optional[str] = None
    ) -> BytesIO:
        """
        Exportar reporte a PDF
        
        Args:
            report: Reporte a exportar
            profile: Perfil psicológico (opcional)
            output_path: Ruta de salida (opcional)
            
        Returns:
            BytesIO con el PDF
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Título
            title = Paragraph("Reporte de Validación Psicológica", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 0.2 * inch))
            
            # Fecha
            date_text = f"Fecha de generación: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(date_text, styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))
            
            # Resumen
            story.append(Paragraph("Resumen Ejecutivo", styles['Heading2']))
            summary_para = Paragraph(report.summary.replace('\n', '<br/>'), styles['Normal'])
            story.append(summary_para)
            story.append(Spacer(1, 0.3 * inch))
            
            # Perfil psicológico
            if profile:
                story.append(Paragraph("Perfil Psicológico", styles['Heading2']))
                
                # Rasgos de personalidad
                trait_data = [["Rasgo", "Score"]]
                for trait, score in profile.personality_traits.items():
                    trait_data.append([trait.capitalize(), f"{score:.2f}"])
                
                trait_table = Table(trait_data)
                trait_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(trait_table)
                story.append(Spacer(1, 0.3 * inch))
            
            # Construir PDF
            doc.build(story)
            buffer.seek(0)
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(buffer.read())
                buffer.seek(0)
            
            return buffer
            
        except ImportError:
            logger.warning("reportlab not installed, PDF export not available")
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")
        except Exception as e:
            logger.error("Error generating PDF", error=str(e))
            raise




