"""
Generador de reportes en PDF y JSON
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import io

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportGenerator:
    """Genera reportes de análisis en diferentes formatos"""
    
    def __init__(self):
        """Inicializa el generador de reportes"""
        self.reportlab_available = REPORTLAB_AVAILABLE
    
    def generate_json_report(self, analysis_result: Dict, 
                           recommendations: Optional[Dict] = None,
                           comparison: Optional[Dict] = None) -> str:
        """
        Genera reporte en formato JSON
        
        Args:
            analysis_result: Resultado del análisis
            recommendations: Recomendaciones (opcional)
            comparison: Comparación con análisis previo (opcional)
            
        Returns:
            JSON string del reporte
        """
        report = {
            "report_info": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.1.0"
            },
            "analysis": analysis_result,
            "recommendations": recommendations,
            "comparison": comparison
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)
    
    def generate_pdf_report(self, analysis_result: Dict,
                          recommendations: Optional[Dict] = None,
                          comparison: Optional[Dict] = None,
                          output_path: Optional[str] = None) -> bytes:
        """
        Genera reporte en formato PDF
        
        Args:
            analysis_result: Resultado del análisis
            recommendations: Recomendaciones (opcional)
            comparison: Comparación con análisis previo (opcional)
            output_path: Path para guardar PDF (opcional)
            
        Returns:
            Bytes del PDF
        """
        if not self.reportlab_available:
            raise ImportError(
                "reportlab no está instalado. Instale con: pip install reportlab"
            )
        
        # Crear buffer o archivo
        if output_path:
            buffer = io.BytesIO()
            use_file = True
        else:
            buffer = io.BytesIO()
            use_file = False
        
        # Crear documento
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Contenedor para elementos
        story = []
        styles = getSampleStyleSheet()
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Reporte de Análisis de Piel", title_style))
        story.append(Spacer(1, 12))
        
        # Información del reporte
        info_style = ParagraphStyle(
            'InfoStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        )
        story.append(Paragraph(
            f"Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            info_style
        ))
        story.append(Spacer(1, 20))
        
        # Resumen ejecutivo
        story.append(Paragraph("Resumen Ejecutivo", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        quality_scores = analysis_result.get("quality_scores", {})
        overall_score = quality_scores.get("overall_score", 0)
        skin_type = analysis_result.get("skin_type", "unknown")
        
        summary_text = f"""
        <b>Score General:</b> {overall_score:.1f}/100<br/>
        <b>Tipo de Piel:</b> {skin_type.title()}<br/>
        <b>Condiciones Detectadas:</b> {len(analysis_result.get('conditions', []))}
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Métricas de calidad
        story.append(Paragraph("Métricas de Calidad", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Tabla de métricas
        metrics_data = [["Métrica", "Score", "Estado"]]
        for key, value in quality_scores.items():
            if key != "overall_score":
                metric_name = key.replace("_", " ").title()
                status = "Excelente" if value >= 80 else "Bueno" if value >= 60 else "Regular" if value >= 40 else "Necesita Mejora"
                metrics_data.append([metric_name, f"{value:.1f}", status])
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Condiciones detectadas
        conditions = analysis_result.get("conditions", [])
        if conditions:
            story.append(Paragraph("Condiciones Detectadas", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for condition in conditions:
                cond_text = f"""
                <b>{condition.get('name', 'Unknown').title()}</b><br/>
                Severidad: {condition.get('severity', 'unknown').title()}<br/>
                Confianza: {condition.get('confidence', 0)*100:.1f}%<br/>
                Área afectada: {condition.get('affected_area_percentage', 0):.1f}%<br/>
                <i>{condition.get('description', '')}</i>
                """
                story.append(Paragraph(cond_text, styles['Normal']))
                story.append(Spacer(1, 12))
        
        # Recomendaciones
        if recommendations:
            story.append(PageBreak())
            story.append(Paragraph("Recomendaciones de Skincare", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            routine = recommendations.get("routine", {})
            
            # Rutina de la mañana
            if routine.get("morning"):
                story.append(Paragraph("Rutina de la Mañana", styles['Heading3']))
                for i, product in enumerate(routine["morning"], 1):
                    product_text = f"""
                    <b>{i}. {product.get('name', 'Producto')}</b><br/>
                    Categoría: {product.get('category', '').title()}<br/>
                    Uso: {product.get('usage_frequency', '')}<br/>
                    Ingredientes clave: {', '.join(product.get('key_ingredients', []))}
                    """
                    story.append(Paragraph(product_text, styles['Normal']))
                    story.append(Spacer(1, 8))
            
            # Rutina de la noche
            if routine.get("evening"):
                story.append(Spacer(1, 12))
                story.append(Paragraph("Rutina de la Noche", styles['Heading3']))
                for i, product in enumerate(routine["evening"], 1):
                    product_text = f"""
                    <b>{i}. {product.get('name', 'Producto')}</b><br/>
                    Categoría: {product.get('category', '').title()}<br/>
                    Uso: {product.get('usage_frequency', '')}
                    """
                    story.append(Paragraph(product_text, styles['Normal']))
                    story.append(Spacer(1, 8))
            
            # Tips
            tips = recommendations.get("tips", [])
            if tips:
                story.append(Spacer(1, 12))
                story.append(Paragraph("Consejos Generales", styles['Heading3']))
                for tip in tips:
                    story.append(Paragraph(f"• {tip}", styles['Normal']))
                    story.append(Spacer(1, 4))
        
        # Comparación
        if comparison:
            story.append(PageBreak())
            story.append(Paragraph("Comparación con Análisis Previo", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            score_diffs = comparison.get("score_differences", {})
            if score_diffs:
                comp_data = [["Métrica", "Antes", "Después", "Diferencia"]]
                for key, diff in score_diffs.items():
                    metric_name = key.replace("_", " ").title()
                    improvement = "↑" if diff["improvement"] else "↓"
                    comp_data.append([
                        metric_name,
                        f"{diff['before']:.1f}",
                        f"{diff['after']:.1f}",
                        f"{improvement} {abs(diff['difference']):.1f}"
                    ])
                
                comp_table = Table(comp_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                comp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                story.append(comp_table)
        
        # Construir PDF
        doc.build(story)
        
        # Retornar bytes
        buffer.seek(0)
        pdf_bytes = buffer.read()
        
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
        
        return pdf_bytes
    
    def generate_html_report(self, analysis_result: Dict,
                           recommendations: Optional[Dict] = None,
                           comparison: Optional[Dict] = None) -> str:
        """
        Genera reporte en formato HTML
        
        Args:
            analysis_result: Resultado del análisis
            recommendations: Recomendaciones (opcional)
            comparison: Comparación (opcional)
            
        Returns:
            HTML string del reporte
        """
        quality_scores = analysis_result.get("quality_scores", {})
        overall_score = quality_scores.get("overall_score", 0)
        skin_type = analysis_result.get("skin_type", "unknown")
        conditions = analysis_result.get("conditions", [])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Análisis de Piel</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .score {{ font-size: 48px; color: #3498db; font-weight: bold; text-align: center; margin: 20px 0; }}
                .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0; }}
                .metric {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
                .metric-name {{ font-weight: bold; color: #34495e; }}
                .metric-value {{ font-size: 24px; color: #3498db; }}
                .condition {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; border-radius: 3px; }}
                .product {{ background: #e8f5e9; padding: 15px; margin: 10px 0; border-left: 4px solid #4caf50; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Reporte de Análisis de Piel</h1>
                <p style="color: #7f8c8d;">Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                
                <div class="score">{overall_score:.1f}/100</div>
                <p style="text-align: center; font-size: 18px; color: #34495e;">
                    Tipo de Piel: <strong>{skin_type.title()}</strong>
                </p>
                
                <h2>Métricas de Calidad</h2>
                <div class="metrics">
        """
        
        for key, value in quality_scores.items():
            if key != "overall_score":
                metric_name = key.replace("_", " ").title()
                html += f"""
                    <div class="metric">
                        <div class="metric-name">{metric_name}</div>
                        <div class="metric-value">{value:.1f}</div>
                    </div>
                """
        
        html += """
                </div>
        """
        
        if conditions:
            html += """
                <h2>Condiciones Detectadas</h2>
            """
            for condition in conditions:
                html += f"""
                <div class="condition">
                    <strong>{condition.get('name', 'Unknown').title()}</strong><br/>
                    Severidad: {condition.get('severity', 'unknown').title()} | 
                    Confianza: {condition.get('confidence', 0)*100:.1f}%<br/>
                    <em>{condition.get('description', '')}</em>
                </div>
                """
        
        if recommendations:
            html += """
                <h2>Recomendaciones</h2>
            """
            routine = recommendations.get("routine", {})
            if routine.get("morning"):
                html += "<h3>Rutina de la Mañana</h3>"
                for product in routine["morning"]:
                    html += f"""
                    <div class="product">
                        <strong>{product.get('name', 'Producto')}</strong><br/>
                        {product.get('description', '')}<br/>
                        <small>Uso: {product.get('usage_frequency', '')}</small>
                    </div>
                    """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html






