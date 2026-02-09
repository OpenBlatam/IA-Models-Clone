"""
Generador de Reportes para Control de Calidad
"""

import json
import csv
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import base64

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generador de reportes de inspección de calidad
    """
    
    def __init__(self):
        """Inicializar generador de reportes"""
        logger.info("Report Generator initialized")
    
    def generate_json_report(
        self,
        inspection_result: Dict,
        output_path: str,
        include_images: bool = False
    ) -> bool:
        """
        Generar reporte en formato JSON
        
        Args:
            inspection_result: Resultados de inspección
            output_path: Ruta de salida
            include_images: Incluir imágenes codificadas en base64
            
        Returns:
            True si se generó correctamente
        """
        try:
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "type": "quality_inspection"
                },
                "inspection_results": inspection_result
            }
            
            # Agregar imágenes si se solicita
            if include_images and "image" in inspection_result:
                # Codificar imagen en base64
                import cv2
                import base64
                img = inspection_result["image"]
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                report["inspection_results"]["image_base64"] = img_base64
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}", exc_info=True)
            return False
    
    def generate_csv_report(
        self,
        inspection_results: List[Dict],
        output_path: str
    ) -> bool:
        """
        Generar reporte en formato CSV
        
        Args:
            inspection_results: Lista de resultados de inspección
            output_path: Ruta de salida
            
        Returns:
            True si se generó correctamente
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Encabezados
                writer.writerow([
                    "Timestamp",
                    "Quality Score",
                    "Status",
                    "Objects Detected",
                    "Anomalies Detected",
                    "Defects Detected",
                    "Critical Defects",
                    "Severe Defects",
                    "Moderate Defects",
                    "Minor Defects",
                    "Recommendation"
                ])
                
                # Datos
                for result in inspection_results:
                    if not result.get("success"):
                        continue
                    
                    summary = result.get("summary", {})
                    severity_counts = summary.get("severity_counts", {})
                    
                    writer.writerow([
                        result.get("timestamp", ""),
                        result.get("quality_score", 0),
                        summary.get("status", ""),
                        result.get("objects_detected", 0),
                        result.get("anomalies_detected", 0),
                        result.get("defects_detected", 0),
                        severity_counts.get("critical", 0),
                        severity_counts.get("severe", 0),
                        severity_counts.get("moderate", 0),
                        severity_counts.get("minor", 0),
                        summary.get("recommendation", "")
                    ])
            
            logger.info(f"CSV report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}", exc_info=True)
            return False
    
    def generate_html_report(
        self,
        inspection_result: Dict,
        output_path: str,
        include_charts: bool = True
    ) -> bool:
        """
        Generar reporte en formato HTML
        
        Args:
            inspection_result: Resultados de inspección
            output_path: Ruta de salida
            include_charts: Incluir gráficos
            
        Returns:
            True si se generó correctamente
        """
        try:
            quality_score = inspection_result.get("quality_score", 0)
            summary = inspection_result.get("summary", {})
            defects = inspection_result.get("defects", [])
            
            # Determinar color según score
            if quality_score >= 90:
                color = "#28a745"
                status_icon = "✓"
            elif quality_score >= 75:
                color = "#ffc107"
                status_icon = "⚠"
            elif quality_score >= 60:
                color = "#fd7e14"
                status_icon = "⚠"
            else:
                color = "#dc3545"
                status_icon = "✗"
            
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Quality Inspection Report</title>
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
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid {color};
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid {color};
        }}
        .summary-card h3 {{
            margin-top: 0;
            color: #666;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: {color};
        }}
        .defects-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .defects-table th {{
            background-color: {color};
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .defects-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .defects-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .severity-critical {{ color: #dc3545; font-weight: bold; }}
        .severity-severe {{ color: #fd7e14; font-weight: bold; }}
        .severity-moderate {{ color: #ffc107; }}
        .severity-minor {{ color: #28a745; }}
        .recommendation {{
            background-color: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{status_icon} Quality Inspection Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Quality Score</h3>
                <div class="value">{quality_score:.1f}/100</div>
            </div>
            <div class="summary-card">
                <h3>Status</h3>
                <div class="value">{summary.get('status', 'unknown').upper()}</div>
            </div>
            <div class="summary-card">
                <h3>Defects</h3>
                <div class="value">{len(defects)}</div>
            </div>
            <div class="summary-card">
                <h3>Objects</h3>
                <div class="value">{inspection_result.get('objects_detected', 0)}</div>
            </div>
        </div>
        
        <div class="recommendation">
            <h3>Recommendation</h3>
            <p>{summary.get('recommendation', 'No recommendation available')}</p>
        </div>
        
        <h2>Defects Detected</h2>
        <table class="defects-table">
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                    <th>Area (px²)</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
"""
            
            if defects:
                for defect in defects:
                    severity_class = f"severity-{defect.get('severity', 'minor')}"
                    html += f"""
                <tr>
                    <td>{defect.get('type', 'unknown')}</td>
                    <td class="{severity_class}">{defect.get('severity', 'unknown').upper()}</td>
                    <td>{defect.get('confidence', 0):.2f}</td>
                    <td>{defect.get('area', 0)}</td>
                    <td>{defect.get('description', '')}</td>
                </tr>
"""
            else:
                html += """
                <tr>
                    <td colspan="5" style="text-align: center; color: #28a745;">
                        ✓ No defects detected
                    </td>
                </tr>
"""
            
            html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logger.info(f"HTML report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}", exc_info=True)
            return False
    
    def generate_summary_report(
        self,
        inspection_results: List[Dict],
        output_path: str,
        format: str = "json"
    ) -> bool:
        """
        Generar reporte resumen de múltiples inspecciones
        
        Args:
            inspection_results: Lista de resultados
            output_path: Ruta de salida
            format: Formato ("json", "csv", "html")
            
        Returns:
            True si se generó correctamente
        """
        if format == "json":
            return self.generate_json_report(
                {"batch_results": inspection_results},
                output_path
            )
        elif format == "csv":
            return self.generate_csv_report(inspection_results, output_path)
        elif format == "html":
            # Generar HTML para el primer resultado como ejemplo
            if inspection_results:
                return self.generate_html_report(inspection_results[0], output_path)
            return False
        else:
            logger.error(f"Unknown format: {format}")
            return False






