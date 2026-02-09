"""
Exporters - Exportación de resultados en múltiples formatos
===========================================================
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultExporter:
    """
    Exporta resultados de mejoras en múltiples formatos.
    """
    
    def __init__(self, output_dir: str = "data/exports"):
        """
        Inicializar exportador.
        
        Args:
            output_dir: Directorio de salida
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_json(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Exporta resultados en formato JSON.
        
        Args:
            results: Resultados a exportar
            filename: Nombre del archivo (opcional)
            
        Returns:
            Ruta al archivo exportado
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"improvements_{timestamp}.json"
            
            filepath = self.output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Resultados exportados a JSON: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exportando JSON: {e}")
            raise
    
    def export_markdown(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Exporta resultados en formato Markdown.
        
        Args:
            results: Resultados a exportar
            filename: Nombre del archivo (opcional)
            
        Returns:
            Ruta al archivo exportado
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"improvements_{timestamp}.md"
            
            filepath = self.output_dir / filename
            
            md_content = self._generate_markdown(results)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            logger.info(f"Resultados exportados a Markdown: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exportando Markdown: {e}")
            raise
    
    def export_html(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Exporta resultados en formato HTML.
        
        Args:
            results: Resultados a exportar
            filename: Nombre del archivo (opcional)
            
        Returns:
            Ruta al archivo exportado
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"improvements_{timestamp}.html"
            
            filepath = self.output_dir / filename
            
            html_content = self._generate_html(results)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"Resultados exportados a HTML: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exportando HTML: {e}")
            raise
    
    def _generate_markdown(self, results: Dict[str, Any]) -> str:
        """Genera contenido Markdown desde resultados"""
        lines = [
            "# Code Improvements Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Resumen
        if "summary" in results:
            summary = results["summary"]
            lines.extend([
                "## Summary",
                "",
                f"- **Total Files:** {summary.get('total_files', 0)}",
                f"- **Successful:** {summary.get('successful', 0)}",
                f"- **Failed:** {summary.get('failed', 0)}",
                f"- **Total Improvements:** {summary.get('total_improvements', 0)}",
                ""
            ])
        
        # Mejoras individuales
        if "improvements" in results:
            lines.append("## Improvements")
            lines.append("")
            
            for improvement in results["improvements"]:
                file_info = improvement.get("file_info", {})
                file_path = file_info.get("path", "Unknown")
                
                lines.extend([
                    f"### {file_path}",
                    "",
                    f"**Improvements Applied:** {improvement.get('improvements_applied', 0)}",
                    ""
                ])
                
                if "suggestions" in improvement.get("result", {}):
                    lines.append("**Suggestions:**")
                    for suggestion in improvement["result"]["suggestions"]:
                        lines.append(f"- {suggestion.get('description', '')}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html(self, results: Dict[str, Any]) -> str:
        """Genera contenido HTML desde resultados"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Code Improvements Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .improvement {{ margin: 20px 0; padding: 15px; border-left: 3px solid #4CAF50; }}
        .suggestion {{ margin: 5px 0; padding: 5px; background: #e8f5e9; }}
    </style>
</head>
<body>
    <h1>Code Improvements Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        # Resumen
        if "summary" in results:
            summary = results["summary"]
            html += f"""
    <div class="summary">
        <h2>Summary</h2>
        <ul>
            <li><strong>Total Files:</strong> {summary.get('total_files', 0)}</li>
            <li><strong>Successful:</strong> {summary.get('successful', 0)}</li>
            <li><strong>Failed:</strong> {summary.get('failed', 0)}</li>
            <li><strong>Total Improvements:</strong> {summary.get('total_improvements', 0)}</li>
        </ul>
    </div>
"""
        
        # Mejoras
        if "improvements" in results:
            html += "<h2>Improvements</h2>"
            for improvement in results["improvements"]:
                file_info = improvement.get("file_info", {})
                file_path = file_info.get("path", "Unknown")
                
                html += f"""
    <div class="improvement">
        <h3>{file_path}</h3>
        <p><strong>Improvements Applied:</strong> {improvement.get('improvements_applied', 0)}</p>
"""
                
                if "suggestions" in improvement.get("result", {}):
                    html += "<h4>Suggestions:</h4><ul>"
                    for suggestion in improvement["result"]["suggestions"]:
                        html += f"<li class='suggestion'>{suggestion.get('description', '')}</li>"
                    html += "</ul>"
                
                html += "</div>"
        
        html += """
</body>
</html>
"""
        return html

