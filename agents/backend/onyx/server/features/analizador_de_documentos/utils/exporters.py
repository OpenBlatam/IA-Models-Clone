"""
Exportadores de Resultados
==========================

Sistema para exportar resultados de análisis en múltiples formatos.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultExporter:
    """Exportador de resultados en múltiples formatos"""
    
    @staticmethod
    def export_json(
        data: Any,
        output_path: str,
        indent: int = 2,
        ensure_ascii: bool = False
    ):
        """
        Exportar a JSON
        
        Args:
            data: Datos a exportar
            output_path: Ruta de salida
            indent: Indentación
            ensure_ascii: Si True, escapa caracteres no ASCII
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        logger.info(f"Resultados exportados a JSON: {output_path}")
    
    @staticmethod
    def export_csv(
        data: List[Dict[str, Any]],
        output_path: str,
        fieldnames: Optional[List[str]] = None
    ):
        """
        Exportar a CSV
        
        Args:
            data: Lista de diccionarios
            output_path: Ruta de salida
            fieldnames: Nombres de columnas (opcional)
        """
        if not data:
            logger.warning("No hay datos para exportar")
            return
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Determinar fieldnames
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in data:
                # Flatten nested dicts
                flattened_row = ResultExporter._flatten_dict(row)
                writer.writerow(flattened_row)
        
        logger.info(f"Resultados exportados a CSV: {output_path}")
    
    @staticmethod
    def export_markdown(
        data: Dict[str, Any],
        output_path: str,
        title: Optional[str] = None
    ):
        """
        Exportar a Markdown
        
        Args:
            data: Datos a exportar
            output_path: Ruta de salida
            title: Título del documento
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        lines = []
        
        if title:
            lines.append(f"# {title}\n")
            lines.append(f"*Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        lines.extend(ResultExporter._dict_to_markdown(data))
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Resultados exportados a Markdown: {output_path}")
    
    @staticmethod
    def export_html(
        data: Dict[str, Any],
        output_path: str,
        title: Optional[str] = None
    ):
        """
        Exportar a HTML
        
        Args:
            data: Datos a exportar
            output_path: Ruta de salida
            title: Título del documento
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        html = ["<!DOCTYPE html>", "<html>", "<head>"]
        html.append("<meta charset='utf-8'>")
        html.append(f"<title>{title or 'Análisis de Documento'}</title>")
        html.append("""
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .metadata { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }
        </style>
        """)
        html.append("</head>")
        html.append("<body>")
        
        if title:
            html.append(f"<h1>{title}</h1>")
        
        html.append(f"<p><em>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")
        
        html.extend(ResultExporter._dict_to_html(data))
        
        html.append("</body>")
        html.append("</html>")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        
        logger.info(f"Resultados exportados a HTML: {output_path}")
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """Aplanar diccionario anidado"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ResultExporter._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convertir listas a string
                items.append((new_key, ", ".join(str(item) for item in v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def _dict_to_markdown(d: Dict[str, Any], level: int = 2) -> List[str]:
        """Convertir diccionario a Markdown"""
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{'#' * level} {key}")
                lines.append("")
                lines.extend(ResultExporter._dict_to_markdown(value, level + 1))
            elif isinstance(value, list):
                lines.append(f"### {key}")
                lines.append("")
                for item in value:
                    if isinstance(item, dict):
                        lines.extend(ResultExporter._dict_to_markdown(item, level + 1))
                    else:
                        lines.append(f"- {item}")
                lines.append("")
            else:
                lines.append(f"**{key}**: {value}")
                lines.append("")
        return lines
    
    @staticmethod
    def _dict_to_html(d: Dict[str, Any]) -> List[str]:
        """Convertir diccionario a HTML"""
        html = []
        for key, value in d.items():
            if isinstance(value, dict):
                html.append(f"<h2>{key}</h2>")
                html.append("<div class='metadata'>")
                html.extend(ResultExporter._dict_to_html(value))
                html.append("</div>")
            elif isinstance(value, list):
                html.append(f"<h3>{key}</h3>")
                if value and isinstance(value[0], dict):
                    # Tabla para listas de diccionarios
                    html.append("<table>")
                    html.append("<thead><tr>")
                    for k in value[0].keys():
                        html.append(f"<th>{k}</th>")
                    html.append("</tr></thead>")
                    html.append("<tbody>")
                    for item in value:
                        html.append("<tr>")
                        for k in value[0].keys():
                            html.append(f"<td>{item.get(k, '')}</td>")
                        html.append("</tr>")
                    html.append("</tbody></table>")
                else:
                    html.append("<ul>")
                    for item in value:
                        html.append(f"<li>{item}</li>")
                    html.append("</ul>")
            else:
                html.append(f"<p><strong>{key}</strong>: {value}</p>")
        return html
    
    @staticmethod
    def export_multiple_formats(
        data: Any,
        base_path: str,
        formats: List[str] = ["json", "csv", "markdown", "html"]
    ):
        """
        Exportar en múltiples formatos
        
        Args:
            data: Datos a exportar
            base_path: Ruta base (sin extensión)
            formats: Lista de formatos a exportar
        """
        path = Path(base_path)
        base_name = path.stem
        output_dir = path.parent
        
        for fmt in formats:
            output_path = output_dir / f"{base_name}.{fmt}"
            
            try:
                if fmt == "json":
                    if isinstance(data, list):
                        ResultExporter.export_json(data, str(output_path))
                    else:
                        ResultExporter.export_json([data], str(output_path))
                elif fmt == "csv":
                    if isinstance(data, list):
                        ResultExporter.export_csv(data, str(output_path))
                    else:
                        ResultExporter.export_csv([data], str(output_path))
                elif fmt == "markdown":
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            ResultExporter.export_markdown(
                                item,
                                str(output_dir / f"{base_name}_{i}.md")
                            )
                    else:
                        ResultExporter.export_markdown(data, str(output_path))
                elif fmt == "html":
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            ResultExporter.export_html(
                                item,
                                str(output_dir / f"{base_name}_{i}.html")
                            )
                    else:
                        ResultExporter.export_html(data, str(output_path))
            except Exception as e:
                logger.error(f"Error exportando a {fmt}: {e}")
















