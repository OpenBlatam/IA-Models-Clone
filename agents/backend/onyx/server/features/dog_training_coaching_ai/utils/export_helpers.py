"""
Export Helpers
==============
Utilidades para exportar datos.
"""

import csv
import json
from typing import List, Dict, Any, Optional
from io import StringIO
from datetime import datetime


def export_to_json(data: List[Dict[str, Any]], pretty: bool = True) -> str:
    """
    Exportar datos a JSON.
    
    Args:
        data: Lista de diccionarios
        pretty: Formatear de forma legible
        
    Returns:
        String JSON
    """
    indent = 2 if pretty else None
    return json.dumps(data, indent=indent, ensure_ascii=False, default=str)


def export_to_csv(
    data: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None
) -> str:
    """
    Exportar datos a CSV.
    
    Args:
        data: Lista de diccionarios
        fieldnames: Nombres de campos (opcional)
        
    Returns:
        String CSV
    """
    if not data:
        return ""
    
    if not fieldnames:
        fieldnames = list(data[0].keys())
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    
    return output.getvalue()


def export_to_text(
    data: List[Dict[str, Any]],
    separator: str = " | ",
    include_headers: bool = True
) -> str:
    """
    Exportar datos a texto plano.
    
    Args:
        data: Lista de diccionarios
        separator: Separador entre campos
        include_headers: Incluir encabezados
        
    Returns:
        String de texto
    """
    if not data:
        return ""
    
    lines = []
    fieldnames = list(data[0].keys())
    
    if include_headers:
        lines.append(separator.join(fieldnames))
        lines.append("-" * len(separator.join(fieldnames)))
    
    for item in data:
        values = [str(item.get(field, "")) for field in fieldnames]
        lines.append(separator.join(values))
    
    return "\n".join(lines)


def create_export_filename(prefix: str, extension: str) -> str:
    """
    Crear nombre de archivo para exportación.
    
    Args:
        prefix: Prefijo del archivo
        extension: Extensión (json, csv, txt)
        
    Returns:
        Nombre de archivo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

