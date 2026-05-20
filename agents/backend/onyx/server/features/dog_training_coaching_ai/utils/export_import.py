"""
Export/Import Utilities
=======================
Utilidades para exportar e importar datos.
"""

import json
import csv
import io
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path

from .logger import get_logger
from .file_helpers import ensure_directory, write_json_file, read_json_file

logger = get_logger(__name__)


class DataExporter:
    """Exportador de datos."""
    
    def __init__(self, output_dir: str = "exports"):
        """
        Inicializar exportador.
        
        Args:
            output_dir: Directorio de salida
        """
        self.output_dir = Path(output_dir)
        ensure_directory(str(self.output_dir))
    
    def export_to_json(
        self,
        data: Any,
        filename: Optional[str] = None,
        pretty: bool = True
    ) -> str:
        """
        Exportar a JSON.
        
        Args:
            data: Datos a exportar
            filename: Nombre del archivo
            pretty: Formato bonito
            
        Returns:
            Ruta del archivo exportado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        if pretty:
            write_json_file(str(filepath), data)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f)
        
        logger.info(f"Exported to JSON: {filepath}")
        return str(filepath)
    
    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filename: Optional[str] = None,
        headers: Optional[List[str]] = None
    ) -> str:
        """
        Exportar a CSV.
        
        Args:
            data: Datos a exportar
            filename: Nombre del archivo
            headers: Headers personalizados
            
        Returns:
            Ruta del archivo exportado
        """
        if not data:
            raise ValueError("No data to export")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Obtener headers
        if headers is None:
            headers = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Exported to CSV: {filepath}")
        return str(filepath)
    
    def export_to_text(
        self,
        data: Any,
        filename: Optional[str] = None,
        format_func: Optional[Callable] = None
    ) -> str:
        """
        Exportar a texto.
        
        Args:
            data: Datos a exportar
            filename: Nombre del archivo
            format_func: Función de formateo
            
        Returns:
            Ruta del archivo exportado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        
        if format_func:
            content = format_func(data)
        else:
            content = str(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Exported to text: {filepath}")
        return str(filepath)


class DataImporter:
    """Importador de datos."""
    
    def __init__(self):
        """Inicializar importador."""
        pass
    
    def import_from_json(self, filepath: str) -> Any:
        """
        Importar desde JSON.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Datos importados
        """
        data = read_json_file(filepath)
        logger.info(f"Imported from JSON: {filepath}")
        return data
    
    def import_from_csv(
        self,
        filepath: str,
        encoding: str = 'utf-8'
    ) -> List[Dict[str, Any]]:
        """
        Importar desde CSV.
        
        Args:
            filepath: Ruta del archivo
            encoding: Codificación
            
        Returns:
            Datos importados
        """
        data = []
        
        with open(filepath, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        logger.info(f"Imported from CSV: {filepath}")
        return data
    
    def import_from_text(
        self,
        filepath: str,
        parser: Optional[Callable] = None
    ) -> Any:
        """
        Importar desde texto.
        
        Args:
            filepath: Ruta del archivo
            parser: Función de parsing
            
        Returns:
            Datos importados
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if parser:
            data = parser(content)
        else:
            data = content
        
        logger.info(f"Imported from text: {filepath}")
        return data


def export_data(
    data: Any,
    format: str = "json",
    filename: Optional[str] = None,
    **kwargs
) -> str:
    """
    Exportar datos.
    
    Args:
        data: Datos a exportar
        format: Formato (json, csv, text)
        filename: Nombre del archivo
        **kwargs: Argumentos adicionales
        
    Returns:
        Ruta del archivo exportado
    """
    exporter = DataExporter()
    
    if format == "json":
        return exporter.export_to_json(data, filename, kwargs.get("pretty", True))
    elif format == "csv":
        if not isinstance(data, list):
            raise ValueError("CSV export requires list of dictionaries")
        return exporter.export_to_csv(data, filename, kwargs.get("headers"))
    elif format == "text":
        return exporter.export_to_text(data, filename, kwargs.get("format_func"))
    else:
        raise ValueError(f"Unsupported format: {format}")


def import_data(
    filepath: str,
    format: Optional[str] = None
) -> Any:
    """
    Importar datos.
    
    Args:
        filepath: Ruta del archivo
        format: Formato (auto-detect si None)
        
    Returns:
        Datos importados
    """
    importer = DataImporter()
    
    if format is None:
        # Auto-detectar formato
        path = Path(filepath)
        format = path.suffix[1:].lower()  # Remover el punto
    
    if format == "json":
        return importer.import_from_json(filepath)
    elif format == "csv":
        return importer.import_from_csv(filepath)
    elif format == "text" or format == "txt":
        return importer.import_from_text(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

