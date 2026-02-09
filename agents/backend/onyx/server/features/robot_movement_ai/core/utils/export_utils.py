"""
Export Utilities
================

Utilidades para exportar datos y configuraciones.
"""

import json
import yaml
import csv
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ExportManager:
    """
    Gestor de exportaciones.
    
    Gestiona exportaciones de datos en múltiples formatos.
    """
    
    def __init__(self):
        """Inicializar gestor de exportaciones."""
        self.export_history: List[Dict[str, Any]] = []
    
    def export_json(
        self,
        data: Any,
        filepath: str,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> str:
        """
        Exportar datos a JSON.
        
        Args:
            data: Datos a exportar
            filepath: Ruta del archivo
            indent: Indentación
            ensure_ascii: Si asegurar ASCII
            
        Returns:
            Ruta del archivo exportado
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        self._record_export("json", filepath, data)
        logger.info(f"Data exported to JSON: {filepath}")
        
        return str(path)
    
    def export_yaml(
        self,
        data: Any,
        filepath: str,
        default_flow_style: bool = False
    ) -> str:
        """
        Exportar datos a YAML.
        
        Args:
            data: Datos a exportar
            filepath: Ruta del archivo
            default_flow_style: Estilo de flujo
            
        Returns:
            Ruta del archivo exportado
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=default_flow_style, allow_unicode=True)
        
        self._record_export("yaml", filepath, data)
        logger.info(f"Data exported to YAML: {filepath}")
        
        return str(path)
    
    def export_csv(
        self,
        data: List[Dict[str, Any]],
        filepath: str,
        fieldnames: Optional[List[str]] = None
    ) -> str:
        """
        Exportar datos a CSV.
        
        Args:
            data: Lista de diccionarios
            filepath: Ruta del archivo
            fieldnames: Nombres de campos (None = auto-detect)
            
        Returns:
            Ruta del archivo exportado
        """
        if not data:
            raise ValueError("Data cannot be empty for CSV export")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        self._record_export("csv", filepath, data)
        logger.info(f"Data exported to CSV: {filepath}")
        
        return str(path)
    
    def export_markdown(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        filepath: str,
        title: str = "Data Export"
    ) -> str:
        """
        Exportar datos a Markdown.
        
        Args:
            data: Datos a exportar
            filepath: Ruta del archivo
            title: Título del documento
            
        Returns:
            Ruta del archivo exportado
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [f"# {title}\n", f"*Generated: {datetime.now().isoformat()}*\n"]
        
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                # Tabla
                lines.append("\n| " + " | ".join(data[0].keys()) + " |")
                lines.append("| " + " | ".join(["---"] * len(data[0].keys())) + " |")
                for item in data:
                    lines.append("| " + " | ".join(str(v) for v in item.values()) + " |")
            else:
                # Lista
                for item in data:
                    lines.append(f"- {item}")
        elif isinstance(data, dict):
            # Diccionario
            for key, value in data.items():
                lines.append(f"\n## {key}\n")
                if isinstance(value, (dict, list)):
                    lines.append(f"```json\n{json.dumps(value, indent=2)}\n```")
                else:
                    lines.append(f"{value}")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        self._record_export("markdown", filepath, data)
        logger.info(f"Data exported to Markdown: {filepath}")
        
        return str(path)
    
    def _record_export(self, format: str, filepath: str, data: Any) -> None:
        """Registrar exportación."""
        self.export_history.append({
            "format": format,
            "filepath": filepath,
            "timestamp": datetime.now().isoformat(),
            "data_size": len(str(data))
        })
    
    def get_export_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de exportaciones."""
        return self.export_history[-limit:]


# Instancia global
_export_manager: Optional[ExportManager] = None


def get_export_manager() -> ExportManager:
    """Obtener instancia global del gestor de exportaciones."""
    global _export_manager
    if _export_manager is None:
        _export_manager = ExportManager()
    return _export_manager






