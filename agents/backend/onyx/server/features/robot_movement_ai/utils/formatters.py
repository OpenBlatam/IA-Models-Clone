"""Formatters - Formateadores de datos mejorados"""

from typing import Any, Dict, Optional
from datetime import datetime, date
from decimal import Decimal
import json


class Formatter:
    """Formateador de datos mejorado"""
    
    @staticmethod
    def format(data: Any, format_type: str, **kwargs) -> str:
        """
        Formatea datos según el tipo.
        
        Args:
            data: Datos a formatear
            format_type: Tipo de formato (json, yaml, xml, csv, table)
            **kwargs: Opciones adicionales de formato
        
        Returns:
            String formateado
        """
        if format_type == "json":
            return Formatter.to_json(data, **kwargs)
        elif format_type == "yaml":
            return Formatter.to_yaml(data, **kwargs)
        elif format_type == "xml":
            return Formatter.to_xml(data, **kwargs)
        elif format_type == "csv":
            return Formatter.to_csv(data, **kwargs)
        elif format_type == "table":
            return Formatter.to_table(data, **kwargs)
        else:
            return str(data)
    
    @staticmethod
    def to_json(
        data: Any,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False,
        sort_keys: bool = False
    ) -> str:
        """
        Convertir a JSON.
        
        Args:
            data: Datos a convertir
            indent: Indentación
            ensure_ascii: Escapar caracteres no ASCII
            sort_keys: Ordenar keys
        
        Returns:
            JSON string
        """
        return json.dumps(
            data,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            default=Formatter._json_default
        )
    
    @staticmethod
    def to_yaml(data: Any, **kwargs) -> str:
        """
        Convertir a YAML.
        
        Args:
            data: Datos a convertir
            **kwargs: Opciones de YAML
        
        Returns:
            YAML string
        """
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False, **kwargs)
        except ImportError:
            return json.dumps(data, indent=2)
    
    @staticmethod
    def to_xml(data: Any, root_name: str = "root", **kwargs) -> str:
        """
        Convertir a XML.
        
        Args:
            data: Datos a convertir
            root_name: Nombre del elemento raíz
            **kwargs: Opciones adicionales
        
        Returns:
            XML string
        """
        def dict_to_xml(d: Dict[str, Any], root: str = "item") -> str:
            xml_parts = [f"<{root}>"]
            for key, value in d.items():
                if isinstance(value, dict):
                    xml_parts.append(dict_to_xml(value, key))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            xml_parts.append(dict_to_xml(item, key))
                        else:
                            xml_parts.append(f"<{key}>{item}</{key}>")
                else:
                    xml_parts.append(f"<{key}>{value}</{key}>")
            xml_parts.append(f"</{root}>")
            return "".join(xml_parts)
        
        if isinstance(data, dict):
            return f'<?xml version="1.0" encoding="UTF-8"?>\n{dict_to_xml(data, root_name)}'
        else:
            return f'<?xml version="1.0" encoding="UTF-8"?>\n<{root_name}>{data}</{root_name}>'
    
    @staticmethod
    def to_csv(data: Any, delimiter: str = ",", headers: Optional[list] = None) -> str:
        """
        Convertir a CSV.
        
        Args:
            data: Datos a convertir (lista de dicts o lista de listas)
            delimiter: Delimitador
            headers: Headers opcionales
        
        Returns:
            CSV string
        """
        if not data:
            return ""
        
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                if headers is None:
                    headers = list(data[0].keys())
                rows = [delimiter.join(str(item.get(h, "")) for h in headers) for item in data]
                return delimiter.join(headers) + "\n" + "\n".join(rows)
            elif isinstance(data[0], list):
                rows = [delimiter.join(str(cell) for cell in row) for row in data]
                if headers:
                    return delimiter.join(headers) + "\n" + "\n".join(rows)
                return "\n".join(rows)
        
        return str(data)
    
    @staticmethod
    def to_table(
        data: Any,
        headers: Optional[list] = None,
        max_width: int = 80
    ) -> str:
        """
        Convertir a tabla formateada.
        
        Args:
            data: Datos a convertir
            headers: Headers opcionales
            max_width: Ancho máximo de columna
        
        Returns:
            Tabla formateada
        """
        if not data:
            return ""
        
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                if headers is None:
                    headers = list(data[0].keys())
                
                col_widths = {h: len(str(h)) for h in headers}
                for row in data:
                    for h in headers:
                        val_len = len(str(row.get(h, "")))
                        col_widths[h] = max(col_widths[h], min(val_len, max_width))
                
                separator = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"
                header_row = "|" + "|".join(f" {h:<{col_widths[h]}} " for h in headers) + "|"
                
                rows = [separator, header_row, separator]
                for row in data:
                    row_str = "|" + "|".join(
                        f" {str(row.get(h, ''))[:max_width]:<{col_widths[h]}} "
                        for h in headers
                    ) + "|"
                    rows.append(row_str)
                rows.append(separator)
                
                return "\n".join(rows)
        
        return str(data)
    
    @staticmethod
    def format_number(value: Any, decimals: int = 2, thousands_sep: str = ",") -> str:
        """
        Formatear número.
        
        Args:
            value: Número a formatear
            decimals: Decimales
            thousands_sep: Separador de miles
        
        Returns:
            Número formateado
        """
        try:
            num = float(value)
            formatted = f"{num:,.{decimals}f}"
            return formatted.replace(",", thousands_sep) if thousands_sep != "," else formatted
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def format_bytes(bytes_value: int, binary: bool = False) -> str:
        """
        Formatear bytes a formato legible.
        
        Args:
            bytes_value: Valor en bytes
            binary: Si True, usa unidades binarias (KiB, MiB, etc.)
        
        Returns:
            String formateado (ej: "1.5 MB")
        """
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        if binary:
            units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
            base = 1024
        else:
            base = 1000
        
        if bytes_value == 0:
            return "0 B"
        
        size = float(bytes_value)
        unit_index = 0
        
        while size >= base and unit_index < len(units) - 1:
            size /= base
            unit_index += 1
        
        return f"{size:.2f} {units[unit_index]}"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Formatear duración en segundos a formato legible.
        
        Args:
            seconds: Segundos
        
        Returns:
            String formateado (ej: "1h 23m 45s")
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"
    
    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Default handler para JSON serialization."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

