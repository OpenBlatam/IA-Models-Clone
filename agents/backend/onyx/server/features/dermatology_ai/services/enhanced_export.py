"""
Sistema de exportación mejorado
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import csv
import io
from pathlib import Path


class EnhancedExportManager:
    """Gestor de exportación mejorado"""
    
    def __init__(self):
        """Inicializa el gestor de exportación"""
        pass
    
    def export_to_json(self, data: Any, pretty: bool = True) -> bytes:
        """
        Exporta a JSON
        
        Args:
            data: Datos a exportar
            pretty: Formato bonito
            
        Returns:
            Bytes del JSON
        """
        if pretty:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            json_str = json.dumps(data, ensure_ascii=False)
        
        return json_str.encode('utf-8')
    
    def export_to_csv(self, data: List[Dict], delimiter: str = ",") -> bytes:
        """
        Exporta a CSV
        
        Args:
            data: Lista de diccionarios
            delimiter: Delimitador
            
        Returns:
            Bytes del CSV
        """
        if not data:
            return b""
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys(), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue().encode('utf-8')
    
    def export_to_xml(self, data: Dict, root_name: str = "data") -> bytes:
        """
        Exporta a XML
        
        Args:
            data: Datos a exportar
            root_name: Nombre del elemento raíz
            
        Returns:
            Bytes del XML
        """
        xml_lines = [f'<?xml version="1.0" encoding="UTF-8"?>', f'<{root_name}>']
        xml_lines.extend(self._dict_to_xml(data, indent=1))
        xml_lines.append(f'</{root_name}>')
        
        return '\n'.join(xml_lines).encode('utf-8')
    
    def _dict_to_xml(self, data: Any, indent: int = 0) -> List[str]:
        """Convierte diccionario a XML"""
        lines = []
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f'{indent_str}<{key}>')
                    lines.extend(self._dict_to_xml(value, indent + 1))
                    lines.append(f'{indent_str}</{key}>')
                else:
                    lines.append(f'{indent_str}<{key}>{value}</{key}>')
        elif isinstance(data, list):
            for item in data:
                lines.append(f'{indent_str}<item>')
                lines.extend(self._dict_to_xml(item, indent + 1))
                lines.append(f'{indent_str}</item>')
        else:
            lines.append(f'{indent_str}{data}')
        
        return lines
    
    def export_to_markdown(self, data: Dict, title: str = "Report") -> bytes:
        """
        Exporta a Markdown
        
        Args:
            data: Datos a exportar
            title: Título del documento
            
        Returns:
            Bytes del Markdown
        """
        lines = [f"# {title}", "", f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*", ""]
        lines.extend(self._dict_to_markdown(data))
        
        return '\n'.join(lines).encode('utf-8')
    
    def _dict_to_markdown(self, data: Any, level: int = 1) -> List[str]:
        """Convierte diccionario a Markdown"""
        lines = []
        prefix = "#" * level
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix} {key}")
                    lines.append("")
                    lines.extend(self._dict_to_markdown(value, level + 1))
                    lines.append("")
                else:
                    lines.append(f"**{key}**: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lines.extend(self._dict_to_markdown(item, level))
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(str(data))
        
        return lines
    
    def export_to_yaml(self, data: Dict) -> bytes:
        """
        Exporta a YAML
        
        Args:
            data: Datos a exportar
            
        Returns:
            Bytes del YAML
        """
        try:
            import yaml
            yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True)
            return yaml_str.encode('utf-8')
        except ImportError:
            # Si no hay PyYAML, retornar JSON
            return self.export_to_json(data)
    
    def get_supported_formats(self) -> List[str]:
        """Obtiene formatos soportados"""
        formats = ["json", "csv", "xml", "markdown"]
        
        try:
            import yaml
            formats.append("yaml")
        except ImportError:
            pass
        
        return formats






