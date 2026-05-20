"""
Formatters - Soporte para múltiples formatos de contenido
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ContentFormat(Enum):
    """Formatos de contenido soportados"""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    CODE = "code"


class ContentFormatter:
    """Formateador de contenido para diferentes formatos"""

    def __init__(self):
        """Inicializar el formateador"""
        pass

    def detect_format(self, content: str) -> str:
        """
        Detectar el formato del contenido.

        Args:
            content: Contenido a analizar

        Returns:
            Formato detectado
        """
        # Detectar JSON
        try:
            json.loads(content)
            return ContentFormat.JSON.value
        except:
            pass

        # Detectar HTML
        if re.search(r'<[a-z][\s\S]*>', content):
            return ContentFormat.HTML.value

        # Detectar Markdown
        if re.search(r'^#+\s|^\*\s|^-\s|```', content, re.MULTILINE):
            return ContentFormat.MARKDOWN.value

        # Detectar código
        if re.search(r'^\s*(def|class|function|import|const|let|var)\s', content, re.MULTILINE):
            return ContentFormat.CODE.value

        return ContentFormat.PLAIN_TEXT.value

    def add_to_markdown(
        self,
        content: str,
        addition: str,
        position: str = "end"
    ) -> str:
        """Agregar contenido a Markdown preservando estructura"""
        if position == "start":
            return f"{addition}\n\n---\n\n{content}"
        elif position == "end":
            return f"{content}\n\n---\n\n{addition}"
        else:
            return f"{content}\n\n{addition}"

    def add_to_json(
        self,
        content: str,
        addition: Dict[str, Any],
        position: str = "end"
    ) -> str:
        """Agregar contenido a JSON preservando estructura"""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                if position == "end":
                    data.update(addition)
                elif position == "start":
                    new_data = addition.copy()
                    new_data.update(data)
                    data = new_data
            elif isinstance(data, list):
                if position == "end":
                    data.append(addition)
                elif position == "start":
                    data.insert(0, addition)
            
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error al agregar a JSON: {e}")
            return content

    def remove_from_markdown(
        self,
        content: str,
        pattern: str
    ) -> str:
        """Eliminar contenido de Markdown preservando estructura"""
        # Eliminar secciones que coincidan con el patrón
        lines = content.split('\n')
        result_lines = []
        skip_section = False
        
        for line in lines:
            if re.search(pattern, line, re.IGNORECASE):
                skip_section = True
                continue
            if skip_section and (line.startswith('#') or line.strip() == ''):
                skip_section = False
            if not skip_section:
                result_lines.append(line)
        
        return '\n'.join(result_lines)

    def remove_from_json(
        self,
        content: str,
        keys: List[str]
    ) -> str:
        """Eliminar claves de JSON"""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                for key in keys:
                    data.pop(key, None)
            elif isinstance(data, list):
                data = [item for item in data if not any(k in str(item) for k in keys)]
            
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error al eliminar de JSON: {e}")
            return content

    def format_content(
        self,
        content: str,
        target_format: str
    ) -> str:
        """Formatear contenido a un formato específico"""
        current_format = self.detect_format(content)
        
        if current_format == target_format:
            return content
        
        # Conversiones básicas
        if target_format == ContentFormat.PLAIN_TEXT.value:
            # Remover markup
            if current_format == ContentFormat.MARKDOWN.value:
                content = re.sub(r'#+\s', '', content)
                content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
                content = re.sub(r'\*([^*]+)\*', r'\1', content)
            elif current_format == ContentFormat.HTML.value:
                content = re.sub(r'<[^>]+>', '', content)
        
        return content






