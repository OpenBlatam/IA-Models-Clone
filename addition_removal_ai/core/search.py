"""
Search - Sistema de búsqueda y filtrado avanzado
"""

import logging
import re
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SearchOperator(Enum):
    """Operadores de búsqueda"""
    EQUALS = "equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    BETWEEN = "between"


class SearchEngine:
    """Motor de búsqueda avanzado"""

    def __init__(self):
        """Inicializar el motor de búsqueda"""
        pass

    def search_content(
        self,
        content: str,
        query: str,
        case_sensitive: bool = False,
        whole_words: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Buscar en contenido.

        Args:
            content: Contenido donde buscar
            query: Consulta de búsqueda
            case_sensitive: Si es sensible a mayúsculas
            whole_words: Si busca palabras completas

        Returns:
            Lista de coincidencias
        """
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE
        
        if whole_words:
            pattern = r'\b' + re.escape(query) + r'\b'
        else:
            pattern = re.escape(query)
        
        for match in re.finditer(pattern, content, flags):
            matches.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "line": content[:match.start()].count('\n') + 1,
                "context": self._get_context(content, match.start(), match.end())
            })
        
        return matches

    def _get_context(self, content: str, start: int, end: int, context_size: int = 50) -> str:
        """
        Obtener contexto alrededor de una coincidencia.

        Args:
            content: Contenido
            start: Inicio de la coincidencia
            end: Fin de la coincidencia
            context_size: Tamaño del contexto

        Returns:
            Contexto
        """
        context_start = max(0, start - context_size)
        context_end = min(len(content), end + context_size)
        return content[context_start:context_end]

    def filter_operations(
        self,
        operations: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filtrar operaciones.

        Args:
            operations: Lista de operaciones
            filters: Filtros a aplicar

        Returns:
            Operaciones filtradas
        """
        filtered = operations
        
        for field, condition in filters.items():
            if isinstance(condition, dict):
                operator = condition.get("operator", SearchOperator.EQUALS.value)
                value = condition.get("value")
                
                if operator == SearchOperator.EQUALS.value:
                    filtered = [op for op in filtered if op.get(field) == value]
                elif operator == SearchOperator.CONTAINS.value:
                    filtered = [op for op in filtered if value in str(op.get(field, ""))]
                elif operator == SearchOperator.STARTS_WITH.value:
                    filtered = [op for op in filtered if str(op.get(field, "")).startswith(value)]
                elif operator == SearchOperator.ENDS_WITH.value:
                    filtered = [op for op in filtered if str(op.get(field, "")).endswith(value)]
                elif operator == SearchOperator.REGEX.value:
                    pattern = re.compile(value)
                    filtered = [op for op in filtered if pattern.search(str(op.get(field, "")))]
            else:
                # Búsqueda simple por igualdad
                filtered = [op for op in filtered if op.get(field) == condition]
        
        return filtered

    def search_with_highlight(
        self,
        content: str,
        query: str,
        highlight_tag: str = "mark"
    ) -> Dict[str, Any]:
        """
        Buscar y resaltar coincidencias.

        Args:
            content: Contenido
            query: Consulta
            highlight_tag: Tag HTML para resaltar

        Returns:
            Contenido con resaltado y estadísticas
        """
        matches = self.search_content(content, query)
        
        if not matches:
            return {
                "content": content,
                "highlighted": content,
                "matches": [],
                "count": 0
            }
        
        # Resaltar coincidencias
        highlighted = content
        offset = 0
        
        for match in matches:
            start = match["start"] + offset
            end = match["end"] + offset
            highlighted_text = f"<{highlight_tag}>{match['text']}</{highlight_tag}>"
            highlighted = highlighted[:start] + highlighted_text + highlighted[end:]
            offset += len(highlighted_text) - (end - start)
        
        return {
            "content": content,
            "highlighted": highlighted,
            "matches": matches,
            "count": len(matches)
        }

    def advanced_search(
        self,
        content: str,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Búsqueda avanzada con múltiples criterios.

        Args:
            content: Contenido
            query: Consulta avanzada

        Returns:
            Resultados de búsqueda
        """
        results = {
            "matches": [],
            "statistics": {}
        }
        
        # Búsqueda de texto
        if "text" in query:
            text_matches = self.search_content(
                content,
                query["text"],
                case_sensitive=query.get("case_sensitive", False),
                whole_words=query.get("whole_words", False)
            )
            results["matches"].extend(text_matches)
        
        # Búsqueda por regex
        if "regex" in query:
            pattern = re.compile(query["regex"])
            for match in pattern.finditer(content):
                results["matches"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "line": content[:match.start()].count('\n') + 1
                })
        
        # Estadísticas
        results["statistics"] = {
            "total_matches": len(results["matches"]),
            "unique_matches": len(set(m["text"] for m in results["matches"])),
            "content_length": len(content)
        }
        
        return results






