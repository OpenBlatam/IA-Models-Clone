"""
Diff - Sistema de comparación y diferencias
"""

import logging
import difflib
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DiffType(Enum):
    """Tipos de diferencias"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class ContentDiff:
    """Sistema de comparación de contenido"""

    def __init__(self):
        """Inicializar el comparador"""
        pass

    def compute_diff(
        self,
        original: str,
        modified: str,
        context_lines: int = 3
    ) -> Dict[str, Any]:
        """
        Calcular diferencias entre dos versiones.

        Args:
            original: Contenido original
            modified: Contenido modificado
            context_lines: Líneas de contexto a mostrar

        Returns:
            Diccionario con las diferencias
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            original_lines,
            modified_lines,
            lineterm='',
            n=context_lines
        ))
        
        # Analizar cambios
        changes = self._analyze_changes(original, modified)
        
        return {
            "unified_diff": diff,
            "changes": changes,
            "summary": {
                "original_length": len(original),
                "modified_length": len(modified),
                "original_lines": len(original_lines),
                "modified_lines": len(modified_lines),
                "diff_lines": len(diff),
                "change_count": len(changes)
            }
        }

    def _analyze_changes(self, original: str, modified: str) -> List[Dict[str, Any]]:
        """
        Analizar cambios en detalle.

        Args:
            original: Contenido original
            modified: Contenido modificado

        Returns:
            Lista de cambios detectados
        """
        changes = []
        
        # Usar SequenceMatcher para encontrar bloques similares
        matcher = difflib.SequenceMatcher(None, original, modified)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            
            change = {
                "type": tag,
                "original_text": original[i1:i2],
                "modified_text": modified[j1:j2],
                "original_start": i1,
                "original_end": i2,
                "modified_start": j1,
                "modified_end": j2
            }
            
            # Clasificar el tipo de cambio
            if tag == 'delete':
                change["diff_type"] = DiffType.REMOVED.value
            elif tag == 'insert':
                change["diff_type"] = DiffType.ADDED.value
            elif tag == 'replace':
                change["diff_type"] = DiffType.MODIFIED.value
            
            changes.append(change)
        
        return changes

    def get_additions(self, diff_result: Dict[str, Any]) -> List[str]:
        """
        Extraer solo las adiciones del diff.

        Args:
            diff_result: Resultado de compute_diff

        Returns:
            Lista de textos agregados
        """
        additions = []
        for change in diff_result.get("changes", []):
            if change.get("diff_type") == DiffType.ADDED.value:
                additions.append(change.get("modified_text", ""))
        return additions

    def get_removals(self, diff_result: Dict[str, Any]) -> List[str]:
        """
        Extraer solo las eliminaciones del diff.

        Args:
            diff_result: Resultado de compute_diff

        Returns:
            Lista de textos eliminados
        """
        removals = []
        for change in diff_result.get("changes", []):
            if change.get("diff_type") == DiffType.REMOVED.value:
                removals.append(change.get("original_text", ""))
        return removals

    def compute_similarity(self, original: str, modified: str) -> float:
        """
        Calcular similitud entre dos textos (0-1).

        Args:
            original: Contenido original
            modified: Contenido modificado

        Returns:
            Ratio de similitud (0-1)
        """
        return difflib.SequenceMatcher(None, original, modified).ratio()

    def highlight_changes(
        self,
        original: str,
        modified: str,
        format: str = "html"
    ) -> str:
        """
        Generar versión con cambios resaltados.

        Args:
            original: Contenido original
            modified: Contenido modificado
            format: Formato de salida (html, markdown, plain)

        Returns:
            Texto con cambios resaltados
        """
        if format == "html":
            return self._highlight_html(original, modified)
        elif format == "markdown":
            return self._highlight_markdown(original, modified)
        else:
            return self._highlight_plain(original, modified)

    def _highlight_html(self, original: str, modified: str) -> str:
        """Generar HTML con cambios resaltados"""
        diff_result = self.compute_diff(original, modified)
        html_parts = []
        
        for change in diff_result.get("changes", []):
            if change.get("diff_type") == DiffType.ADDED.value:
                html_parts.append(f'<ins style="background-color: #d4edda;">{change["modified_text"]}</ins>')
            elif change.get("diff_type") == DiffType.REMOVED.value:
                html_parts.append(f'<del style="background-color: #f8d7da;">{change["original_text"]}</del>')
            elif change.get("diff_type") == DiffType.MODIFIED.value:
                html_parts.append(f'<del style="background-color: #f8d7da;">{change["original_text"]}</del>')
                html_parts.append(f'<ins style="background-color: #d4edda;">{change["modified_text"]}</ins>')
        
        return "".join(html_parts)

    def _highlight_markdown(self, original: str, modified: str) -> str:
        """Generar Markdown con cambios resaltados"""
        diff_result = self.compute_diff(original, modified)
        md_parts = []
        
        for change in diff_result.get("changes", []):
            if change.get("diff_type") == DiffType.ADDED.value:
                md_parts.append(f"+++{change['modified_text']}+++")
            elif change.get("diff_type") == DiffType.REMOVED.value:
                md_parts.append(f"---{change['original_text']}---")
            elif change.get("diff_type") == DiffType.MODIFIED.value:
                md_parts.append(f"---{change['original_text']}---")
                md_parts.append(f"+++{change['modified_text']}+++")
        
        return "".join(md_parts)

    def _highlight_plain(self, original: str, modified: str) -> str:
        """Generar texto plano con marcadores"""
        diff_result = self.compute_diff(original, modified)
        parts = []
        
        for change in diff_result.get("changes", []):
            if change.get("diff_type") == DiffType.ADDED.value:
                parts.append(f"[+ {change['modified_text']}]")
            elif change.get("diff_type") == DiffType.REMOVED.value:
                parts.append(f"[- {change['original_text']}]")
            elif change.get("diff_type") == DiffType.MODIFIED.value:
                parts.append(f"[- {change['original_text']}]")
                parts.append(f"[+ {change['modified_text']}]")
        
        return "".join(parts)






