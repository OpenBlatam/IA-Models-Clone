"""
Structure Analyzer - Sistema de análisis de estructura de documentos
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentStructure:
    """Estructura de documento"""
    has_title: bool
    has_introduction: bool
    has_conclusion: bool
    sections: List[Dict[str, Any]]
    headers: List[Dict[str, Any]]
    lists: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]


class StructureAnalyzer:
    """Analizador de estructura de documentos"""

    def __init__(self):
        """Inicializar analizador"""
        pass

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar estructura del documento.

        Args:
            content: Contenido

        Returns:
            Análisis de estructura
        """
        structure = DocumentStructure(
            has_title=self._has_title(content),
            has_introduction=self._has_introduction(content),
            has_conclusion=self._has_conclusion(content),
            sections=self._extract_sections(content),
            headers=self._extract_headers(content),
            lists=self._extract_lists(content),
            links=self._extract_links(content),
            images=self._extract_images(content),
            tables=self._extract_tables(content)
        )
        
        # Calcular score de estructura
        structure_score = self._calculate_structure_score(structure)
        
        return {
            "structure": {
                "has_title": structure.has_title,
                "has_introduction": structure.has_introduction,
                "has_conclusion": structure.has_conclusion,
                "section_count": len(structure.sections),
                "header_count": len(structure.headers),
                "list_count": len(structure.lists),
                "link_count": len(structure.links),
                "image_count": len(structure.images),
                "table_count": len(structure.tables)
            },
            "sections": structure.sections,
            "headers": structure.headers,
            "lists": structure.lists,
            "links": structure.links,
            "images": structure.images,
            "tables": structure.tables,
            "structure_score": structure_score,
            "suggestions": self._generate_structure_suggestions(structure)
        }

    def _has_title(self, content: str) -> bool:
        """Verificar si tiene título"""
        lines = content.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Verificar si es un header Markdown
        if first_line.startswith('#'):
            return True
        
        # Verificar si es corto y parece título
        if len(first_line) < 100 and len(first_line) > 5:
            return True
        
        return False

    def _has_introduction(self, content: str) -> bool:
        """Verificar si tiene introducción"""
        content_lower = content.lower()
        intro_keywords = [
            'introducción', 'introduction', 'resumen', 'summary',
            'objetivo', 'objetivos', 'purpose', 'objetivo de',
            'este documento', 'este artículo', 'este texto'
        ]
        
        # Verificar primeros 500 caracteres
        first_part = content_lower[:500]
        return any(keyword in first_part for keyword in intro_keywords)

    def _has_conclusion(self, content: str) -> bool:
        """Verificar si tiene conclusión"""
        content_lower = content.lower()
        concl_keywords = [
            'conclusión', 'conclusion', 'resumen', 'summary',
            'finalmente', 'en resumen', 'en conclusión',
            'para concluir', 'en definitiva', 'en suma'
        ]
        
        # Verificar últimos 500 caracteres
        last_part = content_lower[-500:]
        return any(keyword in last_part for keyword in concl_keywords)

    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extraer secciones"""
        sections = []
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            # Detectar headers (Markdown)
            if line.startswith('#'):
                if current_section:
                    sections.append({
                        "name": current_section,
                        "content_length": len(' '.join(current_content)),
                        "line_count": len(current_content)
                    })
                current_section = line.lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections.append({
                "name": current_section,
                "content_length": len(' '.join(current_content)),
                "line_count": len(current_content)
            })
        
        # Si no hay headers, dividir por párrafos
        if not sections:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            sections = [
                {
                    "name": f"Section {i+1}",
                    "content_length": len(para),
                    "line_count": len(para.split('\n'))
                }
                for i, para in enumerate(paragraphs)
            ]
        
        return sections

    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extraer headers"""
        headers = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                text = line.lstrip('#').strip()
                headers.append({
                    "level": level,
                    "text": text,
                    "line_number": i + 1
                })
        
        return headers

    def _extract_lists(self, content: str) -> List[Dict[str, Any]]:
        """Extraer listas"""
        lists = []
        lines = content.split('\n')
        
        current_list = []
        for i, line in enumerate(lines):
            # Detectar items de lista (Markdown)
            if re.match(r'^[\*\-\+]\s', line) or re.match(r'^\d+\.\s', line):
                current_list.append({
                    "text": line.strip(),
                    "line_number": i + 1
                })
            else:
                if current_list:
                    lists.append({
                        "items": current_list,
                        "item_count": len(current_list)
                    })
                    current_list = []
        
        if current_list:
            lists.append({
                "items": current_list,
                "item_count": len(current_list)
            })
        
        return lists

    def _extract_links(self, content: str) -> List[Dict[str, Any]]:
        """Extraer links"""
        links = []
        
        # Links Markdown [text](url)
        markdown_links = re.finditer(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        for match in markdown_links:
            links.append({
                "text": match.group(1),
                "url": match.group(2),
                "type": "markdown"
            })
        
        # URLs directas
        url_pattern = re.compile(r'https?://[^\s]+')
        for match in url_pattern.finditer(content):
            links.append({
                "text": match.group(),
                "url": match.group(),
                "type": "direct"
            })
        
        return links

    def _extract_images(self, content: str) -> List[Dict[str, Any]]:
        """Extraer imágenes"""
        images = []
        
        # Imágenes Markdown ![alt](url)
        markdown_images = re.finditer(r'!\[([^\]]*)\]\(([^\)]+)\)', content)
        for match in markdown_images:
            images.append({
                "alt": match.group(1),
                "url": match.group(2),
                "type": "markdown"
            })
        
        return images

    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extraer tablas"""
        tables = []
        lines = content.split('\n')
        
        current_table = []
        in_table = False
        
        for i, line in enumerate(lines):
            # Detectar inicio de tabla (Markdown)
            if '|' in line and line.strip().startswith('|'):
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            else:
                if in_table and current_table:
                    tables.append({
                        "rows": current_table,
                        "row_count": len(current_table),
                        "start_line": i - len(current_table) + 1
                    })
                    current_table = []
                    in_table = False
        
        if in_table and current_table:
            tables.append({
                "rows": current_table,
                "row_count": len(current_table),
                "start_line": len(lines) - len(current_table) + 1
            })
        
        return tables

    def _calculate_structure_score(self, structure: DocumentStructure) -> float:
        """Calcular score de estructura"""
        score = 0.0
        
        # Título (20%)
        if structure.has_title:
            score += 0.2
        
        # Introducción (20%)
        if structure.has_introduction:
            score += 0.2
        
        # Conclusión (20%)
        if structure.has_conclusion:
            score += 0.2
        
        # Secciones (20%)
        if structure.sections:
            section_score = min(0.2, len(structure.sections) * 0.05)
            score += section_score
        
        # Headers (10%)
        if structure.headers:
            header_score = min(0.1, len(structure.headers) * 0.02)
            score += header_score
        
        # Listas (5%)
        if structure.lists:
            list_score = min(0.05, len(structure.lists) * 0.01)
            score += list_score
        
        # Links e imágenes (5%)
        if structure.links or structure.images:
            score += 0.05
        
        return min(1.0, score)

    def _generate_structure_suggestions(self, structure: DocumentStructure) -> List[str]:
        """Generar sugerencias de estructura"""
        suggestions = []
        
        if not structure.has_title:
            suggestions.append("Considera agregar un título al documento.")
        
        if not structure.has_introduction:
            suggestions.append("Considera agregar una introducción.")
        
        if not structure.has_conclusion:
            suggestions.append("Considera agregar una conclusión.")
        
        if len(structure.sections) < 2:
            suggestions.append("Considera dividir el contenido en más secciones.")
        
        if not structure.headers:
            suggestions.append("Considera usar headers para organizar el contenido.")
        
        if not structure.lists:
            suggestions.append("Considera usar listas para mejorar la legibilidad.")
        
        return suggestions






