"""
Document Structure Advanced - Análisis de Estructura Avanzado
=============================================================

Sistema avanzado de análisis de estructura de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Sección de documento."""
    section_id: str
    title: str
    level: int
    content: str
    start_position: int
    end_position: int
    subsections: List['DocumentSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureAnalysis:
    """Análisis de estructura."""
    document_id: str
    sections: List[DocumentSection]
    hierarchy_depth: int
    total_sections: int
    structure_score: float
    has_table_of_contents: bool = False
    has_index: bool = False
    has_bibliography: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedStructureAnalyzer:
    """Analizador de estructura avanzado."""
    
    def __init__(self, analyzer):
        """Inicializar analizador."""
        self.analyzer = analyzer
    
    async def analyze_structure(
        self,
        document_id: str,
        content: str
    ) -> StructureAnalysis:
        """
        Analizar estructura del documento.
        
        Args:
            document_id: ID del documento
            content: Contenido del documento
        
        Returns:
            StructureAnalysis con resultados
        """
        # Extraer secciones
        sections = self._extract_sections(content)
        
        # Analizar jerarquía
        hierarchy_depth = self._calculate_hierarchy_depth(sections)
        
        # Detectar elementos especiales
        has_toc = self._detect_table_of_contents(content)
        has_index = self._detect_index(content)
        has_bibliography = self._detect_bibliography(content)
        
        # Calcular score de estructura
        structure_score = self._calculate_structure_score(
            sections, hierarchy_depth, has_toc, has_index, has_bibliography
        )
        
        return StructureAnalysis(
            document_id=document_id,
            sections=sections,
            hierarchy_depth=hierarchy_depth,
            total_sections=len(sections),
            structure_score=structure_score,
            has_table_of_contents=has_toc,
            has_index=has_index,
            has_bibliography=has_bibliography
        )
    
    def _extract_sections(self, content: str) -> List[DocumentSection]:
        """Extraer secciones del documento."""
        sections = []
        lines = content.split('\n')
        current_section = None
        section_stack = []
        
        for i, line in enumerate(lines):
            # Detectar encabezados (múltiples formatos)
            header_match = self._detect_header(line)
            
            if header_match:
                level, title = header_match
                
                # Crear nueva sección
                section = DocumentSection(
                    section_id=f"section_{len(sections)}",
                    title=title,
                    level=level,
                    content="",
                    start_position=i,
                    end_position=i
                )
                
                # Gestionar jerarquía
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                
                if section_stack:
                    section_stack[-1].subsections.append(section)
                else:
                    sections.append(section)
                
                section_stack.append(section)
                current_section = section
            elif current_section:
                # Agregar contenido a la sección actual
                current_section.content += line + "\n"
        
        # Actualizar posiciones finales
        for section in sections:
            self._update_section_positions(section)
        
        return sections
    
    def _detect_header(self, line: str) -> Optional[Tuple[int, str]]:
        """Detectar encabezado en una línea."""
        line = line.strip()
        
        # Markdown headers (# ## ###)
        md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if md_match:
            level = len(md_match.group(1))
            title = md_match.group(2).strip()
            return (level, title)
        
        # Numeric headers (1. 1.1 1.1.1)
        num_match = re.match(r'^(\d+(?:\.\d+)*)\.\s+(.+)$', line)
        if num_match:
            level = len(num_match.group(1).split('.'))
            title = num_match.group(2).strip()
            return (level, title)
        
        # Uppercase headers (asumiendo que son títulos)
        if line.isupper() and len(line) > 3 and len(line) < 100:
            return (1, line.title())
        
        return None
    
    def _calculate_hierarchy_depth(self, sections: List[DocumentSection]) -> int:
        """Calcular profundidad de jerarquía."""
        def get_max_depth(section: DocumentSection) -> int:
            if not section.subsections:
                return section.level
            return max(get_max_depth(sub) for sub in section.subsections)
        
        if not sections:
            return 0
        
        return max(get_max_depth(s) for s in sections)
    
    def _detect_table_of_contents(self, content: str) -> bool:
        """Detectar tabla de contenidos."""
        toc_patterns = [
            r'table\s+of\s+contents',
            r'contents',
            r'índice',
            r'contenido'
        ]
        
        first_500 = content[:500].lower()
        return any(re.search(pattern, first_500) for pattern in toc_patterns)
    
    def _detect_index(self, content: str) -> bool:
        """Detectar índice."""
        index_patterns = [
            r'\bindex\b',
            r'índice\s+alfabético',
            r'índice\s+temático'
        ]
        
        last_1000 = content[-1000:].lower()
        return any(re.search(pattern, last_1000) for pattern in index_patterns)
    
    def _detect_bibliography(self, content: str) -> bool:
        """Detectar bibliografía."""
        bib_patterns = [
            r'\bbibliography\b',
            r'\breferences\b',
            r'\breferencias\b',
            r'\bbibliograf[aí]a\b'
        ]
        
        last_1000 = content[-1000:].lower()
        return any(re.search(pattern, last_1000) for pattern in bib_patterns)
    
    def _calculate_structure_score(
        self,
        sections: List[DocumentSection],
        hierarchy_depth: int,
        has_toc: bool,
        has_index: bool,
        has_bibliography: bool
    ) -> float:
        """Calcular score de estructura."""
        score = 0.0
        
        # Puntos por tener secciones
        if sections:
            score += 0.3
            score += min(len(sections) / 10, 0.2)  # Hasta 0.2 por cantidad
        
        # Puntos por jerarquía
        if hierarchy_depth >= 2:
            score += 0.2
        if hierarchy_depth >= 3:
            score += 0.1
        
        # Puntos por elementos especiales
        if has_toc:
            score += 0.1
        if has_index:
            score += 0.1
        if has_bibliography:
            score += 0.1
        
        return min(score, 1.0)
    
    def _update_section_positions(self, section: DocumentSection):
        """Actualizar posiciones de sección."""
        if section.subsections:
            for sub in section.subsections:
                self._update_section_positions(sub)
                section.end_position = max(section.end_position, sub.end_position)


__all__ = [
    "AdvancedStructureAnalyzer",
    "StructureAnalysis",
    "DocumentSection"
]


