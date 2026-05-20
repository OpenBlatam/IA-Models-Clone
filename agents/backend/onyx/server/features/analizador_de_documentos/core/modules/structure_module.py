"""
Structure Module - Módulo de Estructura
========================================

Módulo especializado para análisis de estructura.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


@dataclass
class StructureInfo:
    """Información de estructura."""
    sections: List[Dict[str, Any]]
    hierarchy_depth: int
    total_sections: int
    has_table_of_contents: bool
    has_index: bool
    has_bibliography: bool
    structure_score: float


class StructureModule:
    """Módulo de estructura."""
    
    def __init__(self):
        """Inicializar módulo."""
        self.module_id = "structure"
        self.name = "Structure Module"
        self.version = "1.0.0"
        logger.info(f"{self.name} inicializado")
    
    async def analyze_structure(self, content: str) -> StructureInfo:
        """Analizar estructura del documento."""
        sections = self._extract_sections(content)
        hierarchy_depth = self._calculate_hierarchy_depth(sections)
        has_toc = self._detect_table_of_contents(content)
        has_index = self._detect_index(content)
        has_bibliography = self._detect_bibliography(content)
        structure_score = self._calculate_structure_score(
            sections, hierarchy_depth, has_toc, has_index, has_bibliography
        )
        
        return StructureInfo(
            sections=sections,
            hierarchy_depth=hierarchy_depth,
            total_sections=len(sections),
            has_table_of_contents=has_toc,
            has_index=has_index,
            has_bibliography=has_bibliography,
            structure_score=structure_score
        )
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extraer secciones."""
        sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            header_match = self._detect_header(line)
            if header_match:
                level, title = header_match
                sections.append({
                    "level": level,
                    "title": title,
                    "position": i
                })
        
        return sections
    
    def _detect_header(self, line: str) -> Optional[tuple]:
        """Detectar encabezado."""
        line = line.strip()
        
        # Markdown
        md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if md_match:
            return (len(md_match.group(1)), md_match.group(2))
        
        # Numérico
        num_match = re.match(r'^(\d+(?:\.\d+)*)\.\s+(.+)$', line)
        if num_match:
            return (len(num_match.group(1).split('.')), num_match.group(2))
        
        return None
    
    def _calculate_hierarchy_depth(self, sections: List[Dict[str, Any]]) -> int:
        """Calcular profundidad de jerarquía."""
        if not sections:
            return 0
        return max(s["level"] for s in sections)
    
    def _detect_table_of_contents(self, content: str) -> bool:
        """Detectar tabla de contenidos."""
        patterns = [r'table\s+of\s+contents', r'contents', r'índice']
        first_500 = content[:500].lower()
        return any(re.search(p, first_500) for p in patterns)
    
    def _detect_index(self, content: str) -> bool:
        """Detectar índice."""
        patterns = [r'\bindex\b', r'índice\s+alfabético']
        last_1000 = content[-1000:].lower()
        return any(re.search(p, last_1000) for p in patterns)
    
    def _detect_bibliography(self, content: str) -> bool:
        """Detectar bibliografía."""
        patterns = [r'\bbibliography\b', r'\breferences\b', r'\breferencias\b']
        last_1000 = content[-1000:].lower()
        return any(re.search(p, last_1000) for p in patterns)
    
    def _calculate_structure_score(
        self,
        sections: List[Dict[str, Any]],
        hierarchy_depth: int,
        has_toc: bool,
        has_index: bool,
        has_bibliography: bool
    ) -> float:
        """Calcular score de estructura."""
        score = 0.0
        
        if sections:
            score += 0.3
            score += min(len(sections) / 10, 0.2)
        
        if hierarchy_depth >= 2:
            score += 0.2
        if hierarchy_depth >= 3:
            score += 0.1
        
        if has_toc:
            score += 0.1
        if has_index:
            score += 0.1
        if has_bibliography:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información del módulo."""
        return {
            "module_id": self.module_id,
            "name": self.name,
            "version": self.version
        }


__all__ = [
    "StructureModule",
    "StructureInfo"
]


