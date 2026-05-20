"""
Manual Parser
=============

Parser para extraer información estructurada de manuales generados.
"""

from typing import Dict, Any
from .extractors import (
    TitleExtractor,
    DifficultyExtractor,
    TimeExtractor,
    ToolsExtractor,
    MaterialsExtractor,
    SafetyExtractor,
    TagsExtractor,
)


class ManualParser:
    """Parser para extraer información de manuales."""
    
    def __init__(self):
        """Inicializar parser."""
        self.title_extractor = TitleExtractor()
        self.difficulty_extractor = DifficultyExtractor()
        self.time_extractor = TimeExtractor()
        self.tools_extractor = ToolsExtractor()
        self.materials_extractor = MaterialsExtractor()
        self.safety_extractor = SafetyExtractor()
        self.tags_extractor = TagsExtractor()
    
    def parse_manual(self, manual_content: str, category: str) -> Dict[str, Any]:
        """
        Parsear manual completo y extraer toda la información.
        
        Args:
            manual_content: Contenido del manual
            category: Categoría del manual
        
        Returns:
            Diccionario con información extraída
        """
        tags = self.tags_extractor.extract(manual_content, category=category)
        
        return {
            "title": self.title_extractor.extract(manual_content),
            "difficulty": self.difficulty_extractor.extract(manual_content),
            "estimated_time": self.time_extractor.extract(manual_content),
            "tools_required": self.tools_extractor.extract(manual_content),
            "materials_required": self.materials_extractor.extract(manual_content),
            "safety_warnings": self.safety_extractor.extract(manual_content),
            "tags": ",".join(tags)
        }

