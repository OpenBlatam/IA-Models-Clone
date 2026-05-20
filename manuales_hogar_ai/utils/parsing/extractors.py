"""
Content Extractors
==================

Extractores especializados para diferentes tipos de contenido.
"""

import re
from typing import Optional, List
from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """Clase base para extractores."""
    
    @abstractmethod
    def extract(self, content: str, **kwargs) -> Optional[str]:
        """Extraer información del contenido."""
        pass


class TitleExtractor(BaseExtractor):
    """Extractor de títulos."""
    
    def extract(self, content: str, **kwargs) -> Optional[str]:
        """Extraer título del manual."""
        patterns = [
            r'^1\.\s*TÍTULO[:\s]+(.+?)$',
            r'^#\s*(.+?)$',
            r'^TÍTULO[:\s]+(.+?)$',
            r'^(.+?)\n.*DIAGNÓSTICO',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if 10 < len(title) < 200:
                    return title
        
        first_line = content.split('\n')[0].strip()
        if len(first_line) > 10:
            return first_line[:200]
        
        return None


class DifficultyExtractor(BaseExtractor):
    """Extractor de dificultad."""
    
    def extract(self, content: str, **kwargs) -> Optional[str]:
        """Extraer dificultad del manual."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['dificultad: fácil', 'dificultad fácil', 'fácil']):
            return "Fácil"
        elif any(word in content_lower for word in ['dificultad: difícil', 'dificultad difícil', 'difícil']):
            return "Difícil"
        elif any(word in content_lower for word in ['dificultad: media', 'dificultad media', 'media']):
            return "Media"
        
        return None


class TimeExtractor(BaseExtractor):
    """Extractor de tiempo estimado."""
    
    def extract(self, content: str, **kwargs) -> Optional[str]:
        """Extraer tiempo estimado."""
        patterns = [
            r'⏱️\s*Tiempo[:\s]+(.+?)(?:\n|$)',
            r'Tiempo[:\s]+(.+?)(?:\n|$)',
            r'(\d+\s*(?:minutos?|horas?|días?))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                time_str = match.group(1).strip()
                if len(time_str) < 50:
                    return time_str
        
        return None


class ToolsExtractor(BaseExtractor):
    """Extractor de herramientas."""
    
    def extract(self, content: str, **kwargs) -> Optional[str]:
        """Extraer lista de herramientas."""
        tools_pattern = r'HERRAMIENTAS[:\s]+(.*?)(?=\n\n|\n\d+\.|$)'
        match = re.search(tools_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if match:
            tools_text = match.group(1).strip()
            tools_text = re.sub(r'\n+', ', ', tools_text)
            tools_text = re.sub(r'\s+', ' ', tools_text)
            if len(tools_text) > 10:
                return tools_text[:500]
        
        return None


class MaterialsExtractor(BaseExtractor):
    """Extractor de materiales."""
    
    def extract(self, content: str, **kwargs) -> Optional[str]:
        """Extraer lista de materiales."""
        materials_pattern = r'MATERIALES[:\s]+(.*?)(?=\n\n|\n\d+\.|$)'
        match = re.search(materials_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if match:
            materials_text = match.group(1).strip()
            materials_text = re.sub(r'\n+', ', ', materials_text)
            materials_text = re.sub(r'\s+', ' ', materials_text)
            if len(materials_text) > 10:
                return materials_text[:500]
        
        return None


class SafetyExtractor(BaseExtractor):
    """Extractor de advertencias de seguridad."""
    
    def extract(self, content: str, **kwargs) -> Optional[str]:
        """Extraer advertencias de seguridad."""
        safety_pattern = r'ADVERTENCIAS?\s+DE\s+SEGURIDAD[:\s]+(.*?)(?=\n\n|\n\d+\.|$)'
        match = re.search(safety_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if match:
            safety_text = match.group(1).strip()
            if len(safety_text) > 10:
                return safety_text[:1000]
        
        return None


class TagsExtractor(BaseExtractor):
    """Extractor de tags."""
    
    def extract(self, content: str, category: str = "", **kwargs) -> List[str]:
        """Extraer tags del manual."""
        tags = [category] if category else []
        
        content_lower = content.lower()
        
        tag_keywords = {
            "emergencia": ["urgente", "emergencia", "inmediato"],
            "principiante": ["fácil", "simple", "básico"],
            "avanzado": ["complejo", "avanzado", "experto"],
            "herramientas-especiales": ["herramienta especial", "equipo especial"],
            "materiales-especiales": ["material especial", "componente especial"],
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        
        return tags[:10]

