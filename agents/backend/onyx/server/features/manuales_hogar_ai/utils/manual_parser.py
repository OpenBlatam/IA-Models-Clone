"""
Parser de Manuales
==================

Utilidades para extraer información estructurada de manuales generados.
"""

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ManualParser:
    """Parser para extraer información de manuales."""
    
    @staticmethod
    def extract_title(manual_content: str) -> Optional[str]:
        """
        Extraer título del manual.
        
        Args:
            manual_content: Contenido del manual
        
        Returns:
            Título extraído o None
        """
        # Buscar título en formato "1. TÍTULO DEL MANUAL" o similar
        patterns = [
            r'^1\.\s*TÍTULO[:\s]+(.+?)$',
            r'^#\s*(.+?)$',
            r'^TÍTULO[:\s]+(.+?)$',
            r'^(.+?)\n.*DIAGNÓSTICO',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, manual_content, re.MULTILINE | re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 10 and len(title) < 200:
                    return title
        
        # Si no se encuentra, usar primeras palabras
        first_line = manual_content.split('\n')[0].strip()
        if len(first_line) > 10:
            return first_line[:200]
        
        return None
    
    @staticmethod
    def extract_difficulty(manual_content: str) -> Optional[str]:
        """
        Extraer dificultad del manual.
        
        Args:
            manual_content: Contenido del manual
        
        Returns:
            Dificultad (Fácil, Media, Difícil) o None
        """
        content_lower = manual_content.lower()
        
        if any(word in content_lower for word in ['dificultad: fácil', 'dificultad fácil', 'fácil']):
            return "Fácil"
        elif any(word in content_lower for word in ['dificultad: difícil', 'dificultad difícil', 'difícil']):
            return "Difícil"
        elif any(word in content_lower for word in ['dificultad: media', 'dificultad media', 'media']):
            return "Media"
        
        return None
    
    @staticmethod
    def extract_estimated_time(manual_content: str) -> Optional[str]:
        """
        Extraer tiempo estimado.
        
        Args:
            manual_content: Contenido del manual
        
        Returns:
            Tiempo estimado o None
        """
        patterns = [
            r'⏱️\s*Tiempo[:\s]+(.+?)(?:\n|$)',
            r'Tiempo[:\s]+(.+?)(?:\n|$)',
            r'(\d+\s*(?:minutos?|horas?|días?))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, manual_content, re.IGNORECASE | re.MULTILINE)
            if match:
                time_str = match.group(1).strip()
                if len(time_str) < 50:
                    return time_str
        
        return None
    
    @staticmethod
    def extract_tools(manual_content: str) -> Optional[str]:
        """
        Extraer lista de herramientas.
        
        Args:
            manual_content: Contenido del manual
        
        Returns:
            Lista de herramientas o None
        """
        # Buscar sección de herramientas
        tools_pattern = r'HERRAMIENTAS[:\s]+(.*?)(?=\n\n|\n\d+\.|$)'
        match = re.search(tools_pattern, manual_content, re.IGNORECASE | re.DOTALL)
        
        if match:
            tools_text = match.group(1).strip()
            # Limpiar y formatear
            tools_text = re.sub(r'\n+', ', ', tools_text)
            tools_text = re.sub(r'\s+', ' ', tools_text)
            if len(tools_text) > 10:
                return tools_text[:500]
        
        return None
    
    @staticmethod
    def extract_materials(manual_content: str) -> Optional[str]:
        """
        Extraer lista de materiales.
        
        Args:
            manual_content: Contenido del manual
        
        Returns:
            Lista de materiales o None
        """
        # Buscar sección de materiales
        materials_pattern = r'MATERIALES[:\s]+(.*?)(?=\n\n|\n\d+\.|$)'
        match = re.search(materials_pattern, manual_content, re.IGNORECASE | re.DOTALL)
        
        if match:
            materials_text = match.group(1).strip()
            # Limpiar y formatear
            materials_text = re.sub(r'\n+', ', ', materials_text)
            materials_text = re.sub(r'\s+', ' ', materials_text)
            if len(materials_text) > 10:
                return materials_text[:500]
        
        return None
    
    @staticmethod
    def extract_safety_warnings(manual_content: str) -> Optional[str]:
        """
        Extraer advertencias de seguridad.
        
        Args:
            manual_content: Contenido del manual
        
        Returns:
            Advertencias de seguridad o None
        """
        # Buscar sección de seguridad
        safety_pattern = r'ADVERTENCIAS?\s+DE\s+SEGURIDAD[:\s]+(.*?)(?=\n\n|\n\d+\.|$)'
        match = re.search(safety_pattern, manual_content, re.IGNORECASE | re.DOTALL)
        
        if match:
            safety_text = match.group(1).strip()
            if len(safety_text) > 10:
                return safety_text[:1000]
        
        return None
    
    @staticmethod
    def extract_tags(manual_content: str, category: str) -> List[str]:
        """
        Extraer tags del manual.
        
        Args:
            manual_content: Contenido del manual
            category: Categoría del manual
        
        Returns:
            Lista de tags
        """
        tags = [category]
        
        # Agregar tags basados en contenido
        content_lower = manual_content.lower()
        
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
        
        return tags[:10]  # Máximo 10 tags
    
    @classmethod
    def parse_manual(cls, manual_content: str, category: str) -> Dict[str, Any]:
        """
        Parsear manual completo y extraer toda la información.
        
        Args:
            manual_content: Contenido del manual
            category: Categoría del manual
        
        Returns:
            Diccionario con información extraída
        """
        return {
            "title": cls.extract_title(manual_content),
            "difficulty": cls.extract_difficulty(manual_content),
            "estimated_time": cls.extract_estimated_time(manual_content),
            "tools_required": cls.extract_tools(manual_content),
            "materials_required": cls.extract_materials(manual_content),
            "safety_warnings": cls.extract_safety_warnings(manual_content),
            "tags": ",".join(cls.extract_tags(manual_content, category))
        }




