"""
Text Processing Utilities
==========================
Utilidades para procesar y analizar texto.
"""

import re
from typing import List, Dict, Any, Optional


def extract_sections(text: str) -> Dict[str, str]:
    """
    Extraer secciones de un texto estructurado.
    
    Args:
        text: Texto a procesar
        
    Returns:
        Diccionario con secciones extraídas
    """
    sections = {}
    
    # Buscar secciones con títulos (##, ###, o líneas con :)
    section_pattern = r'(?:^|\n)(?:#{1,3}\s*)?([A-Z][^:\n]+):\s*\n?(.+?)(?=\n(?:#{1,3}\s*)?[A-Z]|\Z)'
    matches = re.finditer(section_pattern, text, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        sections[title.lower().replace(' ', '_')] = content
    
    return sections


def extract_list_items(text: str) -> List[str]:
    """
    Extraer items de una lista del texto.
    
    Args:
        text: Texto con lista
        
    Returns:
        Lista de items extraídos
    """
    items = []
    
    # Buscar listas numeradas
    numbered = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|\Z)', text, re.DOTALL)
    items.extend([item.strip() for item in numbered])
    
    # Buscar listas con bullets
    bullets = re.findall(r'[•\-\*]\s*(.+?)(?=\n[•\-\*]|\n\n|\Z)', text, re.DOTALL)
    items.extend([item.strip() for item in bullets])
    
    return items[:10]  # Limitar a 10 items


def extract_duration(text: str) -> Optional[str]:
    """
    Extraer duración estimada del texto.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Duración extraída o None
    """
    # Buscar patrones de duración
    patterns = [
        r'(\d+)\s*(?:weeks?|semanas?)',
        r'(\d+)\s*(?:days?|días?)',
        r'(\d+)\s*(?:months?|meses?)',
        r'(\d+)\s*(?:hours?|horas?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None


def extract_phases(text: str) -> List[Dict[str, Any]]:
    """
    Extraer fases de un plan de entrenamiento.
    
    Args:
        text: Texto del plan
        
    Returns:
        Lista de fases extraídas
    """
    phases = []
    
    # Buscar fases con títulos
    phase_pattern = r'(?:Phase|Fase)\s*(\d+)[:]\s*(.+?)(?=(?:Phase|Fase)\s*\d+|\Z)'
    matches = re.finditer(phase_pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        phase_num = match.group(1)
        content = match.group(2).strip()
        phases.append({
            "phase": int(phase_num),
            "description": content,
            "duration": extract_duration(content)
        })
    
    return phases

