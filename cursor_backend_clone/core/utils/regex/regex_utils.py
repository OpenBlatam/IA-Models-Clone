"""
Regex Utils - Utilidades de Regex Avanzadas
===========================================

Utilidades avanzadas para expresiones regulares.
"""

import logging
import re
from typing import List, Optional, Dict, Any, Pattern, Match, Callable

logger = logging.getLogger(__name__)


def find_all_matches(pattern: str, text: str, flags: int = 0) -> List[Match]:
    """
    Encontrar todas las coincidencias de un patrón.
    
    Args:
        pattern: Patrón regex
        text: Texto a buscar
        flags: Flags de regex
        
    Returns:
        Lista de matches
    """
    return list(re.finditer(pattern, text, flags))


def find_first_match(pattern: str, text: str, flags: int = 0) -> Optional[Match]:
    """
    Encontrar la primera coincidencia.
    
    Args:
        pattern: Patrón regex
        text: Texto a buscar
        flags: Flags de regex
        
    Returns:
        Primer match o None
    """
    match = re.search(pattern, text, flags)
    return match


def extract_groups(pattern: str, text: str, flags: int = 0) -> List[Dict[str, Any]]:
    """
    Extraer grupos nombrados de todas las coincidencias.
    
    Args:
        pattern: Patrón regex con grupos nombrados
        text: Texto a buscar
        flags: Flags de regex
        
    Returns:
        Lista de diccionarios con grupos
    """
    matches = find_all_matches(pattern, text, flags)
    results = []
    
    for match in matches:
        groups = match.groupdict()
        groups['full_match'] = match.group(0)
        groups['start'] = match.start()
        groups['end'] = match.end()
        results.append(groups)
    
    return results


def replace_with_callback(
    pattern: str,
    text: str,
    callback: Callable[[Match], str],
    flags: int = 0
) -> str:
    """
    Reemplazar usando callback.
    
    Args:
        pattern: Patrón regex
        text: Texto a procesar
        callback: Función que recibe Match y retorna string
        flags: Flags de regex
        
    Returns:
        Texto reemplazado
    """
    def replacer(match: Match) -> str:
        return callback(match)
    
    return re.sub(pattern, replacer, text, flags=flags)


def split_keep_delimiter(pattern: str, text: str, flags: int = 0) -> List[str]:
    """
    Dividir texto manteniendo delimitadores.
    
    Args:
        pattern: Patrón regex delimitador
        text: Texto a dividir
        flags: Flags de regex
        
    Returns:
        Lista de partes incluyendo delimitadores
    """
    parts = []
    last_end = 0
    
    for match in re.finditer(pattern, text, flags):
        # Parte antes del match
        if match.start() > last_end:
            parts.append(text[last_end:match.start()])
        
        # Delimitador
        parts.append(match.group(0))
        last_end = match.end()
    
    # Parte final
    if last_end < len(text):
        parts.append(text[last_end:])
    
    return parts


def validate_pattern(pattern: str) -> tuple[bool, Optional[str]]:
    """
    Validar patrón regex.
    
    Args:
        pattern: Patrón a validar
        
    Returns:
        Tupla (es_válido, mensaje_error)
    """
    try:
        re.compile(pattern)
        return True, None
    except re.error as e:
        return False, str(e)


def escape_regex(text: str) -> str:
    """
    Escapar caracteres especiales de regex.
    
    Args:
        text: Texto a escapar
        
    Returns:
        Texto escapado
    """
    return re.escape(text)


def compile_pattern(pattern: str, flags: int = 0) -> Pattern:
    """
    Compilar patrón regex (con caché opcional).
    
    Args:
        pattern: Patrón regex
        flags: Flags de regex
        
    Returns:
        Pattern compilado
    """
    return re.compile(pattern, flags)


class RegexMatcher:
    """
    Matcher de regex con múltiples patrones.
    """
    
    def __init__(self):
        self.patterns: List[Pattern] = []
        self.named_patterns: Dict[str, Pattern] = {}
    
    def add_pattern(self, pattern: str, name: Optional[str] = None, flags: int = 0) -> None:
        """
        Agregar patrón.
        
        Args:
            pattern: Patrón regex
            name: Nombre opcional del patrón
            flags: Flags de regex
        """
        compiled = re.compile(pattern, flags)
        
        if name:
            self.named_patterns[name] = compiled
        else:
            self.patterns.append(compiled)
    
    def match(self, text: str) -> List[Dict[str, Any]]:
        """
        Buscar coincidencias de todos los patrones.
        
        Args:
            text: Texto a buscar
            
        Returns:
            Lista de matches con información
        """
        results = []
        
        # Patrones sin nombre
        for i, pattern in enumerate(self.patterns):
            for match in pattern.finditer(text):
                results.append({
                    'pattern_index': i,
                    'match': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'groups': match.groups()
                })
        
        # Patrones con nombre
        for name, pattern in self.named_patterns.items():
            for match in pattern.finditer(text):
                results.append({
                    'pattern_name': name,
                    'match': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'groups': match.groups()
                })
        
        return results
    
    def match_first(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Buscar primera coincidencia.
        
        Args:
            text: Texto a buscar
            
        Returns:
            Primer match o None
        """
        matches = self.match(text)
        return matches[0] if matches else None


# Patrones comunes predefinidos
COMMON_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'url': r'https?://[^\s<>"{}|\\^`\[\]]+',
    'ipv4': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'ipv6': r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    'time': r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b',
    'hashtag': r'#\w+',
    'mention': r'@\w+',
    'hex_color': r'#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})\b',
    'uuid': r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
}


def get_common_pattern(name: str) -> Optional[str]:
    """
    Obtener patrón común predefinido.
    
    Args:
        name: Nombre del patrón
        
    Returns:
        Patrón regex o None
    """
    return COMMON_PATTERNS.get(name)


def extract_by_pattern(pattern_name: str, text: str) -> List[str]:
    """
    Extraer coincidencias usando patrón común.
    
    Args:
        pattern_name: Nombre del patrón común
        text: Texto a buscar
        
    Returns:
        Lista de coincidencias
    """
    pattern = get_common_pattern(pattern_name)
    if not pattern:
        return []
    
    matches = find_all_matches(pattern, text, re.IGNORECASE)
    return [match.group(0) for match in matches]




