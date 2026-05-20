"""
Text Utils - Utilidades Avanzadas de Texto
==========================================

Utilidades para manipulación y procesamiento de texto.
"""

import logging
import re
import unicodedata
from typing import List, Optional, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)


def slugify(text: str, separator: str = "-") -> str:
    """
    Convertir texto a slug (URL-friendly).
    
    Args:
        text: Texto a convertir
        separator: Separador a usar
        
    Returns:
        Slug generado
    """
    # Normalizar unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Reemplazar espacios y caracteres especiales
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', separator, text)
    
    # Eliminar separadores al inicio y final
    text = text.strip(separator)
    
    return text


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncar texto a longitud máxima.
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        suffix: Sufijo a agregar si se trunca
        
    Returns:
        Texto truncado
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def truncate_words(text: str, max_words: int, suffix: str = "...") -> str:
    """
    Truncar texto por número de palabras.
    
    Args:
        text: Texto a truncar
        max_words: Número máximo de palabras
        suffix: Sufijo a agregar si se trunca
        
    Returns:
        Texto truncado
    """
    words = text.split()
    
    if len(words) <= max_words:
        return text
    
    truncated = " ".join(words[:max_words])
    return truncated + suffix


def extract_words(text: str, min_length: int = 1) -> List[str]:
    """
    Extraer palabras de texto.
    
    Args:
        text: Texto del cual extraer palabras
        min_length: Longitud mínima de palabras
        
    Returns:
        Lista de palabras
    """
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if len(w) >= min_length]


def count_words(text: str) -> int:
    """
    Contar palabras en texto.
    
    Args:
        text: Texto a contar
        
    Returns:
        Número de palabras
    """
    return len(extract_words(text))


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Contar caracteres en texto.
    
    Args:
        text: Texto a contar
        include_spaces: Si incluir espacios
        
    Returns:
        Número de caracteres
    """
    if include_spaces:
        return len(text)
    return len(text.replace(" ", ""))


def remove_accents(text: str) -> str:
    """
    Remover acentos de texto.
    
    Args:
        text: Texto del cual remover acentos
        
    Returns:
        Texto sin acentos
    """
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def normalize_whitespace(text: str) -> str:
    """
    Normalizar espacios en blanco.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    # Reemplazar múltiples espacios con uno solo
    text = re.sub(r'\s+', ' ', text)
    # Eliminar espacios al inicio y final
    return text.strip()


def extract_emails(text: str) -> List[str]:
    """
    Extraer emails de texto.
    
    Args:
        text: Texto del cual extraer emails
        
    Returns:
        Lista de emails encontrados
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_urls(text: str) -> List[str]:
    """
    Extraer URLs de texto.
    
    Args:
        text: Texto del cual extraer URLs
        
    Returns:
        Lista de URLs encontradas
    """
    pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(pattern, text)


def extract_hashtags(text: str) -> List[str]:
    """
    Extraer hashtags de texto.
    
    Args:
        text: Texto del cual extraer hashtags
        
    Returns:
        Lista de hashtags encontrados
    """
    pattern = r'#\w+'
    return re.findall(pattern, text)


def extract_mentions(text: str) -> List[str]:
    """
    Extraer menciones (@username) de texto.
    
    Args:
        text: Texto del cual extraer menciones
        
    Returns:
        Lista de menciones encontradas
    """
    pattern = r'@\w+'
    return re.findall(pattern, text)


def mask_sensitive_data(text: str, mask_char: str = "*") -> str:
    """
    Enmascarar datos sensibles (emails, números de teléfono, etc.).
    
    Args:
        text: Texto a enmascarar
        mask_char: Carácter para enmascarar
        
    Returns:
        Texto enmascarado
    """
    # Enmascarar emails
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        lambda m: mask_char * len(m.group()),
        text
    )
    
    # Enmascarar números de teléfono (formato básico)
    text = re.sub(
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        lambda m: mask_char * len(m.group()),
        text
    )
    
    return text


def highlight_keywords(text: str, keywords: List[str], tag: str = "**") -> str:
    """
    Resaltar palabras clave en texto.
    
    Args:
        text: Texto en el cual resaltar
        keywords: Lista de palabras clave
        tag: Tag para resaltar (ej: **, <mark>, etc.)
        
    Returns:
        Texto con palabras clave resaltadas
    """
    for keyword in keywords:
        pattern = re.escape(keyword)
        text = re.sub(
            pattern,
            f"{tag}{keyword}{tag}",
            text,
            flags=re.IGNORECASE
        )
    
    return text


def calculate_readability(text: str) -> Dict[str, float]:
    """
    Calcular métricas de legibilidad básicas.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Diccionario con métricas
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = extract_words(text)
    
    if not sentences or not words:
        return {
            "avg_words_per_sentence": 0.0,
            "avg_chars_per_word": 0.0,
            "avg_chars_per_sentence": 0.0
        }
    
    return {
        "avg_words_per_sentence": len(words) / len(sentences),
        "avg_chars_per_word": sum(len(w) for w in words) / len(words) if words else 0,
        "avg_chars_per_sentence": sum(len(s) for s in sentences) / len(sentences)
    }


def find_most_common_words(text: str, top_n: int = 10) -> List[tuple]:
    """
    Encontrar palabras más comunes.
    
    Args:
        text: Texto a analizar
        top_n: Número de palabras a retornar
        
    Returns:
        Lista de tuplas (palabra, frecuencia)
    """
    words = extract_words(text)
    counter = Counter(words)
    return counter.most_common(top_n)


def remove_html_tags(text: str) -> str:
    """
    Remover tags HTML de texto.
    
    Args:
        text: Texto con HTML
        
    Returns:
        Texto sin tags HTML
    """
    return re.sub(r'<[^>]+>', '', text)


def extract_text_from_html(html: str) -> str:
    """
    Extraer texto de HTML.
    
    Args:
        html: HTML del cual extraer texto
        
    Returns:
        Texto extraído
    """
    text = remove_html_tags(html)
    text = normalize_whitespace(text)
    return text




