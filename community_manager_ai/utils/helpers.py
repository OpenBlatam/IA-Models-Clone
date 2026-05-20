"""
Helpers - Funciones Auxiliares
===============================

Funciones auxiliares para el sistema.
"""

import re
import uuid
from typing import Optional, List
from datetime import datetime


def generate_post_id() -> str:
    """
    Generar ID único para un post
    
    Returns:
        ID único
    """
    return str(uuid.uuid4())


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Formatear datetime a string
    
    Args:
        dt: Datetime a formatear
        format_str: Formato deseado
        
    Returns:
        String formateado
    """
    return dt.strftime(format_str)


def sanitize_content(content: str) -> str:
    """
    Sanitizar contenido (remover caracteres peligrosos)
    
    Args:
        content: Contenido a sanitizar
        
    Returns:
        Contenido sanitizado
    """
    # Remover caracteres de control excepto newlines y tabs
    content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', content)
    
    # Normalizar espacios múltiples
    content = re.sub(r' +', ' ', content)
    
    # Remover espacios al inicio y final de cada línea
    lines = [line.strip() for line in content.split('\n')]
    content = '\n'.join(lines)
    
    return content.strip()


def truncate_content(content: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncar contenido a una longitud máxima
    
    Args:
        content: Contenido a truncar
        max_length: Longitud máxima
        suffix: Sufijo a agregar si se trunca
        
    Returns:
        Contenido truncado
    """
    if len(content) <= max_length:
        return content
    
    return content[:max_length - len(suffix)] + suffix


def extract_hashtags(content: str) -> List[str]:
    """
    Extraer hashtags de un contenido
    
    Args:
        content: Contenido a analizar
        
    Returns:
        Lista de hashtags encontrados
    """
    hashtags = re.findall(r'#\w+', content)
    return list(set(hashtags))  # Remover duplicados


def extract_mentions(content: str) -> List[str]:
    """
    Extraer menciones (@username) de un contenido
    
    Args:
        content: Contenido a analizar
        
    Returns:
        Lista de menciones encontradas
    """
    mentions = re.findall(r'@\w+', content)
    return list(set(mentions))  # Remover duplicados


def calculate_engagement_rate(
    likes: int,
    comments: int,
    shares: int,
    reach: int
) -> float:
    """
    Calcular engagement rate
    
    Args:
        likes: Número de likes
        comments: Número de comentarios
        shares: Número de compartidos
        reach: Alcance del post
        
    Returns:
        Engagement rate como porcentaje
    """
    if reach == 0:
        return 0.0
    
    total_engagement = likes + comments + shares
    return (total_engagement / reach) * 100


def format_engagement_rate(rate: float) -> str:
    """
    Formatear engagement rate como string
    
    Args:
        rate: Engagement rate
        
    Returns:
        String formateado (ej: "5.23%")
    """
    return f"{rate:.2f}%"


def format_number(num: int) -> str:
    """
    Formatear número con separadores de miles
    
    Args:
        num: Número a formatear
        
    Returns:
        String formateado (ej: "1,234,567")
    """
    return f"{num:,}"


def format_file_size(size_bytes: int) -> str:
    """
    Formatear tamaño de archivo en formato legible
    
    Args:
        size_bytes: Tamaño en bytes
        
    Returns:
        String formateado (ej: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def parse_datetime(date_string: str) -> Optional[datetime]:
    """
    Parsear string de fecha en múltiples formatos
    
    Args:
        date_string: String de fecha
        
    Returns:
        Datetime o None si no se puede parsear
    """
    from dateutil import parser
    
    try:
        return parser.parse(date_string)
    except Exception:
        return None


def is_valid_url(url: str) -> bool:
    """
    Validar si una URL es válida
    
    Args:
        url: URL a validar
        
    Returns:
        True si es válida
    """
    import re
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return url_pattern.match(url) is not None


def clean_filename(filename: str) -> str:
    """
    Limpiar nombre de archivo removiendo caracteres inválidos
    
    Args:
        filename: Nombre de archivo
        
    Returns:
        Nombre de archivo limpio
    """
    import re
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip('. ')
    return filename[:255]


def generate_slug(text: str) -> str:
    """
    Generar slug a partir de texto
    
    Args:
        text: Texto a convertir
        
    Returns:
        Slug generado
    """
    import re
    import unicodedata
    
    try:
        text = unicodedata.normalize('NFKD', text)
    except Exception:
        pass
    
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def chunk_text(text: str, max_length: int, overlap: int = 0) -> List[str]:
    """
    Dividir texto en chunks de longitud máxima
    
    Args:
        text: Texto a dividir
        max_length: Longitud máxima por chunk
        overlap: Caracteres de solapamiento entre chunks
        
    Returns:
        Lista de chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        chunk = text[start:end]
        
        last_space = chunk.rfind(' ')
        if last_space > max_length * 0.5:
            end = start + last_space
            chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def extract_urls(text: str) -> List[str]:
    """
    Extraer URLs de un texto
    
    Args:
        text: Texto a analizar
        
    Returns:
        Lista de URLs encontradas
    """
    import re
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)


def count_words(text: str) -> int:
    """
    Contar palabras en un texto
    
    Args:
        text: Texto a analizar
        
    Returns:
        Número de palabras
    """
    return len(text.split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Contar caracteres en un texto
    
    Args:
        text: Texto a analizar
        include_spaces: Incluir espacios en el conteo
        
    Returns:
        Número de caracteres
    """
    if include_spaces:
        return len(text)
    return len(text.replace(' ', '').replace('\n', '').replace('\t', ''))


def get_reading_time(text: str, words_per_minute: int = 200) -> float:
    """
    Calcular tiempo de lectura estimado
    
    Args:
        text: Texto a analizar
        words_per_minute: Palabras por minuto (default: 200)
        
    Returns:
        Tiempo en minutos
    """
    words = count_words(text)
    return words / words_per_minute



