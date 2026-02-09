"""
Helper functions for string operations.
Eliminates repetitive string manipulation patterns.
"""

from typing import List, Optional, Callable
import re


def truncate(
    text: str,
    max_length: int,
    suffix: str = "...",
    preserve_words: bool = True
) -> str:
    """
    Trunca un texto a una longitud máxima.
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        suffix: Sufijo a agregar si se trunca (default: "...")
        preserve_words: Si preservar palabras completas (default: True)
        
    Returns:
        Texto truncado
        
    Usage:
        >>> truncate("This is a long text", 10)
        'This is a...'
        
        >>> truncate("This is a long text", 10, preserve_words=False)
        'This is a ...'
    """
    if len(text) <= max_length:
        return text
    
    if preserve_words:
        # Truncar en el último espacio antes de max_length
        truncated = text[:max_length - len(suffix)]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + suffix
    
    return text[:max_length - len(suffix)] + suffix


def extract_hashtags(text: str) -> List[str]:
    """
    Extrae hashtags de un texto.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Lista de hashtags (sin #)
        
    Usage:
        >>> extract_hashtags("Check out #python #coding")
        ['python', 'coding']
    """
    hashtags = re.findall(r'#\w+', text)
    return [tag[1:] for tag in hashtags]  # Remover el #


def extract_mentions(text: str) -> List[str]:
    """
    Extrae menciones (@username) de un texto.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Lista de menciones (sin @)
        
    Usage:
        >>> extract_mentions("Hey @john and @jane")
        ['john', 'jane']
    """
    mentions = re.findall(r'@\w+', text)
    return [mention[1:] for mention in mentions]  # Remover el @


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitiza un nombre de archivo removiendo caracteres inválidos.
    
    Args:
        filename: Nombre de archivo a sanitizar
        max_length: Longitud máxima (default: 255)
        
    Returns:
        Nombre de archivo sanitizado
        
    Usage:
        >>> sanitize_filename("my file@name.txt")
        'my_file_name.txt'
    """
    # Remover caracteres inválidos
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remover espacios múltiples
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Truncar si es muy largo
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def normalize_whitespace(text: str) -> str:
    """
    Normaliza espacios en blanco (múltiples espacios a uno solo).
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
        
    Usage:
        >>> normalize_whitespace("This   has    multiple   spaces")
        'This has multiple spaces'
    """
    return re.sub(r'\s+', ' ', text).strip()


def slugify(text: str, separator: str = "-") -> str:
    """
    Convierte un texto a slug (URL-friendly).
    
    Args:
        text: Texto a convertir
        separator: Separador a usar (default: "-")
        
    Returns:
        Slug generado
        
    Usage:
        >>> slugify("Hello World!")
        'hello-world'
    """
    # Convertir a minúsculas
    text = text.lower()
    # Remover caracteres especiales
    text = re.sub(r'[^\w\s-]', '', text)
    # Reemplazar espacios con separador
    text = re.sub(r'[\s_]+', separator, text)
    # Remover separadores al inicio/final
    text = text.strip(separator)
    return text


def ellipsize(text: str, max_length: int = 50) -> str:
    """
    Agrega ellipsis si el texto es muy largo.
    
    Args:
        text: Texto a procesar
        max_length: Longitud máxima antes de agregar ellipsis
        
    Returns:
        Texto con ellipsis si es necesario
        
    Usage:
        >>> ellipsize("This is a very long text", 10)
        'This is a...'
    """
    return truncate(text, max_length, suffix="...")


def capitalize_words(text: str) -> str:
    """
    Capitaliza cada palabra en un texto.
    
    Args:
        text: Texto a capitalizar
        
    Returns:
        Texto capitalizado
        
    Usage:
        >>> capitalize_words("hello world")
        'Hello World'
    """
    return ' '.join(word.capitalize() for word in text.split())


def remove_emojis(text: str) -> str:
    """
    Remueve emojis de un texto.
    
    Args:
        text: Texto a procesar
        
    Returns:
        Texto sin emojis
        
    Usage:
        >>> remove_emojis("Hello 😀 World 🎉")
        'Hello  World '
    """
    # Patrón para emojis Unicode
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)








