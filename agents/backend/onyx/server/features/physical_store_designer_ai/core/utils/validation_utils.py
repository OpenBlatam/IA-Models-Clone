"""
Validation Utils

Utilities for validation of store types, design styles, and other common validations.
"""


def validate_store_type(store_type: str) -> bool:
    """
    Validar tipo de tienda
    
    Args:
        store_type: Tipo de tienda a validar
        
    Returns:
        True si el tipo es válido, False en caso contrario
    """
    valid_types = [
        "retail", "restaurant", "cafe", "boutique",
        "supermarket", "pharmacy", "electronics",
        "clothing", "furniture", "other"
    ]
    return store_type.lower() in valid_types if isinstance(store_type, str) else False


def validate_design_style(style: str) -> bool:
    """
    Validar estilo de diseño
    
    Args:
        style: Estilo de diseño a validar
        
    Returns:
        True si el estilo es válido, False en caso contrario
    """
    valid_styles = [
        "modern", "classic", "minimalist", "industrial",
        "rustic", "luxury", "eco_friendly", "vintage"
    ]
    return style.lower() in valid_styles if isinstance(style, str) else False

