"""
Response Formatting Utilities
=============================
"""

from typing import Dict, List, Any
import re


def extract_key_points(text: str) -> List[str]:
    """Extraer puntos clave de un texto."""
    # Buscar listas numeradas o con bullets
    points = re.findall(r'(?:^|\n)[•\-\*]\s*(.+?)(?=\n|$)', text, re.MULTILINE)
    if not points:
        # Buscar números
        points = re.findall(r'(?:^|\n)\d+\.\s*(.+?)(?=\n|$)', text, re.MULTILINE)
    return points[:5]  # Limitar a 5 puntos


def extract_next_steps(text: str) -> List[str]:
    """Extraer próximos pasos de un texto."""
    # Buscar secciones de "next steps" o "próximos pasos"
    next_steps_pattern = r'(?:next steps?|próximos pasos?|siguientes pasos?)[:]\s*(.+?)(?=\n\n|\Z)'
    match = re.search(next_steps_pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        steps_text = match.group(1)
        return extract_key_points(steps_text)
    return []


def format_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Formatear respuesta con extracción de datos estructurados."""
    if "advice" in response and response.get("advice"):
        response["key_points"] = extract_key_points(response["advice"])
        response["next_steps"] = extract_next_steps(response["advice"])
    return response

