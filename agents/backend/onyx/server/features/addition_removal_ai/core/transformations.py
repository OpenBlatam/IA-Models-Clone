"""
Transformations - Sistema de transformaciones de contenido
"""

import logging
import re
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Transformation(ABC):
    """Clase base para transformaciones"""

    @abstractmethod
    def transform(self, content: str, **kwargs) -> str:
        """
        Transformar contenido.

        Args:
            content: Contenido a transformar
            **kwargs: Argumentos adicionales

        Returns:
            Contenido transformado
        """
        pass


class CaseTransformation(Transformation):
    """Transformación de mayúsculas/minúsculas"""

    def transform(self, content: str, case: str = "lower", **kwargs) -> str:
        """
        Transformar caso.

        Args:
            content: Contenido
            case: Tipo de caso (lower, upper, title, capitalize)

        Returns:
            Contenido transformado
        """
        if case == "lower":
            return content.lower()
        elif case == "upper":
            return content.upper()
        elif case == "title":
            return content.title()
        elif case == "capitalize":
            return content.capitalize()
        return content


class WhitespaceTransformation(Transformation):
    """Transformación de espacios en blanco"""

    def transform(self, content: str, action: str = "normalize", **kwargs) -> str:
        """
        Transformar espacios en blanco.

        Args:
            content: Contenido
            action: Acción (normalize, remove, trim)

        Returns:
            Contenido transformado
        """
        if action == "normalize":
            # Normalizar espacios múltiples
            return re.sub(r'\s+', ' ', content)
        elif action == "remove":
            # Remover todos los espacios
            return re.sub(r'\s+', '', content)
        elif action == "trim":
            # Eliminar espacios al inicio y final
            return content.strip()
        return content


class LineTransformation(Transformation):
    """Transformación de líneas"""

    def transform(self, content: str, action: str = "normalize", **kwargs) -> str:
        """
        Transformar líneas.

        Args:
            content: Contenido
            action: Acción (normalize, remove_empty, join)

        Returns:
            Contenido transformado
        """
        lines = content.split('\n')
        
        if action == "normalize":
            # Normalizar saltos de línea
            return '\n'.join(line.rstrip() for line in lines)
        elif action == "remove_empty":
            # Remover líneas vacías
            return '\n'.join(line for line in lines if line.strip())
        elif action == "join":
            # Unir todas las líneas
            return ' '.join(line.strip() for line in lines if line.strip())
        
        return content


class EncodingTransformation(Transformation):
    """Transformación de codificación"""

    def transform(self, content: str, action: str = "normalize", **kwargs) -> str:
        """
        Transformar codificación.

        Args:
            content: Contenido
            action: Acción (normalize, remove_accents, ascii)

        Returns:
            Contenido transformado
        """
        if action == "normalize":
            # Normalizar Unicode
            import unicodedata
            return unicodedata.normalize('NFKD', content)
        elif action == "remove_accents":
            # Remover acentos (simplificado)
            import unicodedata
            nfd = unicodedata.normalize('NFD', content)
            return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        elif action == "ascii":
            # Convertir a ASCII
            return content.encode('ascii', 'ignore').decode('ascii')
        
        return content


class TransformationPipeline:
    """Pipeline de transformaciones"""

    def __init__(self):
        """Inicializar pipeline"""
        self.transformations: List[tuple[Transformation, Dict[str, Any]]] = []

    def add_transformation(
        self,
        transformation: Transformation,
        **kwargs
    ):
        """
        Agregar transformación al pipeline.

        Args:
            transformation: Transformación
            **kwargs: Argumentos para la transformación
        """
        self.transformations.append((transformation, kwargs))
        logger.info(f"Transformación agregada: {type(transformation).__name__}")

    def apply(self, content: str) -> str:
        """
        Aplicar todas las transformaciones.

        Args:
            content: Contenido

        Returns:
            Contenido transformado
        """
        result = content
        for transformation, kwargs in self.transformations:
            result = transformation.transform(result, **kwargs)
        return result

    def clear(self):
        """Limpiar pipeline"""
        self.transformations.clear()






