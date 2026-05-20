"""
Validators - Utilidades de validación centralizadas.

Siguiendo DRY, todas las validaciones comunes están aquí.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, List, Optional

from .exceptions import AudioValidationError, AudioFormatError, AudioIOError


def validate_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_file: bool = True
) -> Path:
    """
    Valida y normaliza una ruta.
    
    Args:
        path: Ruta a validar
        must_exist: Si True, el archivo debe existir
        must_be_file: Si True, debe ser un archivo (no directorio)
    
    Returns:
        Path validado y normalizado
    
    Raises:
        AudioIOError: Si el archivo no existe cuando se requiere
        AudioValidationError: Si la validación falla
    """
    path = Path(path).resolve()
    
    if must_exist and not path.exists():
        raise AudioIOError(f"File not found: {path}")
    
    if must_exist and must_be_file and not path.is_file():
        raise AudioValidationError(f"Path is not a file: {path}")
    
    return path


def validate_output_path(
    path: Union[str, Path],
    create_parent: bool = True
) -> Path:
    """
    Valida y prepara una ruta de salida.
    
    Args:
        path: Ruta de salida
        create_parent: Si True, crea el directorio padre si no existe
    
    Returns:
        Path validado
    """
    path = Path(path).resolve()
    
    if path.exists() and path.is_dir():
        raise AudioValidationError(f"Output path is a directory, not a file: {path}")
    
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def validate_output_dir(
    path: Union[str, Path],
    create: bool = True
) -> Path:
    """
    Valida y crea un directorio de salida.
    
    Args:
        path: Ruta al directorio
        create: Si True, crea el directorio si no existe
    
    Returns:
        Path validado
    """
    path = Path(path).resolve()
    
    if path.exists() and not path.is_dir():
        raise AudioValidationError(f"Path exists but is not a directory: {path}")
    
    if create:
        path.mkdir(parents=True, exist_ok=True)
    
    return path


def validate_format(
    path: Union[str, Path],
    supported_formats: List[str],
    component_name: str = "component"
) -> None:
    """
    Valida que el formato de un archivo esté soportado.
    
    Args:
        path: Ruta al archivo
        supported_formats: Lista de extensiones soportadas (con o sin punto)
        component_name: Nombre del componente para mensajes de error
    
    Raises:
        AudioFormatError: Si el formato no está soportado
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    # Normalizar formatos (agregar punto si no lo tiene)
    normalized_formats = [
        f if f.startswith(".") else f".{f}"
        for f in supported_formats
    ]
    
    if suffix not in normalized_formats:
        raise AudioFormatError(
            f"Unsupported format '{suffix}' for {component_name}. "
            f"Supported formats: {', '.join(normalized_formats)}"
        )


def validate_volume(
    volume: float,
    name: str = "volume"
) -> None:
    """
    Valida que un volumen esté en el rango válido.
    
    Args:
        volume: Volumen a validar (0.0-1.0)
        name: Nombre del parámetro para mensajes de error
    
    Raises:
        AudioValidationError: Si el volumen no es válido
        AudioConfigurationError: Si el volumen no es válido (para configs)
    """
    if not isinstance(volume, (int, float)):
        from .exceptions import AudioConfigurationError
        raise AudioConfigurationError(f"{name} must be a number, got {type(volume)}")
    
    if not 0.0 <= volume <= 1.0:
        from .exceptions import AudioConfigurationError
        raise AudioConfigurationError(
            f"{name} must be between 0.0 and 1.0, got {volume}"
        )


def validate_components(
    components: List[str],
    supported: List[str],
    component_name: str = "component"
) -> None:
    """
    Valida que los componentes solicitados estén soportados.
    
    Args:
        components: Lista de componentes solicitados
        supported: Lista de componentes soportados
        component_name: Nombre del componente para mensajes de error
    
    Raises:
        AudioValidationError: Si algún componente no está soportado
    """
    invalid = [c for c in components if c not in supported]
    if invalid:
        raise AudioValidationError(
            f"Unsupported components {invalid} for {component_name}. "
            f"Supported: {supported}"
        )


# ════════════════════════════════════════════════════════════════════════════
# CONFIG VALIDATORS (for AudioConfig and subclasses)
# ════════════════════════════════════════════════════════════════════════════

def validate_sample_rate(sample_rate: int, name: str = "sample_rate") -> None:
    """
    Valida que el sample rate sea positivo.
    
    Args:
        sample_rate: Sample rate a validar
        name: Nombre del parámetro para mensajes de error
    
    Raises:
        AudioConfigurationError: Si el sample rate no es válido
    """
    from .exceptions import AudioConfigurationError
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioConfigurationError(f"{name} must be a positive integer, got {sample_rate}")


def validate_channels(channels: int, name: str = "channels") -> None:
    """
    Valida que el número de canales sea válido (1 o 2).
    
    Args:
        channels: Número de canales a validar
        name: Nombre del parámetro para mensajes de error
    
    Raises:
        AudioConfigurationError: Si el número de canales no es válido
    """
    from .exceptions import AudioConfigurationError
    if channels not in [1, 2]:
        raise AudioConfigurationError(f"{name} must be 1 (mono) or 2 (stereo), got {channels}")


def validate_bit_depth(bit_depth: int, name: str = "bit_depth") -> None:
    """
    Valida que el bit depth sea válido (16, 24, o 32).
    
    Args:
        bit_depth: Bit depth a validar
        name: Nombre del parámetro para mensajes de error
    
    Raises:
        AudioConfigurationError: Si el bit depth no es válido
    """
    from .exceptions import AudioConfigurationError
    if bit_depth not in [16, 24, 32]:
        raise AudioConfigurationError(f"{name} must be 16, 24, or 32, got {bit_depth}")


def validate_positive_integer(value: int, name: str) -> None:
    """
    Valida que un valor sea un entero positivo.
    
    Args:
        value: Valor a validar
        name: Nombre del parámetro para mensajes de error
    
    Raises:
        ValueError: Si el valor no es un entero positivo
    """
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def validate_range(
    value: float,
    min_value: float,
    max_value: float,
    name: str,
    inclusive: bool = True
) -> None:
    """
    Valida que un valor esté en un rango específico.
    
    Args:
        value: Valor a validar
        min_value: Valor mínimo
        max_value: Valor máximo
        name: Nombre del parámetro para mensajes de error
        inclusive: Si True, incluye los límites (>= y <=), si False excluye (> y <)
    
    Raises:
        ValueError: Si el valor está fuera del rango
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")
    
    if inclusive:
        if not min_value <= value <= max_value:
            raise ValueError(
                f"{name} must be between {min_value} and {max_value} (inclusive), got {value}"
            )
    else:
        if not min_value < value < max_value:
            raise ValueError(
                f"{name} must be between {min_value} and {max_value} (exclusive), got {value}"
            )


def validate_non_negative(value: float, name: str) -> None:
    """
    Valida que un valor sea no negativo.
    
    Args:
        value: Valor a validar
        name: Nombre del parámetro para mensajes de error
    
    Raises:
        ValueError: Si el valor es negativo
    """
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_choice(value: str, choices: List[str], name: str) -> None:
    """
    Valida que un valor esté en una lista de opciones válidas.
    
    Args:
        value: Valor a validar
        choices: Lista de opciones válidas
        name: Nombre del parámetro para mensajes de error
    
    Raises:
        ValueError: Si el valor no está en las opciones válidas
    """
    if value not in choices:
        raise ValueError(
            f"{name} must be one of {choices}, got {value}"
        )
