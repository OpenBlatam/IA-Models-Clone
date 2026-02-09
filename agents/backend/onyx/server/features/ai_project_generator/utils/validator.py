"""
Validator - Validador de Proyectos Generados
=============================================

Valida que los proyectos generados sean correctos y completos.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
Refactored with improved error handling and file operations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .file_operations import read_json, FileOperationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """
    Resultado de validación.
    Inmutable para mejor seguridad.
    """
    valid: bool
    errors: List[str]
    warnings: List[str]
    checks_passed: List[str]
    error_count: int
    warning_count: int


def _validate_directory_exists(directory: Path, dir_name: str) -> Optional[str]:
    """
    Validar que un directorio exista (función pura).
    
    Args:
        directory: Directorio a validar
        dir_name: Nombre del directorio para el mensaje de error
        
    Returns:
        Mensaje de error o None si es válido
    """
    if not directory.exists():
        return f"Directorio faltante: {dir_name}"
    return None


def _validate_file_exists(file_path: Path, file_name: str, description: str) -> Optional[str]:
    """
    Validar que un archivo exista (función pura).
    
    Args:
        file_path: Archivo a validar
        file_name: Nombre del archivo
        description: Descripción del archivo
        
    Returns:
        Mensaje de error o None si es válido
    """
    if not file_path.exists():
        return f"Archivo faltante: {file_name} ({description})"
    return None


def _validate_json_file(
    file_path: Path,
    required_fields: List[str],
    error_prefix: str
) -> List[str]:
    """
    Validar archivo JSON (función pura).
    
    Args:
        file_path: Ruta del archivo JSON
        required_fields: Campos requeridos
        error_prefix: Prefijo para mensajes de error
        
    Returns:
        Lista de errores encontrados
    """
    errors: List[str] = []
    
    if not file_path.exists():
        errors.append(f"{error_prefix} no encontrado")
        return errors
    
    try:
        data = read_json(file_path, default=None)
        if data is None:
            errors.append(f"Error leyendo {error_prefix}: archivo no válido")
            return errors
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Campo faltante en {error_prefix}: {field}")
    except FileOperationError as e:
        errors.append(f"Error de formato JSON en {error_prefix}: {e}")
    except Exception as e:
        errors.append(f"Error leyendo {error_prefix}: {e}")
    
    return errors


def _validate_python_syntax(file_path: Path) -> Optional[str]:
    """
    Validar sintaxis Python (función pura).
    
    Args:
        file_path: Archivo Python a validar
        
    Returns:
        Mensaje de error o None si es válido
    """
    if not file_path.exists():
        return None
    
    try:
        compile(file_path.read_text(encoding="utf-8"), str(file_path), 'exec')
        return None
    except SyntaxError as e:
        return f"Error de sintaxis en {file_path.name}: {e}"
    except Exception as e:
        return f"Error validando {file_path.name}: {e}"


def _get_required_directories() -> List[str]:
    """
    Obtiene lista de directorios requeridos (función pura).
    
    Returns:
        Lista de directorios requeridos
    """
    return [
        "backend",
        "frontend",
        "backend/app",
        "backend/app/api",
        "backend/app/core",
        "frontend/src",
    ]


def _get_required_files() -> List[Tuple[str, str]]:
    """
    Obtiene lista de archivos requeridos (función pura).
    
    Returns:
        Lista de tuplas (ruta, descripción)
    """
    return [
        ("backend/main.py", "Archivo principal del backend"),
        ("backend/requirements.txt", "Dependencias del backend"),
        ("frontend/package.json", "Configuración del frontend"),
        ("frontend/src/main.tsx", "Punto de entrada del frontend"),
        ("README.md", "Documentación principal"),
    ]


def _get_optional_files() -> List[Tuple[str, str]]:
    """
    Obtiene lista de archivos opcionales (función pura).
    
    Returns:
        Lista de tuplas (ruta, descripción)
    """
    return [
        ("backend/Dockerfile", "Dockerfile del backend"),
        ("frontend/Dockerfile", "Dockerfile del frontend"),
        ("docker-compose.yml", "Docker Compose"),
        (".gitignore", "Git ignore"),
    ]


class ProjectValidator:
    """
    Validador de proyectos generados.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializa el validador."""
        pass
    
    async def validate_project(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Valida un proyecto generado.

        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto

        Returns:
            Resultado de la validación
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        errors: List[str] = []
        warnings: List[str] = []
        
        # Validar estructura de directorios
        structure_errors = self._validate_structure(project_dir)
        errors.extend(structure_errors)
        
        # Validar archivos esenciales
        files_errors, files_warnings = self._validate_files(project_dir)
        errors.extend(files_errors)
        warnings.extend(files_warnings)
        
        # Validar configuración
        config_errors = self._validate_config(project_dir, project_info)
        errors.extend(config_errors)
        
        # Validar código básico
        code_errors = await self._validate_code(project_dir)
        errors.extend(code_errors)
        
        # Crear resultado
        is_valid = len(errors) == 0
        checks_passed = [
            "Estructura de directorios",
            "Archivos esenciales",
            "Configuración",
            "Código básico",
        ] if is_valid else []
        
        result = ValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            checks_passed=checks_passed,
            error_count=len(errors),
            warning_count=len(warnings),
        )
        
        return {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "checks_passed": result.checks_passed,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
        }
    
    def _validate_structure(self, project_dir: Path) -> List[str]:
        """
        Valida la estructura de directorios.
        
        Args:
            project_dir: Directorio del proyecto
            
        Returns:
            Lista de errores encontrados
        """
        errors: List[str] = []
        required_dirs = _get_required_directories()
        
        for dir_path in required_dirs:
            full_path = project_dir / dir_path
            error = _validate_directory_exists(full_path, dir_path)
            if error:
                errors.append(error)
        
        return errors
    
    def _validate_files(self, project_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Valida archivos esenciales.
        
        Args:
            project_dir: Directorio del proyecto
            
        Returns:
            Tupla (errores, advertencias)
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        # Archivos requeridos
        required_files = _get_required_files()
        for file_path, description in required_files:
            full_path = project_dir / file_path
            error = _validate_file_exists(full_path, file_path, description)
            if error:
                errors.append(error)
        
        # Archivos opcionales
        optional_files = _get_optional_files()
        for file_path, description in optional_files:
            full_path = project_dir / file_path
            warning = _validate_file_exists(full_path, file_path, description)
            if warning:
                warnings.append(f"Archivo opcional faltante: {file_path} ({description})")
        
        return errors, warnings
    
    def _validate_config(self, project_dir: Path, project_info: Dict[str, Any]) -> List[str]:
        """
        Valida configuración.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            
        Returns:
            Lista de errores encontrados
        """
        info_path = project_dir / "project_info.json"
        required_fields = ["name", "description", "version"]
        
        return _validate_json_file(info_path, required_fields, "project_info.json")
    
    async def _validate_code(self, project_dir: Path) -> List[str]:
        """
        Valida código básico.
        
        Args:
            project_dir: Directorio del proyecto
            
        Returns:
            Lista de errores encontrados
        """
        errors: List[str] = []
        
        # Validar sintaxis Python básica
        backend_main = project_dir / "backend" / "main.py"
        syntax_error = _validate_python_syntax(backend_main)
        if syntax_error:
            errors.append(syntax_error)
        
        # Validar package.json
        frontend_package = project_dir / "frontend" / "package.json"
        package_errors = _validate_json_file(
            frontend_package,
            ["name", "version"],
            "package.json"
        )
        errors.extend(package_errors)
        
        return errors
