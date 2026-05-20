"""
Security Generator - Generador de utilidades de seguridad
=========================================================

Genera utilidades para seguridad:
- Input sanitization
- Model security checks
- Data privacy utilities
- Secure model loading
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SecurityGenerator:
    """Generador de utilidades de seguridad"""
    
    def __init__(self):
        """Inicializa el generador de seguridad"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de seguridad.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        security_dir = utils_dir / "security"
        security_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_input_sanitizer(security_dir, keywords, project_info)
        self._generate_model_security(security_dir, keywords, project_info)
        self._generate_security_init(security_dir, keywords)
    
    def _generate_security_init(
        self,
        security_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de seguridad"""
        
        init_content = '''"""
Security Utilities Module
==========================

Utilidades para seguridad y protección de modelos y datos.
"""

from .input_sanitizer import (
    InputSanitizer,
    sanitize_text,
    sanitize_filename,
    check_malicious_content,
)
from .model_security import (
    ModelSecurityChecker,
    verify_model_integrity,
    check_model_safety,
    secure_model_load,
)

__all__ = [
    "InputSanitizer",
    "sanitize_text",
    "sanitize_filename",
    "check_malicious_content",
    "ModelSecurityChecker",
    "verify_model_integrity",
    "check_model_safety",
    "secure_model_load",
]
'''
        
        (security_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_input_sanitizer(
        self,
        security_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera sanitizador de inputs"""
        
        sanitizer_content = '''"""
Input Sanitizer - Sanitizador de inputs
========================================

Utilidades para sanitizar y validar inputs de manera segura.
"""

import re
import os
from typing import Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """
    Sanitizador de inputs para prevenir ataques.
    
    Previene injection attacks, path traversal, y otros vectores de ataque.
    """
    
    def __init__(self):
        """Inicializa el sanitizador"""
        # Patrones peligrosos
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'on\w+\s*=',  # Event handlers
            r'<iframe[^>]*>',  # iframe injection
            r'<object[^>]*>',  # Object injection
            r'<embed[^>]*>',  # Embed injection
            r'\.\./',  # Path traversal
            r'\.\.\\\\',  # Path traversal (Windows)
            r'[<>"\']',  # HTML/XML injection
        ]
        
        # Caracteres peligrosos para filenames
        self.dangerous_filename_chars = ['/', '\\\\', '..', '<', '>', ':', '"', '|', '?', '*']
    
    def sanitize_text(
        self,
        text: str,
        max_length: int = 10000,
        allow_html: bool = False,
    ) -> str:
        """
        Sanitiza texto removiendo contenido peligroso.
        
        Args:
            text: Texto a sanitizar
            max_length: Longitud máxima permitida
            allow_html: Si permitir HTML básico
        
        Returns:
            Texto sanitizado
        """
        if not isinstance(text, str):
            raise ValueError(f"Texto debe ser string, recibido: {type(text)}")
        
        # Limitar longitud
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Texto truncado a {max_length} caracteres")
        
        # Remover patrones peligrosos
        for pattern in self.dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Si no se permite HTML, remover todos los tags
        if not allow_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Remover caracteres de control
        text = ''.join(char for char in text if ord(char) >= 32 or char in ['\\n', '\\t'])
        
        return text.strip()
    
    def sanitize_filename(
        self,
        filename: str,
        max_length: int = 255,
    ) -> str:
        """
        Sanitiza nombre de archivo.
        
        Args:
            filename: Nombre de archivo a sanitizar
            max_length: Longitud máxima
        
        Returns:
            Nombre de archivo sanitizado
        """
        if not isinstance(filename, str):
            raise ValueError(f"Filename debe ser string, recibido: {type(filename)}")
        
        # Remover caracteres peligrosos
        for char in self.dangerous_filename_chars:
            filename = filename.replace(char, '_')
        
        # Remover espacios al inicio/final
        filename = filename.strip()
        
        # Limitar longitud
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            max_name_length = max_length - len(ext)
            filename = name[:max_name_length] + ext
        
        # Asegurar que no esté vacío
        if not filename:
            filename = "unnamed_file"
        
        return filename
    
    def check_malicious_content(
        self,
        content: str,
    ) -> dict:
        """
        Verifica si el contenido contiene patrones maliciosos.
        
        Args:
            content: Contenido a verificar
        
        Returns:
            Diccionario con resultados de verificación
        """
        results = {
            "is_safe": True,
            "threats_found": [],
        }
        
        content_lower = content.lower()
        
        # Verificar patrones peligrosos
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                results["is_safe"] = False
                results["threats_found"].append(f"Patrón peligroso detectado: {pattern}")
        
        # Verificar comandos del sistema
        system_commands = ['rm ', 'del ', 'format ', 'shutdown', 'reboot', 'sudo']
        for cmd in system_commands:
            if cmd in content_lower:
                results["is_safe"] = False
                results["threats_found"].append(f"Comando del sistema detectado: {cmd}")
        
        return results


def sanitize_text(
    text: str,
    **kwargs,
) -> str:
    """
    Función helper para sanitizar texto.
    
    Args:
        text: Texto a sanitizar
        **kwargs: Argumentos adicionales
    
    Returns:
        Texto sanitizado
    """
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_text(text, **kwargs)


def sanitize_filename(
    filename: str,
    **kwargs,
) -> str:
    """
    Función helper para sanitizar filename.
    
    Args:
        filename: Nombre de archivo a sanitizar
        **kwargs: Argumentos adicionales
    
    Returns:
        Nombre de archivo sanitizado
    """
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_filename(filename, **kwargs)


def check_malicious_content(
    content: str,
) -> dict:
    """
    Función helper para verificar contenido malicioso.
    
    Args:
        content: Contenido a verificar
    
    Returns:
        Resultados de verificación
    """
    sanitizer = InputSanitizer()
    return sanitizer.check_malicious_content(content)
'''
        
        (security_dir / "input_sanitizer.py").write_text(sanitizer_content, encoding="utf-8")
    
    def _generate_model_security(
        self,
        security_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de seguridad de modelos"""
        
        model_security_content = '''"""
Model Security - Seguridad de modelos
======================================

Utilidades para verificar seguridad e integridad de modelos.
"""

import torch
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelSecurityChecker:
    """
    Verificador de seguridad de modelos.
    
    Verifica integridad y seguridad de modelos antes de cargarlos.
    """
    
    def __init__(self):
        """Inicializa el verificador"""
        pass
    
    def verify_model_integrity(
        self,
        model_path: Path,
        expected_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verifica integridad de un modelo.
        
        Args:
            model_path: Ruta al modelo
            expected_hash: Hash esperado (opcional)
        
        Returns:
            Diccionario con resultados de verificación
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            results["valid"] = False
            results["errors"].append(f"Modelo no encontrado: {model_path}")
            return results
        
        # Calcular hash
        model_hash = self._calculate_file_hash(model_path)
        results["model_hash"] = model_hash
        
        # Verificar hash esperado
        if expected_hash:
            if model_hash != expected_hash:
                results["valid"] = False
                results["errors"].append(
                    f"Hash no coincide: esperado {expected_hash}, recibido {model_hash}"
                )
        
        # Verificar que el archivo puede ser cargado
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            results["loadable"] = True
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Error cargando modelo: {str(e)}")
            results["loadable"] = False
        
        return results
    
    def check_model_safety(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, Any]:
        """
        Verifica seguridad de un modelo cargado.
        
        Args:
            model: Modelo a verificar
        
        Returns:
            Diccionario con resultados de verificación
        """
        results = {
            "safe": True,
            "warnings": [],
        }
        
        # Verificar que todos los parámetros son finitos
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                results["safe"] = False
                results["warnings"].append(f"Parámetro {name} contiene valores no finitos")
        
        # Verificar que no hay valores extremos
        for name, param in model.named_parameters():
            if param.abs().max() > 1e6:
                results["warnings"].append(
                    f"Parámetro {name} tiene valores muy grandes: {param.abs().max()}"
                )
        
        return results
    
    def secure_model_load(
        self,
        model_path: Path,
        model_class: Optional[torch.nn.Module] = None,
        device: str = "cpu",
        expected_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Carga un modelo de manera segura.
        
        Args:
            model_path: Ruta al modelo
            model_class: Clase del modelo (opcional)
            device: Dispositivo donde cargar
            expected_hash: Hash esperado (opcional)
        
        Returns:
            Diccionario con modelo y resultados de verificación
        """
        # Verificar integridad primero
        integrity_check = self.verify_model_integrity(model_path, expected_hash)
        
        if not integrity_check["valid"]:
            raise ValueError(f"Modelo no válido: {integrity_check['errors']}")
        
        # Cargar modelo de manera segura
        try:
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=True,  # Solo cargar weights, no código
            )
            
            result = {
                "checkpoint": checkpoint,
                "integrity_check": integrity_check,
            }
            
            # Si se proporciona clase del modelo, instanciar
            if model_class:
                model = model_class()
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                result["model"] = model
                
                # Verificar seguridad del modelo
                safety_check = self.check_model_safety(model)
                result["safety_check"] = safety_check
            
            logger.info(f"Modelo cargado de manera segura desde {model_path}")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo de manera segura: {str(e)}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calcula hash SHA256 de un archivo.
        
        Args:
            file_path: Ruta al archivo
        
        Returns:
            Hash del archivo
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def verify_model_integrity(
    model_path: Path,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Función helper para verificar integridad.
    
    Args:
        model_path: Ruta al modelo
        expected_hash: Hash esperado
    
    Returns:
        Resultados de verificación
    """
    checker = ModelSecurityChecker()
    return checker.verify_model_integrity(model_path, expected_hash)


def check_model_safety(
    model: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Función helper para verificar seguridad.
    
    Args:
        model: Modelo a verificar
    
    Returns:
        Resultados de verificación
    """
    checker = ModelSecurityChecker()
    return checker.check_model_safety(model)


def secure_model_load(
    model_path: Path,
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para cargar modelo de manera segura.
    
    Args:
        model_path: Ruta al modelo
        **kwargs: Argumentos adicionales
    
    Returns:
        Diccionario con modelo y resultados
    """
    checker = ModelSecurityChecker()
    return checker.secure_model_load(model_path, **kwargs)
'''
        
        (security_dir / "model_security.py").write_text(model_security_content, encoding="utf-8")

