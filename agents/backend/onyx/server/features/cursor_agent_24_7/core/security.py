"""
Security Enhancements
=====================

Mejoras de seguridad para el manejo de secretos y datos sensibles,
siguiendo las mejores prácticas de Devin.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretDetector:
    """
    Detector de secretos y datos sensibles en código.
    
    Detecta y previene la exposición de:
    - API keys
    - Passwords
    - Tokens
    - Credentials
    """
    
    # Patrones comunes de secretos
    SECRET_PATTERNS = [
        (r'api[_-]?key["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'api_key'),
        (r'password["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'password'),
        (r'secret["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'secret'),
        (r'token["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'token'),
        (r'credential["\']?\s*[:=]\s*["\']([^"\']+)["\']', 'credential'),
        (r'aws[_-]?access[_-]?key', 'aws_key'),
        (r'aws[_-]?secret[_-]?key', 'aws_secret'),
        (r'sk-[a-zA-Z0-9]{32,}', 'openai_key'),
        (r'ghp_[a-zA-Z0-9]{36}', 'github_token'),
    ]
    
    # Variables de entorno comunes que contienen secretos
    SECRET_ENV_VARS = {
        'API_KEY', 'SECRET_KEY', 'PASSWORD', 'TOKEN',
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
        'DATABASE_URL', 'REDIS_URL', 'MONGODB_URI'
    }
    
    def __init__(self) -> None:
        """Inicializar detector de secretos"""
        self.detected_secrets: List[Dict[str, Any]] = []
        logger.info("🔒 Secret detector initialized")
    
    def scan_code(self, code: str, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Escanear código en busca de secretos.
        
        Args:
            code: Código a escanear.
            file_path: Ruta del archivo (opcional).
        
        Returns:
            Lista de secretos detectados.
        """
        detected: List[Dict[str, Any]] = []
        
        for pattern, secret_type in self.SECRET_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                detected.append({
                    "type": secret_type,
                    "pattern": pattern,
                    "line": code[:match.start()].count('\n') + 1,
                    "file_path": file_path,
                    "severity": self._get_severity(secret_type)
                })
        
        if detected:
            self.detected_secrets.extend(detected)
            logger.warning(
                f"⚠️ Detected {len(detected)} potential secrets in {file_path or 'code'}"
            )
        
        return detected
    
    def scan_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Escanear archivo en busca de secretos.
        
        Args:
            file_path: Ruta del archivo.
        
        Returns:
            Lista de secretos detectados.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return []
            
            content = path.read_text(encoding='utf-8')
            return self.scan_code(content, file_path)
            
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}", exc_info=True)
            return []
    
    def _get_severity(self, secret_type: str) -> str:
        """Obtener severidad del tipo de secreto"""
        high_severity = ['password', 'aws_secret', 'openai_key', 'github_token']
        if secret_type in high_severity:
            return 'high'
        return 'medium'
    
    def sanitize_output(self, output: str) -> str:
        """
        Sanitizar salida para remover secretos.
        
        Args:
            output: Salida a sanitizar.
        
        Returns:
            Salida sanitizada.
        """
        sanitized = output
        
        for pattern, secret_type in self.SECRET_PATTERNS:
            sanitized = re.sub(
                pattern,
                lambda m: f'<{secret_type}_REDACTED>',
                sanitized,
                flags=re.IGNORECASE
            )
        
        return sanitized
    
    def get_detected_secrets(self) -> List[Dict[str, Any]]:
        """Obtener lista de secretos detectados"""
        return self.detected_secrets.copy()
    
    def clear_detected_secrets(self) -> None:
        """Limpiar lista de secretos detectados"""
        self.detected_secrets.clear()


class SecurityManager:
    """
    Gestor de seguridad del agente.
    
    Gestiona:
    - Detección de secretos
    - Sanitización de salidas
    - Validación de comandos peligrosos
    """
    
    DANGEROUS_COMMANDS = [
        'rm -rf', 'del /f', 'format', 'mkfs',
        'dd if=', 'shutdown', 'reboot',
        'sudo rm', 'sudo del'
    ]
    
    def __init__(self) -> None:
        """Inicializar gestor de seguridad"""
        self.secret_detector = SecretDetector()
        self.blocked_commands: Set[str] = set()
        logger.info("🛡️ Security manager initialized")
    
    def validate_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Validar comando por seguridad.
        
        Args:
            command: Comando a validar.
        
        Returns:
            Tupla (es_válido, mensaje_error).
        """
        command_lower = command.lower()
        
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous.lower() in command_lower:
                return False, f"Dangerous command detected: {dangerous}"
        
        if command in self.blocked_commands:
            return False, "Command is blocked"
        
        return True, None
    
    def scan_command_for_secrets(self, command: str) -> List[Dict[str, Any]]:
        """
        Escanear comando en busca de secretos.
        
        Args:
            command: Comando a escanear.
        
        Returns:
            Lista de secretos detectados.
        """
        return self.secret_detector.scan_code(command)
    
    def sanitize_output(self, output: str) -> str:
        """
        Sanitizar salida para remover secretos.
        
        Args:
            output: Salida a sanitizar.
        
        Returns:
            Salida sanitizada.
        """
        return self.secret_detector.sanitize_output(output)
    
    def block_command(self, command: str) -> None:
        """
        Bloquear un comando.
        
        Args:
            command: Comando a bloquear.
        """
        self.blocked_commands.add(command)
        logger.info(f"🚫 Command blocked: {command}")
    
    def unblock_command(self, command: str) -> None:
        """
        Desbloquear un comando.
        
        Args:
            command: Comando a desbloquear.
        """
        self.blocked_commands.discard(command)
        logger.info(f"✅ Command unblocked: {command}")

