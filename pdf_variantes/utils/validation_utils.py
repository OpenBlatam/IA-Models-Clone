"""
PDF Variantes Validation Utilities
Utilidades de validación para el sistema PDF Variantes
"""

import re
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ValidationResult:
    """Resultado de validación"""
    
    def __init__(self, is_valid: bool = True, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Agregar error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Agregar advertencia"""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }

class InputValidator:
    """Validador de entrada"""
    
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.username_pattern = re.compile(r'^[a-zA-Z0-9_]{3,20}$')
        self.password_pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
        self.url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
    
    def validate_email(self, email: str) -> ValidationResult:
        """Validar email"""
        result = ValidationResult()
        
        if not email:
            result.add_error("Email is required")
            return result
        
        if not self.email_pattern.match(email):
            result.add_error("Invalid email format")
        
        if len(email) > 254:
            result.add_error("Email is too long")
        
        return result
    
    def validate_username(self, username: str) -> ValidationResult:
        """Validar nombre de usuario"""
        result = ValidationResult()
        
        if not username:
            result.add_error("Username is required")
            return result
        
        if not self.username_pattern.match(username):
            result.add_error("Username must be 3-20 characters long and contain only letters, numbers, and underscores")
        
        return result
    
    def validate_password(self, password: str) -> ValidationResult:
        """Validar contraseña"""
        result = ValidationResult()
        
        if not password:
            result.add_error("Password is required")
            return result
        
        if len(password) < 8:
            result.add_error("Password must be at least 8 characters long")
        
        if not re.search(r'[a-z]', password):
            result.add_error("Password must contain at least one lowercase letter")
        
        if not re.search(r'[A-Z]', password):
            result.add_error("Password must contain at least one uppercase letter")
        
        if not re.search(r'\d', password):
            result.add_error("Password must contain at least one digit")
        
        if not re.search(r'[@$!%*?&]', password):
            result.add_error("Password must contain at least one special character (@$!%*?&)")
        
        return result
    
    def validate_url(self, url: str) -> ValidationResult:
        """Validar URL"""
        result = ValidationResult()
        
        if not url:
            result.add_error("URL is required")
            return result
        
        if not self.url_pattern.match(url):
            result.add_error("Invalid URL format")
        
        return result
    
    def validate_file_upload(self, filename: str, content_type: str, file_size: int) -> ValidationResult:
        """Validar subida de archivo"""
        result = ValidationResult()
        
        if not filename:
            result.add_error("Filename is required")
            return result
        
        # Validar extensión
        allowed_extensions = ['.pdf', '.txt', '.docx', '.doc']
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if f'.{file_extension}' not in allowed_extensions:
            result.add_error(f"File extension .{file_extension} not allowed")
        
        # Validar tipo MIME
        allowed_mime_types = [
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword'
        ]
        
        if content_type not in allowed_mime_types:
            result.add_error(f"MIME type {content_type} not allowed")
        
        # Validar tamaño
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            result.add_error(f"File size {file_size} exceeds maximum {max_size}")
        
        return result

class DataValidator:
    """Validador de datos"""
    
    def __init__(self):
        self.input_validator = InputValidator()
    
    def validate_document_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validar datos de documento"""
        result = ValidationResult()
        
        # Validar campos requeridos
        required_fields = ['title', 'filename', 'file_size']
        for field in required_fields:
            if field not in data or not data[field]:
                result.add_error(f"Field {field} is required")
        
        # Validar título
        if 'title' in data:
            title = data['title']
            if len(title) < 1:
                result.add_error("Title must be at least 1 character long")
            if len(title) > 255:
                result.add_error("Title must be less than 255 characters")
        
        # Validar nombre de archivo
        if 'filename' in data:
            filename = data['filename']
            if len(filename) < 1:
                result.add_error("Filename must be at least 1 character long")
            if len(filename) > 255:
                result.add_error("Filename must be less than 255 characters")
        
        # Validar tamaño de archivo
        if 'file_size' in data:
            file_size = data['file_size']
            if not isinstance(file_size, int) or file_size < 0:
                result.add_error("File size must be a positive integer")
        
        return result
    
    def validate_variant_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validar datos de variante"""
        result = ValidationResult()
        
        # Validar campos requeridos
        required_fields = ['document_id', 'content']
        for field in required_fields:
            if field not in data or not data[field]:
                result.add_error(f"Field {field} is required")
        
        # Validar contenido
        if 'content' in data:
            content = data['content']
            if len(content) < 1:
                result.add_error("Content must be at least 1 character long")
        
        # Validar puntuaciones
        score_fields = ['similarity_score', 'creativity_score', 'quality_score']
        for field in score_fields:
            if field in data:
                score = data[field]
                if not isinstance(score, (int, float)) or score < 0 or score > 1:
                    result.add_error(f"{field} must be a number between 0 and 1")
        
        return result
    
    def validate_user_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validar datos de usuario"""
        result = ValidationResult()
        
        # Validar campos requeridos
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data or not data[field]:
                result.add_error(f"Field {field} is required")
        
        # Validar username
        if 'username' in data:
            username_result = self.input_validator.validate_username(data['username'])
            if not username_result.is_valid:
                result.errors.extend(username_result.errors)
        
        # Validar email
        if 'email' in data:
            email_result = self.input_validator.validate_email(data['email'])
            if not email_result.is_valid:
                result.errors.extend(email_result.errors)
        
        # Validar contraseña
        if 'password' in data:
            password_result = self.input_validator.validate_password(data['password'])
            if not password_result.is_valid:
                result.errors.extend(password_result.errors)
        
        return result
    
    def validate_collaboration_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validar datos de colaboración"""
        result = ValidationResult()
        
        # Validar campos requeridos
        required_fields = ['document_id', 'user_id', 'role']
        for field in required_fields:
            if field not in data or not data[field]:
                result.add_error(f"Field {field} is required")
        
        # Validar rol
        if 'role' in data:
            valid_roles = ['viewer', 'editor', 'admin']
            if data['role'] not in valid_roles:
                result.add_error(f"Role must be one of: {', '.join(valid_roles)}")
        
        # Validar permisos
        if 'permissions' in data:
            permissions = data['permissions']
            if not isinstance(permissions, list):
                result.add_error("Permissions must be a list")
            else:
                valid_permissions = ['view', 'edit', 'delete', 'share']
                for permission in permissions:
                    if permission not in valid_permissions:
                        result.add_error(f"Invalid permission: {permission}")
        
        return result

class SecurityValidator:
    """Validador de seguridad"""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'data:.*?base64',  # Data URI
            r'vbscript:',  # VBScript
            r'onload\s*=',  # Event handlers
            r'onerror\s*=',  # Event handlers
            r'<iframe.*?>',  # Iframe injection
            r'<object.*?>',  # Object injection
            r'<embed.*?>',  # Embed injection
        ]
    
    def validate_input_security(self, input_data: str) -> ValidationResult:
        """Validar seguridad de entrada"""
        result = ValidationResult()
        
        if not input_data:
            return result
        
        # Verificar patrones sospechosos
        for pattern in self.suspicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                result.add_error(f"Suspicious pattern detected: {pattern}")
        
        # Verificar caracteres peligrosos
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
        for char in dangerous_chars:
            if char in input_data:
                result.add_warning(f"Potentially dangerous character detected: {char}")
        
        return result
    
    def validate_file_security(self, file_content: bytes) -> ValidationResult:
        """Validar seguridad de archivo"""
        result = ValidationResult()
        
        if not file_content:
            result.add_error("File content is required")
            return result
        
        # Verificar magic bytes de PDF
        if not file_content.startswith(b'%PDF-'):
            result.add_error("File does not appear to be a valid PDF")
        
        # Verificar tamaño mínimo
        if len(file_content) < 100:
            result.add_error("File is too small to be a valid PDF")
        
        # Verificar contenido sospechoso
        content_str = file_content.decode('utf-8', errors='ignore')
        security_result = self.validate_input_security(content_str)
        if not security_result.is_valid:
            result.errors.extend(security_result.errors)
        result.warnings.extend(security_result.warnings)
        
        return result
    
    def validate_api_key(self, api_key: str) -> ValidationResult:
        """Validar clave API"""
        result = ValidationResult()
        
        if not api_key:
            result.add_error("API key is required")
            return result
        
        # Verificar longitud mínima
        if len(api_key) < 20:
            result.add_error("API key is too short")
        
        # Verificar caracteres válidos
        if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
            result.add_error("API key contains invalid characters")
        
        return result

class BusinessLogicValidator:
    """Validador de lógica de negocio"""
    
    def __init__(self):
        self.data_validator = DataValidator()
        self.security_validator = SecurityValidator()
    
    def validate_document_creation(self, user_id: str, document_data: Dict[str, Any]) -> ValidationResult:
        """Validar creación de documento"""
        result = ValidationResult()
        
        # Validar datos básicos
        data_result = self.data_validator.validate_document_data(document_data)
        if not data_result.is_valid:
            result.errors.extend(data_result.errors)
        
        # Validar usuario
        if not user_id:
            result.add_error("User ID is required")
        
        # Validar límites de usuario
        # Aquí se podrían agregar validaciones como:
        # - Límite de documentos por usuario
        # - Límite de tamaño total por usuario
        # - Verificación de permisos
        
        return result
    
    def validate_variant_generation(self, document_id: str, variant_count: int) -> ValidationResult:
        """Validar generación de variantes"""
        result = ValidationResult()
        
        # Validar documento
        if not document_id:
            result.add_error("Document ID is required")
        
        # Validar cantidad de variantes
        if not isinstance(variant_count, int) or variant_count < 1:
            result.add_error("Variant count must be a positive integer")
        
        if variant_count > 1000:
            result.add_error("Variant count exceeds maximum limit of 1000")
        
        return result
    
    def validate_collaboration_invite(self, inviter_id: str, invitee_email: str, document_id: str) -> ValidationResult:
        """Validar invitación de colaboración"""
        result = ValidationResult()
        
        # Validar inviter
        if not inviter_id:
            result.add_error("Inviter ID is required")
        
        # Validar invitee
        email_result = self.data_validator.input_validator.validate_email(invitee_email)
        if not email_result.is_valid:
            result.errors.extend(email_result.errors)
        
        # Validar documento
        if not document_id:
            result.add_error("Document ID is required")
        
        # Validar que no se invite a sí mismo
        # Esta validación requeriría acceso a la base de datos
        
        return result

# Factory functions
def create_input_validator() -> InputValidator:
    """Crear validador de entrada"""
    return InputValidator()

def create_data_validator() -> DataValidator:
    """Crear validador de datos"""
    return DataValidator()

def create_security_validator() -> SecurityValidator:
    """Crear validador de seguridad"""
    return SecurityValidator()

def create_business_logic_validator() -> BusinessLogicValidator:
    """Crear validador de lógica de negocio"""
    return BusinessLogicValidator()

# Funciones de validación específicas
def validate_pdf_content(content: bytes) -> bool:
    """Validar contenido PDF"""
    try:
        validator = SecurityValidator()
        result = validator.validate_file_security(content)
        return result.is_valid
    except Exception as e:
        logger.error(f"Error validating PDF content: {e}")
        return False

def validate_email(email: str) -> bool:
    """Validar email"""
    try:
        validator = InputValidator()
        result = validator.validate_email(email)
        return result.is_valid
    except Exception as e:
        logger.error(f"Error validating email: {e}")
        return False

def validate_password(password: str) -> bool:
    """Validar contraseña"""
    try:
        validator = InputValidator()
        result = validator.validate_password(password)
        return result.is_valid
    except Exception as e:
        logger.error(f"Error validating password: {e}")
        return False

def validate_document_id(document_id: str) -> bool:
    """Validar ID de documento"""
    try:
        # Asumiendo que los IDs son UUIDs
        import uuid
        uuid.UUID(document_id)
        return True
    except ValueError:
        logger.warning(f"Invalid document ID format: {document_id}")
        return False