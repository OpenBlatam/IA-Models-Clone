"""Validators - Validadores de datos mejorados"""

import re
from typing import Any, Optional, Pattern
from urllib.parse import urlparse
import ipaddress


class Validator:
    """Validador de datos mejorado"""
    
    @staticmethod
    def validate(data: Any, schema: Any) -> bool:
        """
        Valida datos contra un esquema.
        
        Args:
            data: Datos a validar
            schema: Esquema de validación
        
        Returns:
            True si es válido
        """
        if schema is None:
            return True
        
        if isinstance(schema, type):
            return isinstance(data, schema)
        
        if isinstance(schema, dict):
            if not isinstance(data, dict):
                return False
            for key, value_schema in schema.items():
                if key not in data:
                    return False
                if not Validator.validate(data[key], value_schema):
                    return False
            return True
        
        if isinstance(schema, list):
            if not isinstance(data, list):
                return False
            if len(schema) > 0:
                item_schema = schema[0]
                return all(Validator.validate(item, item_schema) for item in data)
            return True
        
        return data == schema
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Valida formato de email.
        
        Args:
            email: Email a validar
        
        Returns:
            True si es válido
        """
        if not email or not isinstance(email, str):
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str, schemes: Optional[list] = None) -> bool:
        """
        Valida formato de URL.
        
        Args:
            url: URL a validar
            schemes: Lista de esquemas permitidos (default: http, https)
        
        Returns:
            True si es válido
        """
        if not url or not isinstance(url, str):
            return False
        
        if schemes is None:
            schemes = ['http', 'https']
        
        try:
            result = urlparse(url)
            return all([result.scheme in schemes, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def validate_ip(ip: str, version: Optional[int] = None) -> bool:
        """
        Valida formato de IP.
        
        Args:
            ip: IP a validar
            version: Versión de IP (4 o 6, None para ambas)
        
        Returns:
            True si es válido
        """
        if not ip or not isinstance(ip, str):
            return False
        
        try:
            addr = ipaddress.ip_address(ip)
            if version is None:
                return True
            elif version == 4:
                return isinstance(addr, ipaddress.IPv4Address)
            elif version == 6:
                return isinstance(addr, ipaddress.IPv6Address)
            return False
        except ValueError:
            return False
    
    @staticmethod
    def validate_phone(phone: str, country_code: Optional[str] = None) -> bool:
        """
        Valida formato de teléfono.
        
        Args:
            phone: Teléfono a validar
            country_code: Código de país opcional
        
        Returns:
            True si es válido
        """
        if not phone or not isinstance(phone, str):
            return False
        
        cleaned = re.sub(r'[\s\-\(\)]', '', phone)
        
        if country_code:
            if not cleaned.startswith(country_code):
                return False
            cleaned = cleaned[len(country_code):]
        
        return bool(re.match(r'^\d{7,15}$', cleaned))
    
    @staticmethod
    def validate_regex(text: str, pattern: Pattern[str]) -> bool:
        """
        Valida texto contra regex.
        
        Args:
            text: Texto a validar
            pattern: Patrón regex compilado
        
        Returns:
            True si coincide
        """
        if not text or not isinstance(text, str):
            return False
        
        return bool(pattern.match(text))
    
    @staticmethod
    def validate_length(
        text: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> bool:
        """
        Valida longitud de texto.
        
        Args:
            text: Texto a validar
            min_length: Longitud mínima
            max_length: Longitud máxima
        
        Returns:
            True si es válido
        """
        if not isinstance(text, str):
            return False
        
        length = len(text)
        
        if min_length is not None and length < min_length:
            return False
        
        if max_length is not None and length > max_length:
            return False
        
        return True
    
    @staticmethod
    def validate_range(
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """
        Valida rango de valor numérico.
        
        Args:
            value: Valor a validar
            min_value: Valor mínimo
            max_value: Valor máximo
        
        Returns:
            True si está en rango
        """
        try:
            num = float(value)
            
            if min_value is not None and num < min_value:
                return False
            
            if max_value is not None and num > max_value:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_required(data: dict, required_fields: list) -> bool:
        """
        Valida que campos requeridos estén presentes.
        
        Args:
            data: Diccionario a validar
            required_fields: Lista de campos requeridos
        
        Returns:
            True si todos los campos están presentes
        """
        if not isinstance(data, dict):
            return False
        
        return all(field in data for field in required_fields)

