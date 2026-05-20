"""
User Validator
=============

Validador especializado para usuarios.
"""

import re
from typing import Tuple, Optional
from ...core.base.service_base import BaseService


class UserValidator(BaseService):
    """Validador para usuarios."""
    
    def __init__(self):
        """Inicializar validador."""
        super().__init__(logger_name=__name__)
    
    def validate_user_id(self, user_id: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validar ID de usuario.
        
        Args:
            user_id: ID de usuario a validar
        
        Returns:
            Tuple de (es_válida, mensaje_error)
        """
        if user_id is None:
            return True, None
        
        if not isinstance(user_id, str):
            return False, "User ID debe ser una cadena de texto"
        
        if len(user_id) < 1 or len(user_id) > 100:
            return False, "User ID debe tener entre 1 y 100 caracteres"
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return False, "User ID contiene caracteres no permitidos"
        
        return True, None

