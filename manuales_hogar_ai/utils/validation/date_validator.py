"""
Date Validator
=============

Validador especializado para fechas.
"""

from typing import Tuple, Optional
from datetime import datetime
from ...core.base.service_base import BaseService


class DateValidator(BaseService):
    """Validador para fechas."""
    
    def __init__(self):
        """Inicializar validador."""
        super().__init__(logger_name=__name__)
    
    def validate_date_range(
        self,
        date_from: Optional[datetime],
        date_to: Optional[datetime]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validar rango de fechas.
        
        Args:
            date_from: Fecha desde
            date_to: Fecha hasta
        
        Returns:
            Tuple de (es_válido, mensaje_error)
        """
        if date_from and date_to:
            if date_from > date_to:
                return False, "La fecha 'desde' no puede ser mayor que la fecha 'hasta'"
            
            if (date_to - date_from).days > 365:
                return False, "El rango de fechas no puede exceder 1 año"
        
        return True, None

