"""
Utils Service - Servicio de utilidades
"""
from .validators import Validator
from .formatters import Formatter
from .helpers import Helper


class UtilsService:
    """Servicio centralizado de utilidades"""
    
    def __init__(self):
        self.validator = Validator()
        self.formatter = Formatter()
        self.helper = Helper()
    
    def validate(self, data: any, schema: any) -> bool:
        """Valida datos usando el validador"""
        return self.validator.validate(data, schema)
    
    def format(self, data: any, format_type: str) -> str:
        """Formatea datos"""
        return self.formatter.format(data, format_type)

