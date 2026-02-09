"""
Validators for Contabilidad Mexicana AI SAM3
===========================================

Validation functions for fiscal data and inputs.
"""

from typing import Dict, Any, Optional, List
import re


def validate_rfc(rfc: str) -> bool:
    """
    Validate Mexican RFC format.
    
    Args:
        rfc: RFC string
        
    Returns:
        True if valid, False otherwise
    """
    # RFC pattern: 4 letters, 6 digits, optional 3 alphanumeric (homoclave)
    pattern = r'^[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{0,3}$'
    return bool(re.match(pattern, rfc.upper()))


def validate_curp(curp: str) -> bool:
    """
    Validate Mexican CURP format.
    
    Args:
        curp: CURP string
        
    Returns:
        True if valid, False otherwise
    """
    # CURP pattern: 18 characters
    pattern = r'^[A-Z]{4}\d{6}[HM][A-Z]{5}[0-9A-Z]\d$'
    return bool(re.match(pattern, curp.upper()))


def validate_fiscal_period(period: str) -> bool:
    """
    Validate fiscal period format.
    
    Args:
        period: Period string (YYYY-MM or YYYY)
        
    Returns:
        True if valid, False otherwise
    """
    # Monthly format: YYYY-MM
    if len(period) == 7 and period[4] == "-":
        year, month = period.split("-")
        try:
            year_int = int(year)
            month_int = int(month)
            if 2000 <= year_int <= 2100 and 1 <= month_int <= 12:
                return True
        except ValueError:
            return False
    
    # Annual format: YYYY
    if len(period) == 4:
        try:
            year_int = int(period)
            if 2000 <= year_int <= 2100:
                return True
        except ValueError:
            return False
    
    return False


def validate_regimen(regimen: str) -> bool:
    """
    Validate fiscal regime.
    
    Args:
        regimen: Regime name
        
    Returns:
        True if valid, False otherwise
    """
    valid_regimes = [
        "RESICO",
        "PFAE",
        "Sueldos y Salarios",
        "Arrendamiento",
        "Intereses",
        "Dividendos",
        "Actividades Profesionales",
    ]
    return regimen in valid_regimes


def validate_tax_type(tipo_impuesto: str) -> bool:
    """
    Validate tax type.
    
    Args:
        tipo_impuesto: Tax type
        
    Returns:
        True if valid, False otherwise
    """
    valid_taxes = ["ISR", "IVA", "IEPS", "ISAN"]
    return tipo_impuesto in valid_taxes


def validate_calculation_data(datos: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate tax calculation data.
    
    Args:
        datos: Calculation data dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["ingresos"]
    
    for field in required_fields:
        if field not in datos:
            return False, f"Campo requerido faltante: {field}"
        
        if not isinstance(datos[field], (int, float)):
            return False, f"Campo {field} debe ser numérico"
        
        if datos[field] < 0:
            return False, f"Campo {field} no puede ser negativo"
    
    # Validate optional fields
    if "gastos" in datos:
        if not isinstance(datos["gastos"], (int, float)):
            return False, "Campo gastos debe ser numérico"
        if datos["gastos"] < 0:
            return False, "Campo gastos no puede ser negativo"
        if datos["gastos"] > datos["ingresos"]:
            return False, "Los gastos no pueden ser mayores que los ingresos"
    
    return True, None


def validate_declaration_data(datos: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate declaration data.
    
    Args:
        datos: Declaration data dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if "rfc" in datos:
        if not validate_rfc(datos["rfc"]):
            return False, "RFC inválido"
    
    if "periodo" in datos:
        if not validate_fiscal_period(datos["periodo"]):
            return False, "Período fiscal inválido"
    
    return True, None
