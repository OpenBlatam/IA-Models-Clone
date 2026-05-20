"""
Formatters for Contabilidad Mexicana AI SAM3
============================================

Utility functions for formatting fiscal data and results.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


def format_currency(amount: float, currency: str = "MXN") -> str:
    """
    Format amount as currency.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "MXN":
        return f"${amount:,.2f} MXN"
    return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format (0-100)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"


def format_fiscal_period(period: str) -> str:
    """
    Format fiscal period.
    
    Args:
        period: Period string (e.g., "2024-01" or "2024")
        
    Returns:
        Formatted period string
    """
    if len(period) == 7 and period[4] == "-":
        # Monthly format: 2024-01
        year, month = period.split("-")
        months = {
            "01": "Enero", "02": "Febrero", "03": "Marzo",
            "04": "Abril", "05": "Mayo", "06": "Junio",
            "07": "Julio", "08": "Agosto", "09": "Septiembre",
            "10": "Octubre", "11": "Noviembre", "12": "Diciembre"
        }
        return f"{months.get(month, month)} {year}"
    return period


def format_tax_calculation_result(result: Dict[str, Any]) -> str:
    """
    Format tax calculation result for display.
    
    Args:
        result: Calculation result dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("RESULTADO DE CÁLCULO DE IMPUESTOS")
    lines.append("=" * 50)
    
    if "base_calculo" in result:
        lines.append(f"\nBase de Cálculo: {format_currency(result['base_calculo'])}")
    
    if "tasa" in result:
        lines.append(f"Tasa Aplicable: {format_percentage(result['tasa'])}")
    
    if "impuesto_calculado" in result:
        lines.append(f"\nImpuesto Calculado: {format_currency(result['impuesto_calculado'])}")
    
    if "fecha_pago" in result:
        lines.append(f"\nFecha de Pago: {result['fecha_pago']}")
    
    if "recomendaciones" in result:
        lines.append(f"\nRecomendaciones:")
        for rec in result["recomendaciones"]:
            lines.append(f"  - {rec}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def format_fiscal_advice(advice: Dict[str, Any]) -> str:
    """
    Format fiscal advice for display.
    
    Args:
        advice: Advice dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("ASESORÍA FISCAL")
    lines.append("=" * 50)
    
    if "analisis" in advice:
        lines.append(f"\nAnálisis:")
        lines.append(advice["analisis"])
    
    if "opciones" in advice:
        lines.append(f"\nOpciones Disponibles:")
        for i, opcion in enumerate(advice["opciones"], 1):
            lines.append(f"{i}. {opcion}")
    
    if "recomendacion" in advice:
        lines.append(f"\nRecomendación:")
        lines.append(advice["recomendacion"])
    
    lines.append("=" * 50)
    return "\n".join(lines)


def format_regimen_info(regimen: str) -> Dict[str, Any]:
    """
    Get formatted information about a fiscal regime.
    
    Args:
        regimen: Regime name
        
    Returns:
        Dictionary with regime information
    """
    regimes = {
        "RESICO": {
            "nombre": "Régimen Simplificado de Confianza",
            "tasa_isr": 1.0,
            "tasa_iva": 0.0,
            "descripcion": "Régimen para personas físicas con ingresos menores a 3.5 millones de pesos anuales"
        },
        "PFAE": {
            "nombre": "Personas Físicas con Actividades Empresariales",
            "tasa_isr": "Variable",
            "tasa_iva": 16.0,
            "descripcion": "Régimen para personas físicas que realizan actividades empresariales"
        },
        "Sueldos y Salarios": {
            "nombre": "Sueldos y Salarios",
            "tasa_isr": "Tabla progresiva",
            "tasa_iva": 0.0,
            "descripcion": "Régimen para trabajadores asalariados"
        }
    }
    
    return regimes.get(regimen, {
        "nombre": regimen,
        "tasa_isr": "N/A",
        "tasa_iva": "N/A",
        "descripcion": "Información no disponible"
    })
