"""
Calculator Helper for Contador AI
==================================

Centralizes logic for using specialized calculators (calculadora_impuestos, comparador_regimenes).
Eliminates repetitive try/except blocks and calculator integration patterns.

Single Responsibility: Handle all specialized calculator operations.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class CalculatorHelper:
    """
    Helper class for specialized calculator operations.
    
    Responsibilities:
    - Try to use specialized calculators when available
    - Handle calculator errors gracefully
    - Integrate calculator results into prompts
    """
    
    @staticmethod
    def try_calcular_impuesto(
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Try to calculate tax using specialized calculator.
        
        Args:
            regimen: Fiscal regime
            tipo_impuesto: Tax type
            datos: Input data
            
        Returns:
            Calculator result if available, None otherwise
        """
        if regimen != "RESICO" or tipo_impuesto != "ISR":
            return None
        
        try:
            from ..services.calculadora_impuestos import CalculadoraImpuestos
            calculadora = CalculadoraImpuestos()
            resultado = calculadora.calcular_impuesto_mensual(
                regimen, tipo_impuesto, datos
            )
            return resultado if "error" not in resultado else None
        except (ImportError, Exception) as e:
            logger.debug(f"Calculadora no disponible: {e}")
            return None
    
    @staticmethod
    def try_comparar_regimenes(
        regimenes: List[str],
        datos: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Try to compare regimes using specialized comparator.
        
        Args:
            regimenes: List of fiscal regimes
            datos: Taxpayer data
            
        Returns:
            Comparator result if available, None otherwise
        """
        try:
            from ..services.comparador_regimenes import ComparadorRegimenes
            comparador = ComparadorRegimenes()
            resultado = comparador.comparar_carga_fiscal(regimenes, datos)
            return resultado
        except (ImportError, Exception) as e:
            logger.debug(f"Comparador no disponible: {e}")
            return None
    
    @staticmethod
    def enhance_prompt_with_calculator_result(
        prompt: str,
        calculator_result: Optional[Dict[str, Any]],
        result_type: str = "cálculo"
    ) -> str:
        """
        Enhance prompt with calculator result if available.
        
        Args:
            prompt: Original prompt
            calculator_result: Result from specialized calculator
            result_type: Type of result ("cálculo" or "comparación")
            
        Returns:
            Enhanced prompt with calculator result
        """
        if not calculator_result:
            return prompt
        
        if result_type == "cálculo":
            return f"{prompt}\n\nNota: El cálculo directo muestra: {calculator_result}"
        elif result_type == "comparación":
            return f"{prompt}\n\nNota: Los cálculos directos muestran: {calculator_result}"
        
        return prompt
    
    @staticmethod
    def add_calculator_result_to_response(
        result: Dict[str, Any],
        calculator_result: Optional[Dict[str, Any]],
        field_name: str = "calculos_directos"
    ) -> Dict[str, Any]:
        """
        Add calculator result to response dictionary.
        
        Args:
            result: Response dictionary
            calculator_result: Result from specialized calculator
            field_name: Name of field to add calculator result to
            
        Returns:
            Response dictionary with calculator result added
        """
        if calculator_result:
            result[field_name] = calculator_result
        return result

