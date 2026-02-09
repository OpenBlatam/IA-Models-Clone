"""
Comparador de Regímenes Fiscales
=================================

Servicio especializado para comparar diferentes regímenes fiscales
y calcular la carga fiscal de cada uno.
"""

import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal

from .calculadora_impuestos import CalculadoraImpuestos

logger = logging.getLogger(__name__)


class ComparadorRegimenes:
    """Compara diferentes regímenes fiscales."""
    
    def __init__(self):
        self.calculadora = CalculadoraImpuestos()
    
    def comparar_carga_fiscal(
        self,
        regimenes: List[str],
        datos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comparar carga fiscal entre diferentes regímenes.
        
        Args:
            regimenes: Lista de regímenes a comparar
            datos: Datos del contribuyente
        
        Returns:
            Diccionario con comparación detallada
        """
        resultados = {}
        
        for regimen in regimenes:
            try:
                if regimen == "RESICO":
                    # Calcular ISR para RESICO
                    ingresos_mensuales = datos.get("ingresos_mensuales", 0)
                    ingresos_anuales = ingresos_mensuales * 12
                    resultado = self.calculadora.calcular_isr_resico(ingresos_anuales)
                    resultado["ingresos_mensuales"] = ingresos_mensuales
                    resultado["impuesto_mensual"] = resultado["impuesto_total"] / 12
                    resultados[regimen] = resultado
                else:
                    # Para otros regímenes, retornar estructura básica
                    resultados[regimen] = {
                        "regimen": regimen,
                        "nota": "Cálculo detallado requiere asesoría de IA"
                    }
            except Exception as e:
                logger.error(f"Error calculando para {regimen}: {e}")
                resultados[regimen] = {
                    "regimen": regimen,
                    "error": str(e)
                }
        
        # Determinar mejor opción
        mejor_regimen = self._determinar_mejor_regimen(resultados)
        
        return {
            "comparacion": resultados,
            "mejor_regimen": mejor_regimen,
            "recomendacion": self._generar_recomendacion(resultados, mejor_regimen)
        }
    
    def _determinar_mejor_regimen(self, resultados: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Determinar el régimen con menor carga fiscal."""
        mejor = None
        menor_impuesto = float('inf')
        
        for regimen, resultado in resultados.items():
            if "impuesto_total" in resultado:
                impuesto = resultado["impuesto_total"]
                if impuesto < menor_impuesto:
                    menor_impuesto = impuesto
                    mejor = regimen
        
        return mejor
    
    def _generar_recomendacion(
        self,
        resultados: Dict[str, Dict[str, Any]],
        mejor_regimen: Optional[str]
    ) -> str:
        """Generar recomendación basada en la comparación."""
        if not mejor_regimen:
            return "Se requiere análisis más detallado con asesoría fiscal"
        
        mejor_resultado = resultados[mejor_regimen]
        impuesto_mejor = mejor_resultado.get("impuesto_total", 0)
        
        recomendacion = f"Basado en los cálculos, {mejor_regimen} presenta la menor carga fiscal "
        recomendacion += f"con un impuesto anual de ${impuesto_mejor:,.2f}."
        
        return recomendacion
