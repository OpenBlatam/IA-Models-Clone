"""
Calculadora de Impuestos Especializada
======================================

Servicio especializado para cálculos de impuestos mexicanos.
Soporta diferentes regímenes fiscales y tipos de impuestos.
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


class CalculadoraImpuestos:
    """Calculadora especializada para impuestos mexicanos."""
    
    # Tasas de ISR por régimen (ejemplos - actualizar según legislación vigente)
    TASAS_ISR_RESICO = {
        "tramo_1": {"limite": 300000, "tasa": 0.01},
        "tramo_2": {"limite": 600000, "tasa": 0.015},
        "tramo_3": {"limite": 1000000, "tasa": 0.02},
        "tramo_4": {"limite": 2000000, "tasa": 0.025},
        "tramo_5": {"limite": 3500000, "tasa": 0.03},
        "tramo_6": {"limite": None, "tasa": 0.035}
    }
    
    TASA_IVA = 0.16  # 16% IVA general
    
    # Tabla de ISR para Sueldos y Salarios (2024)
    TABLA_ISR_SUELDOS = [
        {"limite_inferior": 0.01, "limite_superior": 5782.33, "cuota_fija": 0.00, "porcentaje": 1.92},
        {"limite_inferior": 5782.34, "limite_superior": 49101.50, "cuota_fija": 111.07, "porcentaje": 6.40},
        {"limite_inferior": 49101.51, "limite_superior": 86293.18, "cuota_fija": 2883.59, "porcentaje": 10.88},
        {"limite_inferior": 86293.19, "limite_superior": 100313.11, "cuota_fija": 6920.35, "porcentaje": 16.00},
        {"limite_inferior": 100313.12, "limite_superior": 120372.83, "cuota_fija": 9172.74, "porcentaje": 17.92},
        {"limite_inferior": 120372.84, "limite_superior": 144119.88, "cuota_fija": 11332.77, "porcentaje": 21.36},
        {"limite_inferior": 144119.89, "limite_superior": 290667.75, "cuota_fija": 16402.18, "porcentaje": 23.52},
        {"limite_inferior": 290667.76, "limite_superior": 999999999.99, "cuota_fija": 50837.57, "porcentaje": 30.00}
    ]
    
    # Tabla de ISR para PFAE (simplificada)
    TASA_ISR_PFAE = 0.30  # 30% sobre utilidad fiscal
    
    def calcular_isr_resico(self, ingresos_anuales: float) -> Dict[str, Any]:
        """
        Calcular ISR para régimen RESICO.
        
        Args:
            ingresos_anuales: Ingresos anuales del contribuyente
        
        Returns:
            Diccionario con el cálculo detallado
        """
        ingresos = Decimal(str(ingresos_anuales))
        impuesto_total = Decimal("0")
        desglose = []
        
        ingresos_restantes = ingresos
        tramo_anterior = Decimal("0")
        
        for tramo_key, tramo_data in self.TASAS_ISR_RESICO.items():
            limite = tramo_data["limite"]
            tasa = Decimal(str(tramo_data["tasa"]))
            
            if limite is None:
                # Último tramo
                base = ingresos_restantes
                impuesto = base * tasa
                impuesto_total += impuesto
                desglose.append({
                    "tramo": tramo_key,
                    "base": float(base),
                    "tasa": float(tasa),
                    "impuesto": float(impuesto)
                })
                break
            else:
                limite_decimal = Decimal(str(limite))
                if ingresos_restantes > 0:
                    base_tramo = min(ingresos_restantes, limite_decimal - tramo_anterior)
                    if base_tramo > 0:
                        impuesto_tramo = base_tramo * tasa
                        impuesto_total += impuesto_tramo
                        desglose.append({
                            "tramo": tramo_key,
                            "base": float(base_tramo),
                            "tasa": float(tasa),
                            "impuesto": float(impuesto_tramo)
                        })
                        ingresos_restantes -= base_tramo
                        tramo_anterior = limite_decimal
        
        return {
            "regimen": "RESICO",
            "tipo_impuesto": "ISR",
            "ingresos_anuales": float(ingresos),
            "impuesto_total": float(impuesto_total),
            "tasa_efectiva": float(impuesto_total / ingresos if ingresos > 0 else 0),
            "desglose": desglose
        }
    
    def calcular_iva(self, base_imponible: float, tipo: str = "general") -> Dict[str, Any]:
        """
        Calcular IVA.
        
        Args:
            base_imponible: Base imponible
            tipo: Tipo de IVA (general, exento, tasa_cero)
        
        Returns:
            Diccionario con el cálculo
        """
        base = Decimal(str(base_imponible))
        
        if tipo == "exento":
            iva = Decimal("0")
            tasa = Decimal("0")
        elif tipo == "tasa_cero":
            iva = Decimal("0")
            tasa = Decimal("0")
        else:
            tasa = Decimal(str(self.TASA_IVA))
            iva = base * tasa
        
        return {
            "tipo_impuesto": "IVA",
            "base_imponible": float(base),
            "tasa": float(tasa),
            "iva": float(iva),
            "total_con_iva": float(base + iva)
        }
    
    def calcular_impuesto_mensual(
        self,
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcular impuesto mensual según régimen.
        
        Args:
            regimen: Régimen fiscal
            tipo_impuesto: Tipo de impuesto
            datos: Datos del contribuyente
        
        Returns:
            Diccionario con el cálculo
        """
        if regimen == "RESICO" and tipo_impuesto == "ISR":
            ingresos_mensuales = datos.get("ingresos_mensuales", 0)
            ingresos_anuales = ingresos_mensuales * 12
            resultado_anual = self.calcular_isr_resico(ingresos_anuales)
            resultado_anual["ingresos_mensuales"] = ingresos_mensuales
            resultado_anual["impuesto_mensual"] = resultado_anual["impuesto_total"] / 12
            return resultado_anual
        
        elif tipo_impuesto == "IVA":
            base_imponible = datos.get("base_imponible", 0)
            tipo_iva = datos.get("tipo_iva", "general")
            return self.calcular_iva(base_imponible, tipo_iva)
        
        elif regimen == "PFAE" and tipo_impuesto == "ISR":
            ingresos = datos.get("ingresos", 0)
            gastos_deducibles = datos.get("gastos_deducibles", 0)
            utilidad_fiscal = ingresos - gastos_deducibles
            
            if utilidad_fiscal <= 0:
                impuesto = 0
            else:
                impuesto = utilidad_fiscal * self.TASA_ISR_PFAE
            
            return {
                "regimen": "PFAE",
                "tipo_impuesto": "ISR",
                "ingresos": ingresos,
                "gastos_deducibles": gastos_deducibles,
                "utilidad_fiscal": utilidad_fiscal,
                "tasa": self.TASA_ISR_PFAE,
                "impuesto_total": impuesto,
                "tasa_efectiva": (impuesto / ingresos * 100) if ingresos > 0 else 0
            }
        
        elif regimen == "Sueldos y Salarios" and tipo_impuesto == "ISR":
            sueldo_mensual = datos.get("sueldo_mensual", 0)
            return self.calcular_isr_sueldos(sueldo_mensual)
        
        else:
            return {
                "error": f"Cálculo no implementado para régimen {regimen} y tipo {tipo_impuesto}"
            }
    
    def calcular_isr_sueldos(self, sueldo_mensual: float) -> Dict[str, Any]:
        """
        Calcular ISR para Sueldos y Salarios.
        
        Args:
            sueldo_mensual: Sueldo mensual del trabajador
        
        Returns:
            Diccionario con el cálculo detallado
        """
        sueldo = Decimal(str(sueldo_mensual))
        sueldo_anual = sueldo * 12
        
        # Encontrar el tramo correspondiente
        impuesto_anual = Decimal("0")
        tramo_aplicado = None
        
        for tramo in self.TABLA_ISR_SUELDOS:
            limite_inf = Decimal(str(tramo["limite_inferior"]))
            limite_sup = Decimal(str(tramo["limite_superior"]))
            
            if limite_inf <= sueldo_anual <= limite_sup:
                excedente = sueldo_anual - limite_inf
                porcentaje = Decimal(str(tramo["porcentaje"])) / 100
                cuota_fija = Decimal(str(tramo["cuota_fija"]))
                
                impuesto_anual = cuota_fija + (excedente * porcentaje)
                tramo_aplicado = tramo
                break
        
        impuesto_mensual = impuesto_anual / 12
        
        return {
            "regimen": "Sueldos y Salarios",
            "tipo_impuesto": "ISR",
            "sueldo_mensual": float(sueldo),
            "sueldo_anual": float(sueldo_anual),
            "impuesto_anual": float(impuesto_anual),
            "impuesto_mensual": float(impuesto_mensual),
            "tasa_efectiva": float(impuesto_anual / sueldo_anual * 100) if sueldo_anual > 0 else 0,
            "tramo_aplicado": tramo_aplicado
        }
