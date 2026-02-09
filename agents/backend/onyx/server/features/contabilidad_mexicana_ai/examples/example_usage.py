"""
Ejemplo de uso de Contabilidad Mexicana AI
"""

import asyncio
import os
from dotenv import load_dotenv

from contabilidad_mexicana_ai import ContadorAI, ContadorConfig

load_dotenv()


async def ejemplo_calculo_impuestos():
    """Ejemplo de cálculo de impuestos."""
    print("=" * 60)
    print("Ejemplo 1: Cálculo de Impuestos RESICO")
    print("=" * 60)
    
    config = ContadorConfig()
    contador = ContadorAI(config)
    
    resultado = await contador.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={
            "ingresos_mensuales": 50000,
            "gastos_deducibles": 10000
        }
    )
    
    print(f"Resultado: {resultado['resultado']}")
    print()


async def ejemplo_asesoria_fiscal():
    """Ejemplo de asesoría fiscal."""
    print("=" * 60)
    print("Ejemplo 2: Asesoría Fiscal")
    print("=" * 60)
    
    config = ContadorConfig()
    contador = ContadorAI(config)
    
    asesoria = await contador.asesoria_fiscal(
        pregunta="¿Qué deducciones puedo aplicar en RESICO?",
        contexto={
            "regimen": "RESICO",
            "actividad": "Servicios profesionales"
        }
    )
    
    print(f"Asesoría: {asesoria['asesoria']}")
    print()


async def ejemplo_guia_fiscal():
    """Ejemplo de guía fiscal."""
    print("=" * 60)
    print("Ejemplo 3: Guía Fiscal")
    print("=" * 60)
    
    config = ContadorConfig()
    contador = ContadorAI(config)
    
    guia = await contador.guia_fiscal(
        tema="Deducciones para emprendedores en RESICO",
        nivel_detalle="completo"
    )
    
    print(f"Guía: {guia['guia']}")
    print()


async def ejemplo_tramite_sat():
    """Ejemplo de información sobre trámite SAT."""
    print("=" * 60)
    print("Ejemplo 4: Trámite SAT")
    print("=" * 60)
    
    config = ContadorConfig()
    contador = ContadorAI(config)
    
    tramite = await contador.tramite_sat(
        tipo_tramite="Alta en RFC",
        detalles={
            "tipo_persona": "Persona Física"
        }
    )
    
    print(f"Información: {tramite['informacion']}")
    print()


async def ejemplo_ayuda_declaracion():
    """Ejemplo de ayuda con declaración."""
    print("=" * 60)
    print("Ejemplo 5: Ayuda con Declaración")
    print("=" * 60)
    
    config = ContadorConfig()
    contador = ContadorAI(config)
    
    ayuda = await contador.ayuda_declaracion(
        tipo_declaracion="mensual",
        periodo="2024-01",
        datos={
            "regimen": "RESICO",
            "ingresos": 50000
        }
    )
    
    print(f"Guía: {ayuda['guia']}")
    print()


async def main():
    """Ejecutar todos los ejemplos."""
    print("\n" + "=" * 60)
    print("CONTABILIDAD MEXICANA AI - Ejemplos de Uso")
    print("=" * 60 + "\n")
    
    try:
        await ejemplo_calculo_impuestos()
        await ejemplo_asesoria_fiscal()
        await ejemplo_guia_fiscal()
        await ejemplo_tramite_sat()
        await ejemplo_ayuda_declaracion()
        
        print("=" * 60)
        print("Todos los ejemplos completados exitosamente")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cerrar conexiones
        config = ContadorConfig()
        contador = ContadorAI(config)
        await contador.close()


if __name__ == "__main__":
    asyncio.run(main())
