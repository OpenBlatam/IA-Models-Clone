#!/usr/bin/env python3
"""
Run Advanced Demo - Ejecutar Demo Avanzado
Script para ejecutar el demo de funcionalidades avanzadas
"""

import asyncio
import sys
import os
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

async def main():
    """Función principal para ejecutar el demo"""
    try:
        print("🚀 Iniciando Demo de Funcionalidades Avanzadas...")
        print("=" * 60)
        
        # Importar y ejecutar el demo
        from advanced_features import main as demo_main
        await demo_main()
        
        print("\n🎉 Demo completado exitosamente!")
        print("\n📋 Funcionalidades demostradas:")
        print("  ✅ Análisis Semántico Avanzado")
        print("  ✅ Análisis de Sentimiento Multi-dimensional")
        print("  ✅ Análisis de Calidad de Contenido")
        print("  ✅ Caché Distribuido con Redis")
        print("  ✅ Análisis de Tendencias Temporales")
        print("  ✅ Detección de Plagio")
        print("  ✅ Cifrado de Datos Sensibles")
        print("  ✅ Autenticación Multi-Factor")
        print("  ✅ Métricas Personalizadas")
        print("  ✅ Alertas Inteligentes")
        
        print("\n🚀 Próximos pasos:")
        print("  1. Instalar dependencias: pip install -r requirements-advanced.txt")
        print("  2. Configurar Redis: docker run -d -p 6379:6379 redis")
        print("  3. Ejecutar demo: python run_advanced_demo.py")
        print("  4. Integrar en tu aplicación principal")
        
    except ImportError as e:
        print(f"❌ Error de importación: {str(e)}")
        print("💡 Asegúrate de instalar las dependencias:")
        print("   pip install -r requirements-advanced.txt")
        
    except Exception as e:
        print(f"❌ Error durante el demo: {str(e)}")
        print("💡 Revisa la configuración y dependencias")

if __name__ == "__main__":
    asyncio.run(main())