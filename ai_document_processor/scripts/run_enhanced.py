"""
Script para ejecutar el AI Document Processor Enhanced
====================================================
"""

import asyncio
import os
import sys
import subprocess
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_web_interface():
    """Inicia la interfaz web en segundo plano"""
    try:
        logger.info("🌐 Iniciando interfaz web...")
        
        # Importar y ejecutar interfaz web
        from services.web_interface import run_web_interface
        await run_web_interface()
        
    except Exception as e:
        logger.error(f"Error iniciando interfaz web: {e}")

def start_enhanced_server():
    """Inicia el servidor enhanced principal"""
    try:
        logger.info("🚀 Iniciando servidor enhanced...")
        
        # Configurar variables de entorno
        os.environ.setdefault("HOST", "0.0.0.0")
        os.environ.setdefault("PORT", "8001")
        
        # Importar y ejecutar servidor
        import uvicorn
        uvicorn.run(
            "enhanced_main:app",
            host="0.0.0.0",
            port=8001,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Error iniciando servidor enhanced: {e}")

async def main():
    """Función principal"""
    print("🎉 AI Document Processor Enhanced")
    print("=" * 50)
    print("Iniciando sistema completo...")
    print()
    
    # Verificar dependencias
    try:
        import fastapi
        import uvicorn
        logger.info("✅ Dependencias básicas disponibles")
    except ImportError as e:
        logger.error(f"❌ Dependencias faltantes: {e}")
        print("Instala las dependencias con: pip install -r requirements.txt")
        return
    
    # Verificar configuración
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OpenAI API configurada - Todas las características disponibles")
    else:
        print("⚠️  OpenAI API no configurada - Características limitadas")
        print("   Configura OPENAI_API_KEY para funcionalidad completa")
    
    print()
    print("🚀 Iniciando servicios...")
    print("   - Servidor API: http://localhost:8001")
    print("   - Interfaz Web: http://localhost:8002")
    print("   - Documentación: http://localhost:8001/docs")
    print()
    
    # Iniciar servidor principal
    try:
        start_enhanced_server()
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"Error en servidor principal: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        sys.exit(1)


