"""
PDF Variantes - Sistema Listo para Usar
Archivo de inicio rápido del sistema completo
"""

import asyncio
import logging
import sys
from pathlib import Path

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar el sistema completo
from system import pdf_variantes_system, initialize_system, cleanup_system

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def start_system():
    """Iniciar el sistema PDF Variantes"""
    try:
        logger.info("🚀 Iniciando Sistema PDF Variantes - Listo para Usar")
        
        # Inicializar sistema
        success = await initialize_system()
        
        if success:
            logger.info("✅ Sistema PDF Variantes iniciado exitosamente")
            logger.info("🌐 API disponible en: http://localhost:8000")
            logger.info("📚 Documentación disponible en: http://localhost:8000/docs")
            logger.info("🏥 Health check disponible en: http://localhost:8000/health")
            logger.info("📊 Métricas disponibles en: http://localhost:8000/metrics")
            
            # Mostrar estado del sistema
            status = await pdf_variantes_system.get_system_status()
            logger.info(f"📈 Estado del sistema: {status['system_status']['initialized']}")
            logger.info(f"🔧 Servicios cargados: {status['system_status']['services_loaded']}/{status['system_status']['total_services']}")
            
            # Mantener el sistema ejecutándose
            logger.info("🔄 Sistema ejecutándose... Presiona Ctrl+C para detener")
            
            while True:
                await asyncio.sleep(60)  # Verificar cada minuto
                
                # Verificar salud del sistema
                current_status = await pdf_variantes_system.get_system_status()
                if not current_status.get("system_status", {}).get("initialized", False):
                    logger.warning("⚠️ Sistema no está inicializado correctamente")
                    break
                    
        else:
            logger.error("❌ Error al iniciar el sistema")
            return False
            
    except KeyboardInterrupt:
        logger.info("🛑 Deteniendo sistema...")
        await cleanup_system()
        logger.info("✅ Sistema detenido exitosamente")
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        await cleanup_system()
        return False
    
    return True

def main():
    """Función principal"""
    print("=" * 60)
    print("🚀 PDF VARIANTES - SISTEMA COMPLETO Y LISTO PARA USAR")
    print("=" * 60)
    print("📋 Características incluidas:")
    print("   ✅ API REST completa (50+ endpoints)")
    print("   ✅ WebSockets para colaboración en tiempo real")
    print("   ✅ IA de próxima generación (GPT-4, Claude-3, Llama-2)")
    print("   ✅ Sistema de plugins extensible")
    print("   ✅ Integración blockchain y Web3")
    print("   ✅ Monitoreo y analytics avanzados")
    print("   ✅ Seguridad empresarial")
    print("   ✅ Caché multinivel optimizado")
    print("   ✅ Computación cuántica simulada")
    print("   ✅ Exportación en múltiples formatos")
    print("=" * 60)
    print("🌐 URLs disponibles:")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("   🏥 Health Check: http://localhost:8000/health")
    print("   📊 Métricas: http://localhost:8000/metrics")
    print("   🔄 WebSocket: ws://localhost:8000/ws")
    print("=" * 60)
    print("🚀 Iniciando sistema...")
    print("=" * 60)
    
    # Ejecutar sistema
    success = asyncio.run(start_system())
    
    if success:
        print("✅ Sistema iniciado exitosamente")
    else:
        print("❌ Error al iniciar el sistema")
        sys.exit(1)

if __name__ == "__main__":
    main()
