#!/usr/bin/env python3
"""
Export IA V4.0 - Sistema Enterprise Completo
Script de ejecución principal para el sistema completo
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("export_ia_v4.log")
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Imprimir banner del sistema."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                    🚀 EXPORT IA V4.0 - ULTIMATE ENTERPRISE 🚀                ║
    ║                                                                              ║
    ║  🔐 Security System    🤖 Automation     📊 Monitoring  📈 Analytics        ║
    ║  💾 Data Management    🧠 NLP Advanced   📄 Export IA   🔔 Notifications    ║
    ║  🤖 Machine Learning   🧠 Deep Learning  👁️ Computer Vision ⛓️ Blockchain   ║
    ║                                                                              ║
    ║                    Sistema Enterprise Completo con IA                       ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Verificar dependencias del sistema."""
    logger.info("🔍 Verificando dependencias del sistema...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "opencv-python",
        "cryptography",
        "ecdsa"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} - FALTANTE")
    
    if missing_packages:
        logger.error(f"❌ Faltan {len(missing_packages)} paquetes requeridos:")
        for package in missing_packages:
            logger.error(f"   - {package}")
        logger.error("💡 Instale las dependencias con:")
        logger.error("   pip install -r requirements.txt")
        logger.error("   pip install -r requirements_nlp.txt")
        logger.error("   pip install -r requirements_ai.txt")
        logger.error("   pip install -r requirements_blockchain.txt")
        return False
    
    logger.info("✅ Todas las dependencias están instaladas")
    return True

def check_directories():
    """Verificar y crear directorios necesarios."""
    logger.info("📁 Verificando directorios del sistema...")
    
    directories = [
        "data",
        "models",
        "deep_learning_models",
        "cv_images",
        "blockchain_data",
        "logs",
        "cache"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Directorio creado: {directory}")
        else:
            logger.info(f"✅ Directorio existe: {directory}")
    
    logger.info("✅ Todos los directorios están listos")

def print_system_info():
    """Imprimir información del sistema."""
    logger.info("📊 Información del Sistema V4.0:")
    logger.info("   🏗️  Arquitectura: Enterprise Microservices")
    logger.info("   🔐 Seguridad: Nivel Enterprise")
    logger.info("   🤖 IA: Machine Learning + Deep Learning + Computer Vision")
    logger.info("   ⛓️  Blockchain: Sistema completo con Proof of Work")
    logger.info("   📊 Monitoreo: Tiempo real con alertas automáticas")
    logger.info("   🔔 Notificaciones: Multi-canal (Email, SMS, Slack, Teams, Discord)")
    logger.info("   📈 Analytics: Business Intelligence avanzado")
    logger.info("   💾 Datos: Almacenamiento multi-nivel con cache inteligente")
    logger.info("   🚀 Rendimiento: Optimizado para producción enterprise")

async def main():
    """Función principal."""
    try:
        print_banner()
        
        logger.info("🚀 Iniciando Export IA V4.0 - Sistema Enterprise Completo")
        
        # Verificar dependencias
        if not check_dependencies():
            logger.error("❌ Error: Dependencias faltantes")
            sys.exit(1)
        
        # Verificar directorios
        check_directories()
        
        # Imprimir información del sistema
        print_system_info()
        
        # Importar y ejecutar la aplicación
        logger.info("🔄 Cargando aplicación V4.0...")
        
        from app.api.enhanced_app_v4 import app
        import uvicorn
        
        logger.info("✅ Aplicación V4.0 cargada exitosamente")
        logger.info("🌐 Iniciando servidor en http://localhost:8000")
        logger.info("📚 Documentación disponible en http://localhost:8000/docs")
        logger.info("🔧 Health check disponible en http://localhost:8000/health")
        logger.info("")
        logger.info("🎯 Sistemas disponibles:")
        logger.info("   🔐 Seguridad: /api/v1/advanced/security")
        logger.info("   🤖 Automatización: /api/v1/advanced/automation")
        logger.info("   💾 Datos: /api/v1/advanced/data")
        logger.info("   📊 Monitoreo: /api/v1/advanced/monitoring")
        logger.info("   📈 Analytics: /api/v1/analytics")
        logger.info("   🔔 Notificaciones: /api/v1/notifications")
        logger.info("   🤖 Machine Learning: /api/v1/ai/ml")
        logger.info("   🧠 Deep Learning: /api/v1/ai/dl")
        logger.info("   👁️ Computer Vision: /api/v1/ai/cv")
        logger.info("   ⛓️ Blockchain: /api/v1/blockchain")
        logger.info("   🧠 NLP: /api/v1/nlp")
        logger.info("   📄 Export IA: /api/v1/export")
        logger.info("")
        logger.info("🏆 SISTEMA ENTERPRISE COMPLETO V4.0 - LISTO PARA PRODUCCIÓN 🏆")
        logger.info("")
        
        # Ejecutar servidor
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Sistema detenido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())




