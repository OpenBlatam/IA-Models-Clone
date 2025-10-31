"""
PDF Variantes - Sistema Completo y Listo para Usar
Archivo principal de inicialización del sistema completo
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar configuración
from utils.config import get_settings, Settings
from utils.logging_config import setup_logging

# Importar servicios principales
from services.pdf_service import PDFVariantesService
from services.collaboration_service import CollaborationService
from services.monitoring_service import MonitoringSystem, AnalyticsService, HealthService, NotificationService

# Importar API
from api.main import app
from api.routes import pdf_router

# Importar sistema ultra-avanzado
from ultra_processing import UltraAdvancedSystem

logger = logging.getLogger(__name__)

class PDFVariantesSystem:
    """Sistema completo de PDF Variantes listo para usar"""
    
    def __init__(self):
        self.settings = get_settings()
        self.is_initialized = False
        
        # Servicios principales
        self.pdf_service: Optional[PDFVariantesService] = None
        self.collaboration_service: Optional[CollaborationService] = None
        self.monitoring_system: Optional[MonitoringSystem] = None
        self.analytics_service: Optional[AnalyticsService] = None
        self.health_service: Optional[HealthService] = None
        self.notification_service: Optional[NotificationService] = None
        
        # Sistema ultra-avanzado
        self.ultra_system: Optional[UltraAdvancedSystem] = None
        
        # Estado del sistema
        self.system_status = {
            "initialized": False,
            "services_loaded": 0,
            "total_services": 7,
            "startup_time": None,
            "last_health_check": None,
            "errors": []
        }
    
    async def initialize(self) -> bool:
        """Inicializar sistema completo"""
        try:
            logger.info("🚀 Inicializando Sistema PDF Variantes Completo")
            start_time = datetime.utcnow()
            
            # Configurar logging
            setup_logging(
                log_level=self.settings.LOG_LEVEL,
                log_file=self.settings.LOG_FILE
            )
            
            # Inicializar servicios principales
            await self._initialize_core_services()
            
            # Inicializar sistema ultra-avanzado
            await self._initialize_ultra_system()
            
            # Verificar salud del sistema
            await self._perform_health_check()
            
            # Marcar como inicializado
            self.is_initialized = True
            self.system_status["initialized"] = True
            self.system_status["startup_time"] = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"✅ Sistema PDF Variantes inicializado exitosamente en {self.system_status['startup_time']:.2f}s")
            logger.info(f"📊 Servicios cargados: {self.system_status['services_loaded']}/{self.system_status['total_services']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al inicializar el sistema: {e}")
            self.system_status["errors"].append(str(e))
            return False
    
    async def _initialize_core_services(self):
        """Inicializar servicios principales"""
        try:
            logger.info("🔧 Inicializando servicios principales")
            
            # Servicio PDF
            self.pdf_service = PDFVariantesService(self.settings)
            await self.pdf_service.initialize()
            self.system_status["services_loaded"] += 1
            logger.info("✅ Servicio PDF inicializado")
            
            # Servicio de colaboración
            self.collaboration_service = CollaborationService(self.settings)
            await self.collaboration_service.initialize()
            self.system_status["services_loaded"] += 1
            logger.info("✅ Servicio de Colaboración inicializado")
            
            # Sistema de monitoreo
            self.monitoring_system = MonitoringSystem(self.settings)
            await self.monitoring_system.initialize()
            self.system_status["services_loaded"] += 1
            logger.info("✅ Sistema de Monitoreo inicializado")
            
            # Servicio de analytics
            self.analytics_service = AnalyticsService(self.settings)
            await self.analytics_service.initialize()
            self.system_status["services_loaded"] += 1
            logger.info("✅ Servicio de Analytics inicializado")
            
            # Servicio de salud
            self.health_service = HealthService(self.settings)
            await self.health_service.initialize()
            self.system_status["services_loaded"] += 1
            logger.info("✅ Servicio de Salud inicializado")
            
            # Servicio de notificaciones
            self.notification_service = NotificationService(self.settings)
            await self.notification_service.initialize()
            self.system_status["services_loaded"] += 1
            logger.info("✅ Servicio de Notificaciones inicializado")
            
            logger.info("✅ Servicios principales inicializados exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error al inicializar servicios principales: {e}")
            raise
    
    async def _initialize_ultra_system(self):
        """Inicializar sistema ultra-avanzado"""
        try:
            logger.info("🚀 Inicializando sistema ultra-avanzado")
            
            # Sistema ultra-avanzado
            self.ultra_system = UltraAdvancedSystem(self.settings)
            await self.ultra_system.initialize()
            self.system_status["services_loaded"] += 1
            logger.info("✅ Sistema Ultra-Avanzado inicializado")
            
            logger.info("✅ Sistema ultra-avanzado inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error al inicializar sistema ultra-avanzado: {e}")
            raise
    
    async def _perform_health_check(self):
        """Realizar verificación de salud del sistema"""
        try:
            logger.info("🏥 Realizando verificación de salud del sistema")
            
            # Verificar servicios principales
            health_status = await self.health_service.get_system_health()
            
            # Verificar sistema ultra-avanzado
            if self.ultra_system:
                ultra_status = await self.ultra_system.get_system_status()
                health_status["ultra_system"] = ultra_status
            
            self.system_status["last_health_check"] = datetime.utcnow().isoformat()
            self.system_status["health_status"] = health_status
            
            logger.info("✅ Verificación de salud completada exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error en verificación de salud: {e}")
            self.system_status["errors"].append(str(e))
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema"""
        try:
            return {
                "system_status": self.system_status,
                "services": {
                    "pdf_service": self.pdf_service is not None,
                    "collaboration_service": self.collaboration_service is not None,
                    "monitoring_system": self.monitoring_system is not None,
                    "analytics_service": self.analytics_service is not None,
                    "health_service": self.health_service is not None,
                    "notification_service": self.notification_service is not None,
                    "ultra_system": self.ultra_system is not None
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return {}
    
    async def generate_variants(self, content: str, count: int = 10) -> List[Dict[str, Any]]:
        """Generar variantes del contenido"""
        try:
            if not self.is_initialized:
                raise Exception("Sistema no inicializado")
            
            # Usar sistema ultra-avanzado para generar variantes
            if self.ultra_system:
                variants = await self.ultra_system.generate_ultra_variants(content, count)
            else:
                # Fallback a servicio principal
                variants = await self.pdf_service.generate_variants(content, count)
            
            return variants
            
        except Exception as e:
            logger.error(f"Error generando variantes: {e}")
            return []
    
    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analizar contenido"""
        try:
            if not self.is_initialized:
                raise Exception("Sistema no inicializado")
            
            # Usar sistema ultra-avanzado para análisis
            if self.ultra_system:
                analysis = await self.ultra_system.analyze_content_ultra(content)
            else:
                # Fallback a servicio principal
                analysis = await self.pdf_service.analyze_content(content)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando contenido: {e}")
            return {}
    
    async def cleanup(self):
        """Limpiar sistema"""
        try:
            logger.info("🧹 Limpiando sistema PDF Variantes")
            
            # Limpiar servicios principales
            if self.pdf_service:
                await self.pdf_service.cleanup()
            if self.collaboration_service:
                await self.collaboration_service.cleanup()
            if self.monitoring_system:
                await self.monitoring_system.cleanup()
            if self.analytics_service:
                await self.analytics_service.cleanup()
            if self.health_service:
                await self.health_service.cleanup()
            if self.notification_service:
                await self.notification_service.cleanup()
            
            # Limpiar sistema ultra-avanzado
            if self.ultra_system:
                await self.ultra_system.cleanup()
            
            logger.info("✅ Sistema PDF Variantes limpiado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error limpiando sistema: {e}")

# Instancia global del sistema
pdf_variantes_system = PDFVariantesSystem()

async def initialize_system():
    """Inicializar el sistema completo"""
    return await pdf_variantes_system.initialize()

async def get_system():
    """Obtener instancia del sistema"""
    return pdf_variantes_system

async def cleanup_system():
    """Limpiar el sistema"""
    return await pdf_variantes_system.cleanup()

# Función principal para ejecutar el sistema
async def main():
    """Función principal"""
    try:
        logger.info("🚀 Iniciando Sistema PDF Variantes")
        
        # Inicializar sistema
        success = await initialize_system()
        
        if success:
            logger.info("✅ Sistema PDF Variantes iniciado exitosamente")
            logger.info("🌐 API disponible en: http://localhost:8000")
            logger.info("📚 Documentación disponible en: http://localhost:8000/docs")
            logger.info("🏥 Health check disponible en: http://localhost:8000/health")
            
            # Mantener el sistema ejecutándose
            while True:
                await asyncio.sleep(60)  # Verificar cada minuto
                
                # Verificar salud del sistema
                status = await pdf_variantes_system.get_system_status()
                if not status.get("system_status", {}).get("initialized", False):
                    logger.warning("⚠️ Sistema no está inicializado correctamente")
                    break
        else:
            logger.error("❌ Error al iniciar el sistema")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Deteniendo sistema...")
        await cleanup_system()
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        await cleanup_system()
        sys.exit(1)

if __name__ == "__main__":
    # Ejecutar sistema
    asyncio.run(main())
