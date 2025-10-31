#!/usr/bin/env python3
"""
Ultra-Modular AI Document Processor - Main Application
====================================================

Ultra-modular system with microservices, plugins, and event-driven architecture.
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path
from typing import Optional, Dict, Any

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.registry import ComponentRegistry, ServiceRegistry, get_component_registry, get_service_registry
from modules.plugins import PluginManager, get_plugin_manager
from modules.events import EventBus, EventStore, get_event_bus, get_event_store
from modules.gateway import APIGateway, get_api_gateway
from modules.microservices import ServiceContainer, ServiceDiscovery, get_service_container, get_service_discovery
from microservices.document_processor_service import DocumentProcessorService
from microservices.ai_service import AIService
from microservices.transform_service import TransformService
from microservices.validation_service import ValidationService
from microservices.cache_service import CacheService
from microservices.file_service import FileService
from microservices.notification_service import NotificationService
from microservices.metrics_service import MetricsService
from microservices.api_gateway_service import APIGatewayService
from microservices.message_bus_service import MessageBusService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ultra_modular_processor.log')
    ]
)
logger = logging.getLogger(__name__)


class UltraModularSystem:
    """Ultra-modular AI document processing system."""
    
    def __init__(self):
        self.system_name = "Ultra-Modular AI Document Processor"
        self.version = "4.0.0"
        self.status = "initializing"
        
        # Core systems
        self.component_registry = get_component_registry()
        self.service_registry = get_service_registry()
        self.plugin_manager = get_plugin_manager()
        self.event_bus = get_event_bus()
        self.event_store = get_event_store()
        self.api_gateway = get_api_gateway()
        self.service_container = get_service_container()
        self.service_discovery = get_service_discovery()
        
        # Services
        self.services: Dict[str, Any] = {}
        self.service_configurations: Dict[str, Any] = {}
        
        # System stats
        self.stats = {
            'start_time': None,
            'services_started': 0,
            'components_registered': 0,
            'plugins_loaded': 0,
            'events_processed': 0,
            'requests_processed': 0
        }
    
    async def initialize(self):
        """Initialize the ultra-modular system."""
        try:
            logger.info(f"🚀 Initializing {self.system_name} v{self.version}")
            
            # Start core systems
            await self._start_core_systems()
            
            # Discover and load plugins
            await self._discover_plugins()
            
            # Initialize services
            await self._initialize_services()
            
            # Configure API gateway
            await self._configure_api_gateway()
            
            # Start services
            await self._start_services()
            
            # Register services with discovery
            await self._register_services()
            
            self.status = "running"
            self.stats['start_time'] = asyncio.get_event_loop().time()
            
            logger.info("✅ Ultra-modular system initialized successfully")
            
        except Exception as e:
            self.status = "error"
            logger.error(f"❌ Failed to initialize ultra-modular system: {e}")
            raise
    
    async def _start_core_systems(self):
        """Start core systems."""
        logger.info("🔧 Starting core systems...")
        
        # Start event bus
        await self.event_bus.start()
        logger.info("✅ Event bus started")
        
        # Start component registry health monitoring
        await self.component_registry.start_health_monitoring()
        logger.info("✅ Component registry health monitoring started")
        
        # Start plugin manager
        await self.plugin_manager.discover_plugins()
        logger.info("✅ Plugin manager initialized")
        
        logger.info("✅ Core systems started")
    
    async def _discover_plugins(self):
        """Discover and load plugins."""
        logger.info("🔍 Discovering plugins...")
        
        try:
            # Discover plugins
            discovered_plugins = await self.plugin_manager.discover_plugins()
            
            # Load plugins
            for plugin_info in discovered_plugins:
                try:
                    success = await self.plugin_manager.load_plugin(plugin_info)
                    if success:
                        self.stats['plugins_loaded'] += 1
                        logger.info(f"✅ Loaded plugin: {plugin_info.name}")
                    else:
                        logger.warning(f"⚠️ Failed to load plugin: {plugin_info.name}")
                except Exception as e:
                    logger.error(f"❌ Error loading plugin {plugin_info.name}: {e}")
            
            logger.info(f"✅ Plugin discovery completed: {self.stats['plugins_loaded']} plugins loaded")
            
        except Exception as e:
            logger.error(f"❌ Plugin discovery failed: {e}")
            raise
    
    async def _initialize_services(self):
        """Initialize microservices."""
        logger.info("🏗️ Initializing microservices...")
        
        # Document Processor Service
        doc_processor_config = {
            'name': 'document_processor',
            'service_type': 'document_processor',
            'host': 'localhost',
            'port': 8001,
            'custom_config': {
                'max_workers': 4,
                'queue_size': 1000
            }
        }
        self.service_configurations['document_processor'] = doc_processor_config
        
        # AI Service
        ai_service_config = {
            'name': 'ai_service',
            'service_type': 'ai_service',
            'host': 'localhost',
            'port': 8002,
            'custom_config': {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo'
            }
        }
        self.service_configurations['ai_service'] = ai_service_config
        
        # Transform Service
        transform_service_config = {
            'name': 'transform_service',
            'service_type': 'transform_service',
            'host': 'localhost',
            'port': 8003,
            'custom_config': {
                'supported_formats': ['pdf', 'docx', 'md', 'html']
            }
        }
        self.service_configurations['transform_service'] = transform_service_config
        
        # Validation Service
        validation_service_config = {
            'name': 'validation_service',
            'service_type': 'validation_service',
            'host': 'localhost',
            'port': 8004,
            'custom_config': {
                'max_file_size': 100 * 1024 * 1024,  # 100MB
                'allowed_extensions': ['.pdf', '.docx', '.md', '.html', '.txt']
            }
        }
        self.service_configurations['validation_service'] = validation_service_config
        
        # Cache Service
        cache_service_config = {
            'name': 'cache_service',
            'service_type': 'cache_service',
            'host': 'localhost',
            'port': 8005,
            'custom_config': {
                'backend': 'redis',
                'max_memory': '1GB'
            }
        }
        self.service_configurations['cache_service'] = cache_service_config
        
        # File Service
        file_service_config = {
            'name': 'file_service',
            'service_type': 'file_service',
            'host': 'localhost',
            'port': 8006,
            'custom_config': {
                'temp_dir': '/tmp',
                'max_temp_files': 1000
            }
        }
        self.service_configurations['file_service'] = file_service_config
        
        # Notification Service
        notification_service_config = {
            'name': 'notification_service',
            'service_type': 'notification_service',
            'host': 'localhost',
            'port': 8007,
            'custom_config': {
                'channels': ['email', 'webhook', 'slack']
            }
        }
        self.service_configurations['notification_service'] = notification_service_config
        
        # Metrics Service
        metrics_service_config = {
            'name': 'metrics_service',
            'service_type': 'metrics_service',
            'host': 'localhost',
            'port': 8008,
            'custom_config': {
                'retention_days': 30,
                'aggregation_interval': 60
            }
        }
        self.service_configurations['metrics_service'] = metrics_service_config
        
        # API Gateway Service
        api_gateway_config = {
            'name': 'api_gateway',
            'service_type': 'api_gateway',
            'host': 'localhost',
            'port': 8000,
            'custom_config': {
                'rate_limit': 1000,
                'timeout': 30
            }
        }
        self.service_configurations['api_gateway'] = api_gateway_config
        
        # Message Bus Service
        message_bus_config = {
            'name': 'message_bus',
            'service_type': 'message_bus',
            'host': 'localhost',
            'port': 8009,
            'custom_config': {
                'max_queue_size': 10000,
                'retry_attempts': 3
            }
        }
        self.service_configurations['message_bus'] = message_bus_config
        
        logger.info("✅ Service configurations created")
    
    async def _configure_api_gateway(self):
        """Configure API gateway routes and middleware."""
        logger.info("🌐 Configuring API gateway...")
        
        try:
            # Register middleware
            from modules.gateway import AuthenticationMiddleware, RateLimitingMiddleware, LoggingMiddleware
            
            auth_middleware = AuthenticationMiddleware()
            rate_limit_middleware = RateLimitingMiddleware(requests_per_minute=1000)
            logging_middleware = LoggingMiddleware()
            
            await self.api_gateway.register_middleware(auth_middleware)
            await self.api_gateway.register_middleware(rate_limit_middleware)
            await self.api_gateway.register_middleware(logging_middleware)
            
            # Register routes
            from modules.gateway import Route, RouteMethod, ServiceEndpoint, LoadBalancingStrategy
            
            # Document processing route
            doc_endpoints = [
                ServiceEndpoint(
                    id="doc_processor_1",
                    name="document_processor",
                    url="http://localhost:8001",
                    health_check_url="http://localhost:8001/health"
                )
            ]
            
            doc_route = Route(
                id="document_processing",
                path="/api/v1/documents/*",
                methods=[RouteMethod.POST, RouteMethod.GET, RouteMethod.PUT, RouteMethod.DELETE],
                service_endpoints=doc_endpoints,
                load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                middleware=["authentication", "rate_limiting", "logging"]
            )
            
            await self.api_gateway.register_route(doc_route)
            
            # AI service route
            ai_endpoints = [
                ServiceEndpoint(
                    id="ai_service_1",
                    name="ai_service",
                    url="http://localhost:8002",
                    health_check_url="http://localhost:8002/health"
                )
            ]
            
            ai_route = Route(
                id="ai_processing",
                path="/api/v1/ai/*",
                methods=[RouteMethod.POST, RouteMethod.GET],
                service_endpoints=ai_endpoints,
                load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                middleware=["authentication", "rate_limiting", "logging"]
            )
            
            await self.api_gateway.register_route(ai_route)
            
            logger.info("✅ API gateway configured")
            
        except Exception as e:
            logger.error(f"❌ API gateway configuration failed: {e}")
            raise
    
    async def _start_services(self):
        """Start all microservices."""
        logger.info("🚀 Starting microservices...")
        
        # Create and start services
        service_classes = {
            'document_processor': DocumentProcessorService,
            'ai_service': AIService,
            'transform_service': TransformService,
            'validation_service': ValidationService,
            'cache_service': CacheService,
            'file_service': FileService,
            'notification_service': NotificationService,
            'metrics_service': MetricsService,
            'api_gateway': APIGatewayService,
            'message_bus': MessageBusService
        }
        
        for service_name, service_class in service_classes.items():
            try:
                if service_name in self.service_configurations:
                    config = self.service_configurations[service_name]
                    
                    # Create service instance
                    service = service_class(config)
                    
                    # Register with container
                    service_id = await self.service_container.register_service(service)
                    
                    # Start service
                    success = await self.service_container.start_service(service_id)
                    
                    if success:
                        self.services[service_name] = service
                        self.stats['services_started'] += 1
                        logger.info(f"✅ Started service: {service_name}")
                    else:
                        logger.error(f"❌ Failed to start service: {service_name}")
                
            except Exception as e:
                logger.error(f"❌ Error starting service {service_name}: {e}")
        
        logger.info(f"✅ Microservices started: {self.stats['services_started']} services")
    
    async def _register_services(self):
        """Register services with service discovery."""
        logger.info("🔍 Registering services with discovery...")
        
        for service_name, service in self.services.items():
            try:
                config = self.service_configurations[service_name]
                
                await self.service_discovery.register_service_instance(
                    service_name=service_name,
                    instance_id=f"{service_name}_1",
                    host=config['host'],
                    port=config['port'],
                    metadata={
                        'service_type': config['service_type'],
                        'version': self.version,
                        'started_at': asyncio.get_event_loop().time()
                    }
                )
                
                logger.info(f"✅ Registered service: {service_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to register service {service_name}: {e}")
        
        logger.info("✅ Service registration completed")
    
    async def shutdown(self):
        """Shutdown the ultra-modular system."""
        try:
            logger.info("🛑 Shutting down ultra-modular system...")
            
            # Stop services
            await self._stop_services()
            
            # Stop core systems
            await self._stop_core_systems()
            
            self.status = "stopped"
            logger.info("✅ Ultra-modular system shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown failed: {e}")
            raise
    
    async def _stop_services(self):
        """Stop all microservices."""
        logger.info("🛑 Stopping microservices...")
        
        for service_name, service in self.services.items():
            try:
                # Find service ID
                service_id = None
                for sid, s in self.service_container._services.items():
                    if s == service:
                        service_id = sid
                        break
                
                if service_id:
                    await self.service_container.stop_service(service_id)
                    await self.service_container.unregister_service(service_id)
                
                logger.info(f"✅ Stopped service: {service_name}")
                
            except Exception as e:
                logger.error(f"❌ Error stopping service {service_name}: {e}")
        
        self.services.clear()
        logger.info("✅ Microservices stopped")
    
    async def _stop_core_systems(self):
        """Stop core systems."""
        logger.info("🛑 Stopping core systems...")
        
        # Stop event bus
        await self.event_bus.stop()
        
        # Stop component registry health monitoring
        await self.component_registry.stop_health_monitoring()
        
        # Close API gateway
        await self.api_gateway.close()
        
        logger.info("✅ Core systems stopped")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'system_name': self.system_name,
            'version': self.version,
            'status': self.status,
            'uptime_seconds': asyncio.get_event_loop().time() - self.stats['start_time'] if self.stats['start_time'] else 0,
            'stats': self.stats,
            'component_registry_stats': await self.component_registry.get_registry_stats(),
            'plugin_manager_stats': await self.plugin_manager.get_plugin_manager_stats(),
            'event_bus_stats': await self.event_bus.get_bus_stats(),
            'api_gateway_stats': await self.api_gateway.get_gateway_stats(),
            'service_container_stats': await self.service_container.get_container_stats()
        }


def print_ultra_modular_banner():
    """Print ultra-modular system banner."""
    print("\n" + "="*80)
    print("🏗️ ULTRA-MODULAR AI DOCUMENT PROCESSOR")
    print("="*80)
    print("Microservices • Plugins • Event-Driven • Component Registry")
    print("Version: 4.0.0")
    print("="*80)
    print("🚀 Features:")
    print("   • Ultra-modular architecture")
    print("   • Microservices with independent scaling")
    print("   • Plugin system for extensibility")
    print("   • Event-driven communication")
    print("   • Component registry and discovery")
    print("   • API gateway with load balancing")
    print("   • Health monitoring and metrics")
    print("   • Service mesh architecture")
    print("="*80 + "\n")


def setup_signal_handlers(system: UltraModularSystem):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(system.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main application entry point."""
    try:
        # Print banner
        print_ultra_modular_banner()
        
        # Create system
        system = UltraModularSystem()
        
        # Setup signal handlers
        setup_signal_handlers(system)
        
        # Initialize system
        await system.initialize()
        
        # Print system info
        stats = await system.get_system_stats()
        logger.info(f"📊 System Stats: {stats}")
        
        # Keep system running
        logger.info("🔄 System running... Press Ctrl+C to stop")
        
        # Run forever
        while system.status == "running":
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ System failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

















