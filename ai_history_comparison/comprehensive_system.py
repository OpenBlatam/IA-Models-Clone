"""
Comprehensive AI History Analysis System
=======================================

This module provides a comprehensive integration system that connects all
components of the AI history analysis and model comparison system.

Features:
- Unified system initialization and management
- Component orchestration and coordination
- Real-time data flow between components
- System health monitoring and diagnostics
- Performance optimization and scaling
- Enterprise-grade reliability and fault tolerance
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os
from contextlib import asynccontextmanager

# Import all system components
from .ai_history_analyzer import get_ai_history_analyzer, AIHistoryAnalyzer
from .config import get_ai_history_config, AIHistoryConfig
from .api_endpoints import get_api_app, FastAPI
from .integration_system import get_integration_system, AIHistoryIntegrationSystem
from .realtime_dashboard import get_realtime_dashboard, RealtimeDashboard
from .ml_predictor import get_ml_predictor, MLPredictor
from .intelligent_alerts import get_intelligent_alert_system, IntelligentAlertSystem

logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """System status information"""
    timestamp: datetime
    overall_status: str  # "healthy", "degraded", "unhealthy"
    components: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, float]
    active_alerts: int
    system_uptime: float
    last_health_check: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SystemConfiguration:
    """System configuration"""
    enable_api: bool = True
    enable_dashboard: bool = True
    enable_ml_predictor: bool = True
    enable_alerts: bool = True
    enable_integration: bool = True
    api_port: int = 8002
    dashboard_port: int = 8003
    ml_model_storage_path: str = "ml_models"
    alert_cooldown_minutes: int = 30
    health_check_interval: int = 60
    performance_monitoring: bool = True
    auto_scaling: bool = False
    backup_enabled: bool = True
    log_level: str = "INFO"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ComprehensiveAISystem:
    """Comprehensive AI History Analysis System"""
    
    def __init__(self, config: SystemConfiguration = None):
        self.config = config or SystemConfiguration()
        
        # Core components
        self.analyzer: Optional[AIHistoryAnalyzer] = None
        self.config_manager: Optional[AIHistoryConfig] = None
        self.api_app: Optional[FastAPI] = None
        self.dashboard: Optional[RealtimeDashboard] = None
        self.ml_predictor: Optional[MLPredictor] = None
        self.alert_system: Optional[IntelligentAlertSystem] = None
        self.integration_system: Optional[AIHistoryIntegrationSystem] = None
        
        # System management
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.performance_monitor_task: Optional[asyncio.Task] = None
        
        # System status
        self.system_status: Optional[SystemStatus] = None
        self.component_status: Dict[str, Dict[str, Any]] = {}
        
        # Event handlers
        self.startup_handlers: List[Callable] = []
        self.shutdown_handlers: List[Callable] = []
        self.health_check_handlers: List[Callable] = []
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.error_count = 0
        self.request_count = 0
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self):
        """Setup system logging"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('ai_system.log')
            ]
        )
        logger.info("Comprehensive AI System logging initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Comprehensive AI System...")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize optional components
            await self._initialize_optional_components()
            
            # Setup component integration
            await self._setup_component_integration()
            
            # Run startup handlers
            await self._run_startup_handlers()
            
            logger.info("Comprehensive AI System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            raise
    
    async def _initialize_core_components(self):
        """Initialize core system components"""
        try:
            # Initialize configuration manager
            self.config_manager = get_ai_history_config()
            logger.info("Configuration manager initialized")
            
            # Initialize analyzer
            self.analyzer = get_ai_history_analyzer()
            logger.info("AI History Analyzer initialized")
            
            # Initialize integration system
            if self.config.enable_integration:
                self.integration_system = get_integration_system()
                logger.info("Integration system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing core components: {str(e)}")
            raise
    
    async def _initialize_optional_components(self):
        """Initialize optional system components"""
        try:
            # Initialize API
            if self.config.enable_api:
                self.api_app = get_api_app()
                logger.info("API application initialized")
            
            # Initialize dashboard
            if self.config.enable_dashboard:
                self.dashboard = get_realtime_dashboard()
                await self.dashboard.initialize()
                logger.info("Real-time dashboard initialized")
            
            # Initialize ML predictor
            if self.config.enable_ml_predictor:
                self.ml_predictor = get_ml_predictor(self.config.ml_model_storage_path)
                logger.info("ML predictor initialized")
            
            # Initialize alert system
            if self.config.enable_alerts:
                self.alert_system = get_intelligent_alert_system()
                logger.info("Intelligent alert system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing optional components: {str(e)}")
            raise
    
    async def _setup_component_integration(self):
        """Setup integration between components"""
        try:
            # Connect analyzer with integration system
            if self.integration_system and self.analyzer:
                self.integration_system.analyzer = self.analyzer
                logger.info("Analyzer integrated with integration system")
            
            # Connect ML predictor with analyzer
            if self.ml_predictor and self.analyzer:
                self.ml_predictor.analyzer = self.analyzer
                logger.info("ML predictor integrated with analyzer")
            
            # Connect alert system with analyzer and ML predictor
            if self.alert_system:
                if self.analyzer:
                    self.alert_system.analyzer = self.analyzer
                if self.ml_predictor:
                    self.alert_system.ml_predictor = self.ml_predictor
                logger.info("Alert system integrated with analyzer and ML predictor")
            
            # Connect dashboard with all components
            if self.dashboard:
                if self.analyzer:
                    self.dashboard.analyzer = self.analyzer
                if self.config_manager:
                    self.dashboard.config = self.config_manager
                if self.integration_system:
                    self.dashboard.integration_system = self.integration_system
                logger.info("Dashboard integrated with all components")
            
        except Exception as e:
            logger.error(f"Error setting up component integration: {str(e)}")
            raise
    
    async def _run_startup_handlers(self):
        """Run startup event handlers"""
        try:
            for handler in self.startup_handlers:
                await handler()
        except Exception as e:
            logger.error(f"Error running startup handlers: {str(e)}")
    
    async def start(self):
        """Start the comprehensive system"""
        try:
            if self.is_running:
                logger.warning("System is already running")
                return
            
            logger.info("Starting Comprehensive AI System...")
            
            # Initialize system
            await self.initialize()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Start optional services
            await self._start_optional_services()
            
            # Mark system as running
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("Comprehensive AI System started successfully")
            
            # Keep system running
            await self._keep_alive()
            
        except Exception as e:
            logger.error(f"Error starting system: {str(e)}")
            await self.shutdown()
            raise
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Start health check task
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Start performance monitoring task
            if self.config.performance_monitoring:
                self.performance_monitor_task = asyncio.create_task(self._performance_monitor_loop())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {str(e)}")
            raise
    
    async def _start_optional_services(self):
        """Start optional services"""
        try:
            # Start alert system monitoring
            if self.alert_system:
                await self.alert_system.start_monitoring()
                logger.info("Alert system monitoring started")
            
            # Start dashboard (if enabled)
            if self.dashboard:
                # Dashboard will be started separately via run_dashboard()
                logger.info("Dashboard ready to start")
            
        except Exception as e:
            logger.error(f"Error starting optional services: {str(e)}")
            raise
    
    async def _keep_alive(self):
        """Keep the system running"""
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("System keep-alive cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {str(e)}")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        try:
            while self.is_running:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
        except Exception as e:
            logger.error(f"Error in health check loop: {str(e)}")
    
    async def _performance_monitor_loop(self):
        """Background performance monitoring loop"""
        try:
            while self.is_running:
                await self._collect_performance_metrics()
                await asyncio.sleep(60)  # Collect every minute
        except asyncio.CancelledError:
            logger.info("Performance monitor loop cancelled")
        except Exception as e:
            logger.error(f"Error in performance monitor loop: {str(e)}")
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            component_status = {}
            overall_healthy = True
            
            # Check analyzer
            if self.analyzer:
                analyzer_healthy = await self._check_analyzer_health()
                component_status["analyzer"] = {
                    "status": "healthy" if analyzer_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat()
                }
                if not analyzer_healthy:
                    overall_healthy = False
            
            # Check ML predictor
            if self.ml_predictor:
                ml_healthy = await self._check_ml_predictor_health()
                component_status["ml_predictor"] = {
                    "status": "healthy" if ml_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat()
                }
                if not ml_healthy:
                    overall_healthy = False
            
            # Check alert system
            if self.alert_system:
                alert_healthy = await self._check_alert_system_health()
                component_status["alert_system"] = {
                    "status": "healthy" if alert_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat()
                }
                if not alert_healthy:
                    overall_healthy = False
            
            # Check integration system
            if self.integration_system:
                integration_healthy = await self._check_integration_system_health()
                component_status["integration_system"] = {
                    "status": "healthy" if integration_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat()
                }
                if not integration_healthy:
                    overall_healthy = False
            
            # Update system status
            self.component_status = component_status
            self.system_status = SystemStatus(
                timestamp=datetime.now(),
                overall_status="healthy" if overall_healthy else "degraded",
                components=component_status,
                performance_metrics=self._get_performance_metrics(),
                active_alerts=len(self.alert_system.get_active_alerts()) if self.alert_system else 0,
                system_uptime=(datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                last_health_check=datetime.now()
            )
            
            # Run health check handlers
            for handler in self.health_check_handlers:
                try:
                    await handler(self.system_status)
                except Exception as e:
                    logger.error(f"Error in health check handler: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
    
    async def _check_analyzer_health(self) -> bool:
        """Check analyzer health"""
        try:
            if not self.analyzer:
                return False
            
            # Check if analyzer can access performance stats
            stats = self.analyzer.performance_stats
            return stats is not None
            
        except Exception as e:
            logger.error(f"Error checking analyzer health: {str(e)}")
            return False
    
    async def _check_ml_predictor_health(self) -> bool:
        """Check ML predictor health"""
        try:
            if not self.ml_predictor:
                return False
            
            # Check if ML predictor has access to analyzer
            return self.ml_predictor.analyzer is not None
            
        except Exception as e:
            logger.error(f"Error checking ML predictor health: {str(e)}")
            return False
    
    async def _check_alert_system_health(self) -> bool:
        """Check alert system health"""
        try:
            if not self.alert_system:
                return False
            
            # Check if alert system is monitoring
            return self.alert_system.is_monitoring
            
        except Exception as e:
            logger.error(f"Error checking alert system health: {str(e)}")
            return False
    
    async def _check_integration_system_health(self) -> bool:
        """Check integration system health"""
        try:
            if not self.integration_system:
                return False
            
            # Check if integration system has analyzer
            return self.integration_system.analyzer is not None
            
        except Exception as e:
            logger.error(f"Error checking integration system health: {str(e)}")
            return False
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        try:
            metrics = {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
            
            # Add analyzer metrics if available
            if self.analyzer:
                stats = self.analyzer.performance_stats
                metrics.update({
                    "models_tracked": len(stats.get("models_tracked", [])),
                    "total_measurements": stats.get("total_measurements", 0),
                    "trend_analyses": len(self.analyzer.trend_analyses)
                })
            
            # Add alert system metrics if available
            if self.alert_system:
                alert_stats = self.alert_system.get_alert_statistics()
                metrics.update({
                    "active_alerts": alert_stats.get("active_alerts", 0),
                    "total_alerts": alert_stats.get("total_alerts", 0)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    async def _collect_performance_metrics(self):
        """Collect and store performance metrics"""
        try:
            metrics = self._get_performance_metrics()
            metrics["timestamp"] = datetime.now().isoformat()
            
            self.performance_history.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")
    
    async def shutdown(self):
        """Shutdown the comprehensive system"""
        try:
            if not self.is_running:
                logger.warning("System is not running")
                return
            
            logger.info("Shutting down Comprehensive AI System...")
            
            # Mark system as not running
            self.is_running = False
            
            # Stop background tasks
            if self.health_check_task:
                self.health_check_task.cancel()
            if self.performance_monitor_task:
                self.performance_monitor_task.cancel()
            
            # Stop alert system
            if self.alert_system:
                await self.alert_system.stop_monitoring()
            
            # Stop dashboard
            if self.dashboard:
                await self.dashboard.shutdown()
            
            # Run shutdown handlers
            await self._run_shutdown_handlers()
            
            logger.info("Comprehensive AI System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    async def _run_shutdown_handlers(self):
        """Run shutdown event handlers"""
        try:
            for handler in self.shutdown_handlers:
                await handler()
        except Exception as e:
            logger.error(f"Error running shutdown handlers: {str(e)}")
    
    def add_startup_handler(self, handler: Callable):
        """Add startup event handler"""
        self.startup_handlers.append(handler)
    
    def add_shutdown_handler(self, handler: Callable):
        """Add shutdown event handler"""
        self.shutdown_handlers.append(handler)
    
    def add_health_check_handler(self, handler: Callable):
        """Add health check event handler"""
        self.health_check_handlers.append(handler)
    
    def get_system_status(self) -> Optional[SystemStatus]:
        """Get current system status"""
        return self.system_status
    
    def get_performance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance history"""
        return self.performance_history[-limit:]
    
    async def run_api(self, host: str = "0.0.0.0", port: int = None):
        """Run the API server"""
        if not self.api_app:
            raise RuntimeError("API is not enabled")
        
        port = port or self.config.api_port
        
        import uvicorn
        logger.info(f"Starting API server on {host}:{port}")
        await uvicorn.run(self.api_app, host=host, port=port)
    
    async def run_dashboard(self, host: str = "0.0.0.0", port: int = None):
        """Run the dashboard server"""
        if not self.dashboard:
            raise RuntimeError("Dashboard is not enabled")
        
        port = port or self.config.dashboard_port
        
        logger.info(f"Starting dashboard server on {host}:{port}")
        self.dashboard.run(host=host, port=port)


# Global system instance
_system: Optional[ComprehensiveAISystem] = None


def get_comprehensive_system(config: SystemConfiguration = None) -> ComprehensiveAISystem:
    """Get or create global comprehensive system"""
    global _system
    if _system is None:
        _system = ComprehensiveAISystem(config)
    return _system


# Example usage and CLI interface
async def main():
    """Main entry point for the comprehensive system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive AI History Analysis System")
    parser.add_argument("--mode", choices=["api", "dashboard", "full"], default="full",
                       help="System mode to run")
    parser.add_argument("--api-port", type=int, default=8002,
                       help="API server port")
    parser.add_argument("--dashboard-port", type=int, default=8003,
                       help="Dashboard server port")
    parser.add_argument("--config", type=str,
                       help="Configuration file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Load configuration
    config = SystemConfiguration(
        enable_api=args.mode in ["api", "full"],
        enable_dashboard=args.mode in ["dashboard", "full"],
        api_port=args.api_port,
        dashboard_port=args.dashboard_port,
        log_level=args.log_level
    )
    
    if args.config:
        # Load configuration from file
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create and start system
    system = get_comprehensive_system(config)
    
    try:
        if args.mode == "full":
            # Start system in background
            asyncio.create_task(system.start())
            
            # Start API and dashboard concurrently
            await asyncio.gather(
                system.run_api(),
                system.run_dashboard()
            )
        elif args.mode == "api":
            await system.start()
            await system.run_api()
        elif args.mode == "dashboard":
            await system.start()
            await system.run_dashboard()
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

























