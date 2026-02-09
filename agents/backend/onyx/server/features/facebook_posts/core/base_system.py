#!/usr/bin/env python3
"""
Core Base System - Ultra-Modular Architecture v3.7
Base classes and interfaces for all system modules
"""
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModuleState(Enum):
    """Module states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    DISABLED = "disabled"

class ModulePriority(Enum):
    """Module priorities"""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10

@dataclass
class ModuleConfig:
    """Base configuration for modules"""
    name: str
    version: str
    description: str
    priority: ModulePriority = ModulePriority.NORMAL
    enabled: bool = True
    auto_start: bool = False
    dependencies: List[str] = field(default_factory=list)
    config_path: Optional[str] = None
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0

@dataclass
class ModuleStatus:
    """Module status information"""
    state: ModuleState
    last_update: datetime
    error_count: int = 0
    operation_count: int = 0
    performance_score: float = 100.0
    health_status: str = "healthy"
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseModule(ABC):
    """
    Abstract base class for all system modules
    Provides common functionality and interfaces
    """
    
    def __init__(self, config: ModuleConfig):
        """Initialize base module"""
        self.config = config
        self.status = ModuleStatus(
            state=ModuleState.UNINITIALIZED,
            last_update=datetime.now()
        )
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Module lifecycle
        self._initialized = False
        self._running = False
        self._lock = threading.RLock()
        
        # Event callbacks
        self._event_callbacks: Dict[str, List[Callable]] = {}
        self._health_callbacks: List[Callable] = []
        
        # Performance tracking
        self._start_time: Optional[datetime] = None
        self._performance_history: List[Dict] = []
        
        # Initialize module
        self._initialize()
    
    def _initialize(self):
        """Internal initialization"""
        try:
            self.logger.info(f"Initializing {self.config.name} v{self.config.version}")
            self.status.state = ModuleState.INITIALIZING
            
            # Call abstract initialization method
            self.initialize()
            
            self._initialized = True
            self.status.state = ModuleState.READY
            self.status.last_update = datetime.now()
            
            self.logger.info(f"{self.config.name} initialized successfully")
            self._trigger_event("initialized", self.get_status())
            
            # Auto-start if configured
            if self.config.auto_start:
                self.start()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.name}: {e}")
            self.status.state = ModuleState.ERROR
            self.status.health_status = "error"
            self._trigger_event("error", {"error": str(e)})
            raise
    
    @abstractmethod
    def initialize(self):
        """Abstract method for module-specific initialization"""
        pass
    
    def start(self) -> bool:
        """Start the module"""
        with self._lock:
            if not self._initialized:
                self.logger.error(f"Cannot start {self.config.name}: not initialized")
                return False
            
            if self._running:
                self.logger.warning(f"{self.config.name} is already running")
                return True
            
            try:
                self.logger.info(f"Starting {self.config.name}")
                self.status.state = ModuleState.RUNNING
                
                # Call abstract start method
                success = self.start_module()
                
                if success:
                    self._running = True
                    self._start_time = datetime.now()
                    self.status.last_update = datetime.now()
                    self.logger.info(f"{self.config.name} started successfully")
                    self._trigger_event("started", self.get_status())
                    return True
                else:
                    self.status.state = ModuleState.ERROR
                    self.logger.error(f"Failed to start {self.config.name}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error starting {self.config.name}: {e}")
                self.status.state = ModuleState.ERROR
                self.status.error_count += 1
                return False
    
    @abstractmethod
    def start_module(self) -> bool:
        """Abstract method for module-specific start logic"""
        pass
    
    def stop(self) -> bool:
        """Stop the module"""
        with self._lock:
            if not self._running:
                self.logger.warning(f"{self.config.name} is not running")
                return True
            
            try:
                self.logger.info(f"Stopping {self.config.name}")
                
                # Call abstract stop method
                success = self.stop_module()
                
                if success:
                    self._running = False
                    self.status.state = ModuleState.STOPPED
                    self.status.last_update = datetime.now()
                    self.logger.info(f"{self.config.name} stopped successfully")
                    self._trigger_event("stopped", self.get_status())
                    return True
                else:
                    self.logger.error(f"Failed to stop {self.config.name}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error stopping {self.config.name}: {e}")
                self.status.error_count += 1
                return False
    
    @abstractmethod
    def stop_module(self) -> bool:
        """Abstract method for module-specific stop logic"""
        pass
    
    def pause(self) -> bool:
        """Pause the module"""
        with self._lock:
            if not self._running:
                return False
            
            try:
                self.logger.info(f"Pausing {self.config.name}")
                self.status.state = ModuleState.PAUSED
                self.status.last_update = datetime.now()
                
                # Call abstract pause method
                success = self.pause_module()
                
                if success:
                    self._trigger_event("paused", self.get_status())
                    return True
                else:
                    self.status.state = ModuleState.RUNNING
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error pausing {self.config.name}: {e}")
                self.status.state = ModuleState.RUNNING
                return False
    
    def pause_module(self) -> bool:
        """Default pause implementation - can be overridden"""
        return True
    
    def resume(self) -> bool:
        """Resume the module"""
        with self._lock:
            if self.status.state != ModuleState.PAUSED:
                return False
            
            try:
                self.logger.info(f"Resuming {self.config.name}")
                self.status.state = ModuleState.RUNNING
                self.status.last_update = datetime.now()
                
                # Call abstract resume method
                success = self.resume_module()
                
                if success:
                    self._trigger_event("resumed", self.get_status())
                    return True
                else:
                    self.status.state = ModuleState.PAUSED
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error resuming {self.config.name}: {e}")
                self.status.state = ModuleState.PAUSED
                return False
    
    def resume_module(self) -> bool:
        """Default resume implementation - can be overridden"""
        return True
    
    def get_status(self) -> ModuleStatus:
        """Get current module status"""
        return self.status
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information"""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "description": self.config.description,
            "priority": self.config.priority.value,
            "enabled": self.config.enabled,
            "state": self.status.state.value,
            "health_status": self.status.health_status,
            "initialized": self._initialized,
            "running": self._running,
            "error_count": self.status.error_count,
            "operation_count": self.status.operation_count,
            "performance_score": self.status.performance_score,
            "last_update": self.status.last_update.isoformat(),
            "uptime": self._get_uptime(),
            "dependencies": self.config.dependencies
        }
    
    def _get_uptime(self) -> Optional[float]:
        """Get module uptime in seconds"""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return None
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update module configuration"""
        try:
            # Update config attributes
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Apply configuration changes
            self._apply_config_changes(new_config)
            
            self.logger.info(f"Configuration updated for {self.config.name}")
            self._trigger_event("config_updated", new_config)
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise
    
    def _apply_config_changes(self, changes: Dict[str, Any]):
        """Apply configuration changes - can be overridden"""
        pass
    
    def add_event_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event not in self._event_callbacks:
            self._event_callbacks[event] = []
        self._event_callbacks[event].append(callback)
    
    def add_health_callback(self, callback: Callable):
        """Add health callback"""
        self._health_callbacks.append(callback)
    
    def _trigger_event(self, event: str, data: Any):
        """Trigger event callbacks"""
        if event in self._event_callbacks:
            for callback in self._event_callbacks[event]:
                try:
                    callback(event, data, self)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
    
    def _trigger_health_callback(self, health_data: Dict[str, Any]):
        """Trigger health callbacks"""
        for callback in self._health_callbacks:
            try:
                callback(health_data, self)
            except Exception as e:
                self.logger.error(f"Error in health callback: {e}")
    
    def update_health(self, health_data: Dict[str, Any]):
        """Update module health status"""
        try:
            # Update health status
            if 'health_status' in health_data:
                self.status.health_status = health_data['health_status']
            
            if 'performance_score' in health_data:
                self.status.performance_score = health_data['performance_score']
            
            if 'error_count' in health_data:
                self.status.error_count = health_data['error_count']
            
            if 'operation_count' in health_data:
                self.status.operation_count = health_data['operation_count']
            
            self.status.last_update = datetime.now()
            
            # Trigger health callback
            self._trigger_health_callback(health_data)
            
        except Exception as e:
            self.logger.error(f"Error updating health: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics - can be overridden"""
        return {
            "module_name": self.config.name,
            "state": self.status.state.value,
            "performance_score": self.status.performance_score,
            "error_count": self.status.error_count,
            "operation_count": self.status.operation_count,
            "uptime": self._get_uptime(),
            "health_status": self.status.health_status
        }
    
    def export_data(self, format: str = 'json') -> str:
        """Export module data"""
        try:
            data = {
                "module_info": self.get_info(),
                "performance_metrics": self.get_performance_metrics(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            if format.lower() == 'json':
                return json.dumps(data, indent=2, default=str)
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            return f"Error exporting data: {str(e)}"
    
    def cleanup(self):
        """Cleanup module resources"""
        try:
            self.logger.info(f"Cleaning up {self.config.name}")
            
            # Stop if running
            if self._running:
                self.stop()
            
            # Call abstract cleanup method
            self.cleanup_module()
            
            self.logger.info(f"{self.config.name} cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def cleanup_module(self):
        """Abstract method for module-specific cleanup - can be overridden"""
        pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __str__(self):
        return f"{self.config.name} v{self.config.version} ({self.status.state.value})"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.config.name} v{self.config.version}>"

# Utility functions for modules
def create_module_config(name: str, version: str, **kwargs) -> ModuleConfig:
    """Create module configuration with defaults"""
    return ModuleConfig(
        name=name,
        version=version,
        description=kwargs.get('description', f'Module {name}'),
        priority=kwargs.get('priority', ModulePriority.NORMAL),
        enabled=kwargs.get('enabled', True),
        auto_start=kwargs.get('auto_start', False),
        dependencies=kwargs.get('dependencies', []),
        config_path=kwargs.get('config_path'),
        log_level=kwargs.get('log_level', 'INFO'),
        max_retries=kwargs.get('max_retries', 3),
        timeout=kwargs.get('timeout', 30.0)
    )

def validate_module_dependencies(module: BaseModule, available_modules: List[str]) -> bool:
    """Validate module dependencies"""
    missing_deps = [dep for dep in module.config.dependencies if dep not in available_modules]
    if missing_deps:
        module.logger.error(f"Missing dependencies: {missing_deps}")
        return False
    return True
