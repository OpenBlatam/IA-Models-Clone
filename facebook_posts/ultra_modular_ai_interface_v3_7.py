#!/usr/bin/env python3
"""
Ultra-Modular AI Interface v3.7
Completely modular AI system with dynamic loading and plugin support
"""
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import logging

# Import core modules
try:
    from core import (
        BaseModule, ModuleManager, PluginManager, ConfigManager, EventSystem,
        create_module_config, ModulePriority, EventPriority
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core modules not available: {e}")
    CORE_AVAILABLE = False
    BaseModule = None
    ModuleManager = None
    PluginManager = None
    ConfigManager = None
    EventSystem = None

class UltraModularAIInterface:
    """
    Ultra-modular AI interface with dynamic module loading and plugin support
    """
    
    def __init__(self, config_path: str = "config", modules_path: str = "modules", 
                 plugins_path: str = "plugins"):
        """Initialize the ultra-modular AI interface"""
        self.config_path = Path(config_path)
        self.modules_path = Path(modules_path)
        self.plugins_path = Path(plugins_path)
        
        # Core systems
        self.config_manager = None
        self.module_manager = None
        self.plugin_manager = None
        self.event_system = None
        
        # System state
        self.is_initialized = False
        self.system_state = 'initializing'
        self.startup_time = None
        
        # Performance tracking
        self.operation_count = 0
        self.error_count = 0
        
        # Initialize core systems
        self._initialize_core_systems()
    
    def _initialize_core_systems(self):
        """Initialize all core systems"""
        try:
            if not CORE_AVAILABLE:
                print("❌ Core modules not available")
                self.system_state = 'error'
                return
            
            print("🚀 Initializing Ultra-Modular AI Interface v3.7...")
            
            # Create directories
            self.config_path.mkdir(exist_ok=True)
            self.modules_path.mkdir(exist_ok=True)
            self.plugins_path.mkdir(exist_ok=True)
            
            # Initialize configuration manager
            print("⚙️ Initializing Configuration Manager...")
            self.config_manager = ConfigManager(str(self.config_path))
            
            # Initialize event system
            print("📡 Initializing Event System...")
            self.event_system = EventSystem()
            
            # Initialize module manager
            print("🔧 Initializing Module Manager...")
            self.module_manager = ModuleManager(str(self.modules_path))
            
            # Initialize plugin manager
            print("🔌 Initializing Plugin Manager...")
            self.plugin_manager = PluginManager(str(self.plugins_path))
            
            # Set up system integration
            self._setup_system_integration()
            
            # Load default configuration
            self._load_default_config()
            
            # Discover and load modules
            self._discover_modules()
            
            # Discover and load plugins
            self._discover_plugins()
            
            self.is_initialized = True
            self.system_state = 'ready'
            self.startup_time = datetime.now()
            
            print("🎉 Ultra-Modular AI Interface initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing core systems: {e}")
            self.system_state = 'error'
            self.error_count += 1
            raise
    
    def _setup_system_integration(self):
        """Set up integration between core systems"""
        try:
            # Set up event system integration
            if self.event_system and self.module_manager:
                # Subscribe to module events
                self.event_system.subscribe("module_loaded", self._handle_module_loaded)
                self.event_system.subscribe("module_started", self._handle_module_started)
                self.event_system.subscribe("module_stopped", self._handle_module_stopped)
                self.event_system.subscribe("module_error", self._handle_module_error)
            
            # Set up configuration change handling
            if self.config_manager and self.event_system:
                self.config_manager.add_config_watcher(self._handle_config_changed)
            
            print("✅ System integration configured")
            
        except Exception as e:
            print(f"❌ Error setting up system integration: {e}")
    
    def _load_default_config(self):
        """Load or create default configuration"""
        try:
            default_config_file = self.config_path / "system_config.json"
            
            if not default_config_file.exists():
                print("📝 Creating default configuration...")
                self.config_manager.create_default_config(str(default_config_file))
            
            print("✅ Configuration loaded")
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
    
    def _discover_modules(self):
        """Discover and load available modules"""
        try:
            print("🔍 Discovering modules...")
            
            # The module manager will automatically discover modules
            # We can also manually load specific modules here if needed
            
            print("✅ Module discovery completed")
            
        except Exception as e:
            print(f"❌ Error discovering modules: {e}")
    
    def _discover_plugins(self):
        """Discover and load available plugins"""
        try:
            print("🔍 Discovering plugins...")
            
            # The plugin manager will automatically discover plugins
            # We can also manually load specific plugins here if needed
            
            print("✅ Plugin discovery completed")
            
        except Exception as e:
            print(f"❌ Error discovering plugins: {e}")
    
    def _handle_module_loaded(self, event_name: str, data: Any, config: Dict):
        """Handle module loaded event"""
        try:
            print(f"📦 Module loaded: {data.get('name', 'Unknown')}")
            self.event_system.publish("system_module_loaded", data, "system")
            
        except Exception as e:
            print(f"❌ Error handling module loaded event: {e}")
    
    def _handle_module_started(self, event_name: str, data: Any, config: Dict):
        """Handle module started event"""
        try:
            print(f"▶️ Module started: {data.get('name', 'Unknown')}")
            self.event_system.publish("system_module_started", data, "system")
            
        except Exception as e:
            print(f"❌ Error handling module started event: {e}")
    
    def _handle_module_stopped(self, event_name: str, data: Any, config: Dict):
        """Handle module stopped event"""
        try:
            print(f"⏹️ Module stopped: {data.get('name', 'Unknown')}")
            self.event_system.publish("system_module_stopped", data, "system")
            
        except Exception as e:
            print(f"❌ Error handling module stopped event: {e}")
    
    def _handle_module_error(self, event_name: str, data: Any, config: Dict):
        """Handle module error event"""
        try:
            print(f"🚨 Module error: {data.get('name', 'Unknown')} - {data.get('error', 'Unknown error')}")
            self.event_system.publish("system_module_error", data, "system", EventPriority.HIGH)
            
        except Exception as e:
            print(f"❌ Error handling module error event: {e}")
    
    def _handle_config_changed(self, key: str, value: Any, config: Dict):
        """Handle configuration changes"""
        try:
            print(f"⚙️ Configuration changed: {key} = {value}")
            self.event_system.publish("system_config_changed", {"key": key, "value": value}, "system")
            
        except Exception as e:
            print(f"❌ Error handling configuration change: {e}")
    
    def start_system(self) -> bool:
        """Start the ultra-modular system"""
        try:
            if not self.is_initialized:
                print("❌ System not initialized")
                return False
            
            print("🚀 Starting Ultra-Modular AI System...")
            
            # Start all modules
            if self.module_manager:
                start_results = self.module_manager.start_all_modules()
                print(f"📊 Module startup results: {start_results}")
            
            # Start plugin watcher
            if self.plugin_manager:
                self.plugin_manager.start_plugin_watcher()
            
            # Start configuration watcher
            if self.config_manager:
                self.config_manager.start_file_watcher()
            
            self.system_state = 'running'
            print("✅ Ultra-Modular AI System started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error starting system: {e}")
            self.error_count += 1
            return False
    
    def stop_system(self):
        """Stop the ultra-modular system"""
        try:
            print("🛑 Stopping Ultra-Modular AI System...")
            
            # Stop all modules
            if self.module_manager:
                stop_results = self.module_manager.stop_all_modules()
                print(f"📊 Module shutdown results: {stop_results}")
            
            # Stop plugin watcher
            if self.plugin_manager:
                self.plugin_manager.stop_plugin_watcher()
            
            # Stop configuration watcher
            if self.config_manager:
                self.config_manager.stop_file_watcher()
            
            self.system_state = 'stopped'
            print("✅ Ultra-Modular AI System stopped successfully!")
            
        except Exception as e:
            print(f"❌ Error stopping system: {e}")
            self.error_count += 1
    
    def load_module(self, module_name: str, config: Optional[Dict] = None) -> bool:
        """Load a specific module"""
        try:
            if not self.module_manager:
                print("❌ Module manager not available")
                return False
            
            print(f"📦 Loading module: {module_name}")
            
            # Create module configuration
            if config is None:
                config = create_module_config(
                    name=module_name,
                    version="1.0.0",
                    description=f"Module {module_name}",
                    auto_start=False
                )
            
            # Load module
            module = self.module_manager.load_module(module_name, config)
            
            if module:
                print(f"✅ Module loaded: {module_name}")
                return True
            else:
                print(f"❌ Failed to load module: {module_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading module {module_name}: {e}")
            self.error_count += 1
            return False
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict] = None) -> bool:
        """Load a specific plugin"""
        try:
            if not self.plugin_manager:
                print("❌ Plugin manager not available")
                return False
            
            print(f"🔌 Loading plugin: {plugin_name}")
            
            # Load plugin
            plugin = self.plugin_manager.load_plugin(plugin_name, config)
            
            if plugin:
                print(f"✅ Plugin loaded: {plugin_name}")
                return True
            else:
                print(f"❌ Failed to load plugin: {plugin_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading plugin {plugin_name}: {e}")
            self.error_count += 1
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'interface': {
                    'version': '3.7.0',
                    'state': self.system_state,
                    'initialized': self.is_initialized,
                    'startup_time': self.startup_time.isoformat() if self.startup_time else None,
                    'operation_count': self.operation_count,
                    'error_count': self.error_count
                }
            }
            
            # Add core system status
            if self.config_manager:
                status['configuration'] = self.config_manager.get_config_info()
            
            if self.module_manager:
                status['modules'] = self.module_manager.get_system_status()
            
            if self.plugin_manager:
                status['plugins'] = self.plugin_manager.get_system_status()
            
            if self.event_system:
                status['events'] = self.event_system.get_event_statistics()
            
            return status
            
        except Exception as e:
            return {'error': f'Error getting system status: {str(e)}'}
    
    def get_module_status(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific module"""
        try:
            if self.module_manager:
                return self.module_manager.get_module_status(module_name)
            return None
            
        except Exception as e:
            print(f"❌ Error getting module status: {e}")
            return None
    
    def get_plugin_status(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific plugin"""
        try:
            if self.plugin_manager:
                return self.plugin_manager.get_plugin_status(plugin_name)
            return None
            
        except Exception as e:
            print(f"❌ Error getting plugin status: {e}")
            return None
    
    def publish_event(self, event_name: str, data: Any = None, priority: EventPriority = EventPriority.NORMAL):
        """Publish a system event"""
        try:
            if self.event_system:
                event_id = self.event_system.publish(event_name, data, "system", priority)
                print(f"📡 Event published: {event_name} (ID: {event_id})")
                return event_id
            else:
                print("❌ Event system not available")
                return None
                
        except Exception as e:
            print(f"❌ Error publishing event: {e}")
            return None
    
    def subscribe_to_event(self, event_name: str, callback: Callable, priority: EventPriority = EventPriority.NORMAL):
        """Subscribe to system events"""
        try:
            if self.event_system:
                handler_id = self.event_system.subscribe(event_name, callback, priority)
                print(f"📡 Subscribed to event: {event_name} (Handler ID: {handler_id})")
                return handler_id
            else:
                print("❌ Event system not available")
                return None
                
        except Exception as e:
            print(f"❌ Error subscribing to event: {e}")
            return None
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            if self.config_manager:
                return self.config_manager.get_config(key, default)
            return default
            
        except Exception as e:
            print(f"❌ Error getting configuration: {e}")
            return default
    
    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        try:
            if self.config_manager:
                self.config_manager.set_config(key, value)
                print(f"⚙️ Configuration updated: {key} = {value}")
            else:
                print("❌ Configuration manager not available")
                
        except Exception as e:
            print(f"❌ Error setting configuration: {e}")
    
    def reload_config(self):
        """Reload configuration"""
        try:
            if self.config_manager:
                self.config_manager.reload_config()
                print("🔄 Configuration reloaded")
            else:
                print("❌ Configuration manager not available")
                
        except Exception as e:
            print(f"❌ Error reloading configuration: {e}")
    
    def install_plugin(self, plugin_path: str) -> bool:
        """Install a new plugin"""
        try:
            if not self.plugin_manager:
                print("❌ Plugin manager not available")
                return False
            
            print(f"🔌 Installing plugin: {plugin_path}")
            
            success = self.plugin_manager.install_plugin(plugin_path)
            
            if success:
                print(f"✅ Plugin installed: {plugin_path}")
            else:
                print(f"❌ Failed to install plugin: {plugin_path}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error installing plugin: {e}")
            self.error_count += 1
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            metrics = {
                'system': {
                    'state': self.system_state,
                    'uptime': (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
                    'operation_count': self.operation_count,
                    'error_count': self.error_count
                }
            }
            
            # Add core system metrics
            if self.event_system:
                metrics['events'] = self.event_system.get_event_statistics()
            
            if self.module_manager:
                module_status = self.module_manager.get_system_status()
                metrics['modules'] = {
                    'total': module_status.get('module_manager', {}).get('total_modules', 0),
                    'running': module_status.get('module_manager', {}).get('running_modules', 0),
                    'error': module_status.get('module_manager', {}).get('error_modules', 0)
                }
            
            return metrics
            
        except Exception as e:
            return {'error': f'Error getting performance metrics: {str(e)}'}
    
    def export_system_data(self, format: str = 'json') -> str:
        """Export comprehensive system data"""
        try:
            data = {
                'system_info': {
                    'version': '3.7.0',
                    'state': self.system_state,
                    'initialized': self.is_initialized,
                    'startup_time': self.startup_time.isoformat() if self.startup_time else None
                },
                'status': self.get_system_status(),
                'performance': self.get_performance_metrics(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format.lower() == 'json':
                return json.dumps(data, indent=2, default=str)
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            return f"Error exporting system data: {str(e)}"
    
    def cleanup(self):
        """Cleanup system resources"""
        try:
            print("🧹 Cleaning up Ultra-Modular AI System...")
            
            # Stop system if running
            if self.system_state == 'running':
                self.stop_system()
            
            # Shutdown core systems
            if self.config_manager:
                self.config_manager.shutdown()
            
            if self.module_manager:
                self.module_manager.shutdown()
            
            if self.plugin_manager:
                self.plugin_manager.shutdown()
            
            if self.event_system:
                self.event_system.shutdown()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# Example usage and testing
if __name__ == "__main__":
    try:
        # Create ultra-modular interface
        ai_interface = UltraModularAIInterface()
        
        # Get initial status
        status = ai_interface.get_system_status()
        print(f"Initial status: {status['interface']['state']}")
        
        # Start the system
        if ai_interface.start_system():
            print("🎉 System started successfully!")
            
            # Run for a while to see the system in action
            print("🔄 Running system for 30 seconds...")
            time.sleep(30)
            
            # Get performance metrics
            metrics = ai_interface.get_performance_metrics()
            print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
            
            # Export system data
            export_data = ai_interface.export_system_data('json')
            print(f"System data exported: {len(export_data)} characters")
            
        else:
            print("❌ Failed to start system")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        if 'ai_interface' in locals():
            ai_interface.cleanup()
