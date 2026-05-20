"""
Modular Configuration Loader
Loads and applies the modular production configuration using the new modular architecture
"""

import yaml
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add the TruthGPT path to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent / "Frontier-Model-run" / "scripts" / "TruthGPT-main"))

from core.modules import (
    ModuleManager, DependencyInjector, ConfigManager, 
    PluginManager, RegistryManager, FactoryRegistry
)
from core.modules.base import ModuleState
from core.modules.registry import ComponentType
from core.modules.injection import Scope

logger = logging.getLogger(__name__)

class ModularConfigLoader:
    """Loader for modular configuration"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self.module_manager = ModuleManager()
        self.injector = DependencyInjector()
        self.config_manager = ConfigManager()
        self.plugin_manager = PluginManager()
        self.registry_manager = RegistryManager()
        self.factory_registry = FactoryRegistry()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_path}")
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_modular_system(self) -> bool:
        """Setup the modular system based on configuration"""
        try:
            modular_config = self.config.get('modular_system', {})
            
            # Setup module manager
            if 'module_manager' in modular_config:
                self._setup_module_manager(modular_config['module_manager'])
            
            # Setup plugin system
            if 'plugin_system' in modular_config:
                self._setup_plugin_system(modular_config['plugin_system'])
            
            # Setup dependency injection
            if 'dependency_injection' in modular_config:
                self._setup_dependency_injection(modular_config['dependency_injection'])
            
            # Setup configuration management
            if 'configuration' in modular_config:
                self._setup_configuration_management(modular_config['configuration'])
            
            logger.info("✅ Modular system setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup modular system: {e}")
            return False
    
    def _setup_module_manager(self, config: Dict[str, Any]) -> None:
        """Setup module manager"""
        if config.get('auto_discover_modules', False):
            # Auto-discover modules in directories
            directories = config.get('module_directories', [])
            for directory in directories:
                if os.path.exists(directory):
                    self.module_manager.add_module_directory(directory)
        
        logger.info("Module manager configured")
    
    def _setup_plugin_system(self, config: Dict[str, Any]) -> None:
        """Setup plugin system"""
        if config.get('enabled', False):
            # Add plugin directories
            directories = config.get('plugin_directories', [])
            for directory in directories:
                self.plugin_manager.add_plugin_directory(directory)
            
            # Auto-discover plugins
            if config.get('auto_discover', False):
                discovered = self.plugin_manager.discover_plugins()
                logger.info(f"Discovered plugins: {discovered}")
                
                # Load plugins
                results = self.plugin_manager.discover_and_load_plugins()
                logger.info(f"Plugin load results: {results}")
        
        logger.info("Plugin system configured")
    
    def _setup_dependency_injection(self, config: Dict[str, Any]) -> None:
        """Setup dependency injection"""
        if config.get('enabled', False):
            # Register core services
            self._register_core_services()
            
            # Register micro-modules
            self._register_micro_modules()
            
            # Register plugins
            self._register_plugins()
        
        logger.info("Dependency injection configured")
    
    def _setup_configuration_management(self, config: Dict[str, Any]) -> None:
        """Setup configuration management"""
        # Add configuration sources
        sources = config.get('sources', [])
        for source in sources:
            from core.modules.config import ConfigSource, ConfigFormat
            
            format_map = {
                'yaml': ConfigFormat.YAML,
                'json': ConfigFormat.JSON,
                'env': ConfigFormat.ENV
            }
            
            source_format = format_map.get(source['type'], ConfigFormat.YAML)
            priority = source.get('priority', 0)
            
            config_source = ConfigSource(
                name=source.get('name', 'default'),
                path=source['path'],
                format=source_format,
                priority=priority
            )
            
            self.config_manager.add_source(config_source)
        
        # Load configurations
        if config.get('enable_hot_reload', False):
            # Setup hot reload
            pass
        
        # Validate configurations
        if config.get('validation', False):
            self._validate_configurations()
        
        logger.info("Configuration management configured")
    
    def _register_core_services(self) -> None:
        """Register core services"""
        # Register system services
        self.injector.register_singleton(
            type('SystemConfig', (), {}),
            instance=self.config
        )
        
        # Register managers
        self.injector.register_singleton(
            type('ModuleManager', (), {}),
            instance=self.module_manager
        )
        
        self.injector.register_singleton(
            type('PluginManager', (), {}),
            instance=self.plugin_manager
        )
        
        self.injector.register_singleton(
            type('ConfigManager', (), {}),
            instance=self.config_manager
        )
        
        logger.info("Core services registered")
    
    def _register_micro_modules(self) -> None:
        """Register micro-modules"""
        micro_modules = self.config.get('micro_modules', {})
        
        # Register optimizers
        optimizers = micro_modules.get('optimizers', [])
        for optimizer_config in optimizers:
            self._register_optimizer(optimizer_config)
        
        # Register models
        models = micro_modules.get('models', [])
        for model_config in models:
            self._register_model(model_config)
        
        # Register trainers
        trainers = micro_modules.get('trainers', [])
        for trainer_config in trainers:
            self._register_trainer(trainer_config)
        
        # Register inferencers
        inferencers = micro_modules.get('inferencers', [])
        for inferencer_config in inferencers:
            self._register_inferencer(inferencer_config)
        
        # Register monitors
        monitors = micro_modules.get('monitors', [])
        for monitor_config in monitors:
            self._register_monitor(monitor_config)
        
        # Register benchmarkers
        benchmarkers = micro_modules.get('benchmarkers', [])
        for benchmarker_config in benchmarkers:
            self._register_benchmarker(benchmarker_config)
        
        logger.info("Micro-modules registered")
    
    def _register_optimizer(self, config: Dict[str, Any]) -> None:
        """Register optimizer"""
        name = config['name']
        class_name = config['class']
        optimizer_config = config.get('config', {})
        scope = config.get('scope', 'singleton')
        
        # Create optimizer class dynamically
        optimizer_class = self._create_dynamic_class(class_name, 'Optimizer')
        
        # Register with dependency injection
        if scope == 'singleton':
            self.injector.register_singleton(
                type(f'IOptimizer_{name}', (), {}),
                implementation_type=optimizer_class
            )
        elif scope == 'transient':
            self.injector.register_transient(
                type(f'IOptimizer_{name}', (), {}),
                implementation_type=optimizer_class
            )
        elif scope == 'scoped':
            self.injector.register_scoped(
                type(f'IOptimizer_{name}', (), {}),
                implementation_type=optimizer_class
            )
        
        # Register with module manager
        self.module_manager.register_module_class(name, optimizer_class)
        
        logger.info(f"Registered optimizer: {name}")
    
    def _register_model(self, config: Dict[str, Any]) -> None:
        """Register model"""
        name = config['name']
        class_name = config['class']
        model_config = config.get('config', {})
        scope = config.get('scope', 'singleton')
        
        # Create model class dynamically
        model_class = self._create_dynamic_class(class_name, 'Model')
        
        # Register with dependency injection
        if scope == 'singleton':
            self.injector.register_singleton(
                type(f'IModel_{name}', (), {}),
                implementation_type=model_class
            )
        elif scope == 'transient':
            self.injector.register_transient(
                type(f'IModel_{name}', (), {}),
                implementation_type=model_class
            )
        elif scope == 'scoped':
            self.injector.register_scoped(
                type(f'IModel_{name}', (), {}),
                implementation_type=model_class
            )
        
        # Register with module manager
        self.module_manager.register_module_class(name, model_class)
        
        logger.info(f"Registered model: {name}")
    
    def _register_trainer(self, config: Dict[str, Any]) -> None:
        """Register trainer"""
        name = config['name']
        class_name = config['class']
        trainer_config = config.get('config', {})
        scope = config.get('scope', 'transient')
        
        # Create trainer class dynamically
        trainer_class = self._create_dynamic_class(class_name, 'Trainer')
        
        # Register with dependency injection
        if scope == 'singleton':
            self.injector.register_singleton(
                type(f'ITrainer_{name}', (), {}),
                implementation_type=trainer_class
            )
        elif scope == 'transient':
            self.injector.register_transient(
                type(f'ITrainer_{name}', (), {}),
                implementation_type=trainer_class
            )
        elif scope == 'scoped':
            self.injector.register_scoped(
                type(f'ITrainer_{name}', (), {}),
                implementation_type=trainer_class
            )
        
        # Register with module manager
        self.module_manager.register_module_class(name, trainer_class)
        
        logger.info(f"Registered trainer: {name}")
    
    def _register_inferencer(self, config: Dict[str, Any]) -> None:
        """Register inferencer"""
        name = config['name']
        class_name = config['class']
        inferencer_config = config.get('config', {})
        scope = config.get('scope', 'singleton')
        
        # Create inferencer class dynamically
        inferencer_class = self._create_dynamic_class(class_name, 'Inferencer')
        
        # Register with dependency injection
        if scope == 'singleton':
            self.injector.register_singleton(
                type(f'IInferencer_{name}', (), {}),
                implementation_type=inferencer_class
            )
        elif scope == 'transient':
            self.injector.register_transient(
                type(f'IInferencer_{name}', (), {}),
                implementation_type=inferencer_class
            )
        elif scope == 'scoped':
            self.injector.register_scoped(
                type(f'IInferencer_{name}', (), {}),
                implementation_type=inferencer_class
            )
        
        # Register with module manager
        self.module_manager.register_module_class(name, inferencer_class)
        
        logger.info(f"Registered inferencer: {name}")
    
    def _register_monitor(self, config: Dict[str, Any]) -> None:
        """Register monitor"""
        name = config['name']
        class_name = config['class']
        monitor_config = config.get('config', {})
        scope = config.get('scope', 'singleton')
        
        # Create monitor class dynamically
        monitor_class = self._create_dynamic_class(class_name, 'Monitor')
        
        # Register with dependency injection
        if scope == 'singleton':
            self.injector.register_singleton(
                type(f'IMonitor_{name}', (), {}),
                implementation_type=monitor_class
            )
        elif scope == 'transient':
            self.injector.register_transient(
                type(f'IMonitor_{name}', (), {}),
                implementation_type=monitor_class
            )
        elif scope == 'scoped':
            self.injector.register_scoped(
                type(f'IMonitor_{name}', (), {}),
                implementation_type=monitor_class
            )
        
        # Register with module manager
        self.module_manager.register_module_class(name, monitor_class)
        
        logger.info(f"Registered monitor: {name}")
    
    def _register_benchmarker(self, config: Dict[str, Any]) -> None:
        """Register benchmarker"""
        name = config['name']
        class_name = config['class']
        benchmarker_config = config.get('config', {})
        scope = config.get('scope', 'singleton')
        
        # Create benchmarker class dynamically
        benchmarker_class = self._create_dynamic_class(class_name, 'Benchmarker')
        
        # Register with dependency injection
        if scope == 'singleton':
            self.injector.register_singleton(
                type(f'IBenchmarker_{name}', (), {}),
                implementation_type=benchmarker_class
            )
        elif scope == 'transient':
            self.injector.register_transient(
                type(f'IBenchmarker_{name}', (), {}),
                implementation_type=benchmarker_class
            )
        elif scope == 'scoped':
            self.injector.register_scoped(
                type(f'IBenchmarker_{name}', (), {}),
                implementation_type=benchmarker_class
            )
        
        # Register with module manager
        self.module_manager.register_module_class(name, benchmarker_class)
        
        logger.info(f"Registered benchmarker: {name}")
    
    def _register_plugins(self) -> None:
        """Register plugins"""
        plugins = self.config.get('plugins', {})
        
        # Register optimization plugins
        optimization_plugins = plugins.get('optimization_plugins', [])
        for plugin_config in optimization_plugins:
            self._register_plugin(plugin_config, 'optimization')
        
        # Register model plugins
        model_plugins = plugins.get('model_plugins', [])
        for plugin_config in model_plugins:
            self._register_plugin(plugin_config, 'model')
        
        # Register training plugins
        training_plugins = plugins.get('training_plugins', [])
        for plugin_config in training_plugins:
            self._register_plugin(plugin_config, 'training')
        
        # Register inference plugins
        inference_plugins = plugins.get('inference_plugins', [])
        for plugin_config in inference_plugins:
            self._register_plugin(plugin_config, 'inference')
        
        # Register monitoring plugins
        monitoring_plugins = plugins.get('monitoring_plugins', [])
        for plugin_config in monitoring_plugins:
            self._register_plugin(plugin_config, 'monitoring')
        
        logger.info("Plugins registered")
    
    def _register_plugin(self, config: Dict[str, Any], plugin_type: str) -> None:
        """Register a plugin"""
        name = config['name']
        class_name = config['class']
        plugin_config = config.get('config', {})
        enabled = config.get('enabled', True)
        
        if not enabled:
            logger.info(f"Plugin {name} is disabled, skipping")
            return
        
        # Create plugin class dynamically
        plugin_class = self._create_dynamic_class(class_name, 'Plugin')
        
        # Register with plugin manager
        self.plugin_manager.load_plugin(name, plugin_config)
        
        logger.info(f"Registered {plugin_type} plugin: {name}")
    
    def _create_dynamic_class(self, class_name: str, base_type: str):
        """Create a dynamic class"""
        # This is a simplified version - in practice, you'd import actual classes
        from core.modules.base import BaseModule
        
        class DynamicClass(BaseModule):
            def __init__(self, name: str, config: Dict[str, Any] = None):
                super().__init__(name, config)
                self.class_name = class_name
                self.base_type = base_type
            
            def initialize(self) -> bool:
                self.set_state(ModuleState.INITIALIZED)
                return True
            
            def start(self) -> bool:
                self.set_state(ModuleState.RUNNING)
                return True
            
            def stop(self) -> bool:
                self.set_state(ModuleState.STOPPED)
                return True
            
            def cleanup(self) -> bool:
                return True
        
        return DynamicClass
    
    def _validate_configurations(self) -> None:
        """Validate configurations"""
        # Load all configurations
        results = self.config_manager.load_all()
        
        # Validate each configuration
        for config_name, success in results.items():
            if success:
                errors = self.config_manager.validate_config(config_name)
                if errors:
                    logger.warning(f"Configuration {config_name} has validation errors: {errors}")
                else:
                    logger.info(f"Configuration {config_name} is valid")
            else:
                logger.error(f"Failed to load configuration {config_name}")
    
    def start_system(self) -> bool:
        """Start the modular system"""
        try:
            # Start all modules
            start_results = self.module_manager.start_all()
            
            # Check results
            failed_modules = [name for name, success in start_results.items() if not success]
            if failed_modules:
                logger.error(f"Failed to start modules: {failed_modules}")
                return False
            
            logger.info("✅ All modules started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    def stop_system(self) -> bool:
        """Stop the modular system"""
        try:
            # Stop all modules
            stop_results = self.module_manager.stop_all()
            
            # Check results
            failed_modules = [name for name, success in stop_results.items() if not success]
            if failed_modules:
                logger.error(f"Failed to stop modules: {failed_modules}")
                return False
            
            logger.info("✅ All modules stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "modules": self.module_manager.get_status(),
            "plugins": self.plugin_manager.get_plugin_status(),
            "configurations": {
                name: self.config_manager.get_config_info(name)
                for name in self.config_manager.list_configs()
            }
        }

def main():
    """Main function to demonstrate modular configuration loading"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Modular Configuration Loader')
    parser.add_argument('--config', default='modular_production_config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--action', choices=['load', 'start', 'stop', 'status'], 
                       default='load', help='Action to perform')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create loader
        loader = ModularConfigLoader(args.config)
        
        if args.action == 'load':
            # Load configuration
            config = loader.load_config()
            logger.info(f"Loaded configuration with {len(config)} sections")
            
            # Setup modular system
            if loader.setup_modular_system():
                logger.info("✅ Modular system setup completed")
            else:
                logger.error("❌ Failed to setup modular system")
                return 1
        
        elif args.action == 'start':
            # Load and setup
            loader.load_config()
            loader.setup_modular_system()
            
            # Start system
            if loader.start_system():
                logger.info("✅ System started successfully")
            else:
                logger.error("❌ Failed to start system")
                return 1
        
        elif args.action == 'stop':
            # Load and setup
            loader.load_config()
            loader.setup_modular_system()
            
            # Stop system
            if loader.stop_system():
                logger.info("✅ System stopped successfully")
            else:
                logger.error("❌ Failed to stop system")
                return 1
        
        elif args.action == 'status':
            # Load and setup
            loader.load_config()
            loader.setup_modular_system()
            
            # Get status
            status = loader.get_system_status()
            logger.info(f"System status: {status}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

