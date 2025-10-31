"""
Dynamic Configuration Management for Ultra-Adaptive K/V Cache Engine
Allows runtime configuration changes without engine restart
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, asdict
import logging
import time

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    from ultra_adaptive_kv_cache_engine import (
        UltraAdaptiveKVCacheEngine,
        AdaptiveConfig,
        AdaptiveMode
    )
except ImportError:
    UltraAdaptiveKVCacheEngine = None

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Configuration change event."""
    timestamp: float
    parameter: str
    old_value: Any
    new_value: Any
    source: str


class ConfigManager:
    """Manage dynamic configuration for engine."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine, config_file: Optional[str] = None):
        self.engine = engine
        self.config_file = Path(config_file) if config_file else None
        self.config_history: List[ConfigChange] = []
        self.change_callbacks: List[Callable[[ConfigChange], None]] = []
        self.watchdog_observer = None
        
        if self.config_file:
            self._start_watching()
    
    def _start_watching(self):
        """Start watching config file for changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available - file watching disabled")
            return
        
        try:
            class ConfigFileHandler(FileSystemEventHandler):
                def __init__(self, manager):
                    self.manager = manager
                
                def on_modified(self, event):
                    if not event.is_directory and Path(event.src_path) == self.manager.config_file:
                        logger.info(f"Config file changed: {event.src_path}")
                        asyncio.create_task(self.manager.reload_from_file())
            
            self.watchdog_observer = Observer()
            self.watchdog_observer.schedule(
                ConfigFileHandler(self),
                str(self.config_file.parent),
                recursive=False
            )
            self.watchdog_observer.start()
            logger.info(f"Started watching config file: {self.config_file}")
        
        except Exception as e:
            logger.warning(f"Failed to start file watching: {e}")
    
    def stop_watching(self):
        """Stop watching config file."""
        if self.watchdog_observer:
            self.watchdog_observer.stop()
            self.watchdog_observer.join()
            logger.info("Stopped watching config file")
    
    async def reload_from_file(self):
        """Reload configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return
        
        try:
            with open(self.config_file) as f:
                config_data = json.load(f)
            
            # Apply configuration changes
            for key, value in config_data.items():
                if hasattr(self.engine.config, key):
                    old_value = getattr(self.engine.config, key)
                    if old_value != value:
                        await self.update_config(key, value, source="file")
        
            logger.info("Configuration reloaded from file")
        
        except Exception as e:
            logger.error(f"Failed to reload config from file: {e}")
    
    async def update_config(self, parameter: str, value: Any, source: str = "manual") -> bool:
        """
        Update a configuration parameter at runtime.
        
        Args:
            parameter: Parameter name to update
            value: New value
            source: Source of change (manual, file, api)
            
        Returns:
            True if successful
        """
        if not hasattr(self.engine.config, parameter):
            logger.error(f"Unknown configuration parameter: {parameter}")
            return False
        
        old_value = getattr(self.engine.config, parameter)
        
        # Validate and apply change
        try:
            # Type checking
            expected_type = type(old_value)
            if not isinstance(value, expected_type):
                # Try to convert
                if expected_type == bool:
                    value = str(value).lower() in ('true', '1', 'yes')
                else:
                    value = expected_type(value)
            
            # Update config
            setattr(self.engine.config, parameter, value)
            
            # Apply to engine if needed
            await self._apply_config_change(parameter, value)
            
            # Record change
            change = ConfigChange(
                timestamp=time.time(),
                parameter=parameter,
                old_value=old_value,
                new_value=value,
                source=source
            )
            
            self.config_history.append(change)
            
            # Keep only last 100 changes
            if len(self.config_history) > 100:
                self.config_history = self.config_history[-100:]
            
            # Notify callbacks
            for callback in self.change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Error in config change callback: {e}")
            
            logger.info(f"Configuration updated: {parameter} = {value} (was {old_value})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update configuration {parameter}: {e}")
            return False
    
    async def _apply_config_change(self, parameter: str, value: Any):
        """Apply configuration change to engine components."""
        # Reinitialize components if needed
        if parameter in ['cache_size', 'compression_ratio', 'quantization_bits']:
            if hasattr(self.engine, '_reinitialize_components'):
                self.engine._reinitialize_components()
        
        elif parameter == 'num_workers':
            if hasattr(self.engine, 'executor'):
                self.engine.executor.shutdown(wait=True)
                from concurrent.futures import ThreadPoolExecutor
                self.engine.executor = ThreadPoolExecutor(max_workers=value)
        
        elif parameter in ['enable_cache_persistence', 'cache_persistence_path']:
            if hasattr(self.engine, '_setup_persistence'):
                self.engine._setup_persistence()
        
        elif parameter in ['enable_checkpointing', 'checkpoint_interval']:
            if hasattr(self.engine, '_setup_checkpointing'):
                self.engine._setup_checkpointing()
    
    def register_change_callback(self, callback: Callable[[ConfigChange], None]):
        """Register callback for configuration changes."""
        self.change_callbacks.append(callback)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        try:
            return asdict(self.engine.config)
        except:
            return {
                'model_name': self.engine.config.model_name,
                'model_size': self.engine.config.model_size,
                'cache_size': self.engine.config.cache_size,
                'num_workers': self.engine.config.num_workers,
                # Add more as needed
            }
    
    def get_config_history(self, parameter: Optional[str] = None) -> List[ConfigChange]:
        """Get configuration change history."""
        if parameter:
            return [c for c in self.config_history if c.parameter == parameter]
        return self.config_history.copy()
    
    def save_config_to_file(self, output_file: Optional[str] = None):
        """Save current configuration to file."""
        file_path = Path(output_file) if output_file else self.config_file
        
        if not file_path:
            raise ValueError("No config file specified")
        
        config_dict = self.get_config()
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        default_config = AdaptiveConfig()
        
        for key, value in asdict(default_config).items():
            if hasattr(self.engine.config, key):
                current_value = getattr(self.engine.config, key)
                if current_value != value:
                    asyncio.create_task(self.update_config(key, value, source="reset"))


class ConfigValidator:
    """Validate configuration values."""
    
    @staticmethod
    def validate(config_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate configuration dictionary.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        required_fields = ['model_name', 'cache_size', 'num_workers']
        for field in required_fields:
            if field not in config_dict:
                return False, f"Missing required field: {field}"
        
        # Validate cache_size
        cache_size = config_dict.get('cache_size')
        if not isinstance(cache_size, int) or cache_size < 1 or cache_size > 100000:
            return False, "cache_size must be between 1 and 100000"
        
        # Validate num_workers
        num_workers = config_dict.get('num_workers')
        if not isinstance(num_workers, int) or num_workers < 1 or num_workers > 128:
            return False, "num_workers must be between 1 and 128"
        
        # Validate memory_usage
        max_memory_usage = config_dict.get('max_memory_usage', 0.8)
        if not isinstance(max_memory_usage, (int, float)) or max_memory_usage < 0.1 or max_memory_usage > 1.0:
            return False, "max_memory_usage must be between 0.1 and 1.0"
        
        # Validate compression_ratio
        compression_ratio = config_dict.get('compression_ratio', 0.3)
        if not isinstance(compression_ratio, (int, float)) or compression_ratio < 0.1 or compression_ratio > 1.0:
            return False, "compression_ratio must be between 0.1 and 1.0"
        
        return True, None


class ConfigPreset:
    """Configuration presets for common use cases."""
    
    @staticmethod
    def get_preset(name: str) -> Optional[Dict[str, Any]]:
        """Get configuration preset by name."""
        presets = {
            'development': {
                'cache_size': 1024,
                'num_workers': 2,
                'enable_cache_persistence': False,
                'enable_checkpointing': False,
                'enable_metrics': True,
                'enable_profiling': False
            },
            'production': {
                'cache_size': 16384,
                'num_workers': 8,
                'enable_cache_persistence': True,
                'enable_checkpointing': True,
                'enable_metrics': True,
                'enable_profiling': False
            },
            'high_performance': {
                'cache_size': 32768,
                'num_workers': 16,
                'cache_strategy': 'SPEED',
                'memory_strategy': 'SPEED',
                'enable_prefetching': True,
                'dynamic_batching': True
            },
            'memory_efficient': {
                'cache_size': 4096,
                'compression_ratio': 0.5,
                'quantization_bits': 4,
                'memory_strategy': 'AGGRESSIVE',
                'max_memory_usage': 0.6
            },
            'bulk_processing': {
                'cache_size': 32768,
                'num_workers': 16,
                'adaptive_mode': 'BULK',
                'dynamic_batching': True,
                'batch_size': 20
            }
        }
        
        return presets.get(name.lower())
    
    @staticmethod
    def list_presets() -> List[str]:
        """List available presets."""
        return ['development', 'production', 'high_performance', 'memory_efficient', 'bulk_processing']
    
    @staticmethod
    def apply_preset(engine, preset_name: str) -> bool:
        """Apply preset to engine."""
        preset = ConfigPreset.get_preset(preset_name)
        if not preset:
            return False
        
        config_manager = ConfigManager(engine)
        
        for key, value in preset.items():
            asyncio.create_task(config_manager.update_config(key, value, source="preset"))
        
        return True

