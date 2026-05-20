from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import yaml
            import aiohttp
from typing import Any, List, Dict, Optional
"""
Configuration Management
========================

Dynamic configuration management for microservices:
- Consul KV Store
- etcd
- Kubernetes ConfigMaps
- Environment variables
- File-based configuration
"""


logger = logging.getLogger(__name__)

@dataclass
class ConfigurationItem:
    """Configuration item with metadata."""
    key: str
    value: Any
    version: int = 1
    source: str = "unknown"
    last_updated: Optional[str] = None


class IConfigurationProvider(ABC):
    """Abstract interface for configuration providers."""
    
    @abstractmethod
    async def get_config(self, key: str) -> Optional[ConfigurationItem]:
        """Get configuration value by key."""
        pass
    
    @abstractmethod
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value."""
        pass
    
    @abstractmethod
    async def delete_config(self, key: str) -> bool:
        """Delete configuration value."""
        pass
    
    @abstractmethod
    async def watch_config(self, key: str, callback: Callable[[ConfigurationItem], None]) -> str:
        """Watch configuration changes."""
        pass
    
    @abstractmethod
    async def list_configs(self, prefix: str = "") -> Dict[str, ConfigurationItem]:
        """List all configurations with optional prefix."""
        pass


class ConsulConfigProvider(IConfigurationProvider):
    """Consul KV store configuration provider."""
    
    def __init__(self, consul_url: str = "http://localhost:8500"):
        
    """__init__ function."""
self.consul_url = consul_url.rstrip('/')
        self.session = None
        self.watchers: Dict[str, asyncio.Task] = {}
        
    async def _get_session(self) -> Optional[Dict[str, Any]]:
        """Get HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_config(self, key: str) -> Optional[ConfigurationItem]:
        """Get configuration from Consul KV."""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.consul_url}/v1/kv/{key}") as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                if not data:
                    return None
                
                item = data[0]
                value = json.loads(item['Value'].decode('base64'))
                
                return ConfigurationItem(
                    key=key,
                    value=value,
                    version=item['ModifyIndex'],
                    source="consul",
                    last_updated=item['ModifyIndex']
                )
                
        except Exception as e:
            logger.error(f"Error getting config from Consul: {e}")
            return None
    
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration in Consul KV."""
        try:
            session = await self._get_session()
            
            async with session.put(
                f"{self.consul_url}/v1/kv/{key}",
                data=json.dumps(value)
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error setting config in Consul: {e}")
            return False
    
    async def delete_config(self, key: str) -> bool:
        """Delete configuration from Consul KV."""
        try:
            session = await self._get_session()
            
            async with session.delete(f"{self.consul_url}/v1/kv/{key}") as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error deleting config from Consul: {e}")
            return False
    
    async def watch_config(self, key: str, callback: Callable[[ConfigurationItem], None]) -> str:
        """Watch configuration changes in Consul."""
        watcher_id = f"consul_{key}_{id(callback)}"
        
        async def watch_loop():
            
    """watch_loop function."""
last_index = 0
            while True:
                try:
                    session = await self._get_session()
                    
                    async with session.get(
                        f"{self.consul_url}/v1/kv/{key}",
                        params={"index": last_index, "wait": "30s"}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data:
                                item = data[0]
                                if item['ModifyIndex'] > last_index:
                                    last_index = item['ModifyIndex']
                                    
                                    config_item = ConfigurationItem(
                                        key=key,
                                        value=json.loads(item['Value'].decode('base64')),
                                        version=item['ModifyIndex'],
                                        source="consul"
                                    )
                                    
                                    await callback(config_item)
                        
                        await asyncio.sleep(1)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in Consul watch loop: {e}")
                    await asyncio.sleep(5)
        
        task = asyncio.create_task(watch_loop())
        self.watchers[watcher_id] = task
        return watcher_id
    
    async def list_configs(self, prefix: str = "") -> Dict[str, ConfigurationItem]:
        """List configurations from Consul KV."""
        try:
            session = await self._get_session()
            
            async with session.get(
                f"{self.consul_url}/v1/kv/{prefix}",
                params={"recurse": "true"}
            ) as response:
                if response.status != 200:
                    return {}
                
                data = await response.json()
                configs = {}
                
                for item in data:
                    key = item['Key']
                    value = json.loads(item['Value'].decode('base64'))
                    
                    configs[key] = ConfigurationItem(
                        key=key,
                        value=value,
                        version=item['ModifyIndex'],
                        source="consul"
                    )
                
                return configs
                
        except Exception as e:
            logger.error(f"Error listing configs from Consul: {e}")
            return {}


class EnvironmentConfigProvider(IConfigurationProvider):
    """Environment variables configuration provider."""
    
    def __init__(self, prefix: str = ""):
        
    """__init__ function."""
self.prefix = prefix
        
    async def get_config(self, key: str) -> Optional[ConfigurationItem]:
        """Get configuration from environment."""
        env_key = f"{self.prefix}{key}" if self.prefix else key
        value = os.getenv(env_key)
        
        if value is None:
            return None
        
        # Try to parse as JSON, fallback to string
        try:
            parsed_value = json.loads(value)
        except:
            parsed_value = value
        
        return ConfigurationItem(
            key=key,
            value=parsed_value,
            source="environment"
        )
    
    async def set_config(self, key: str, value: Any) -> bool:
        """Set environment variable (for current process only)."""
        env_key = f"{self.prefix}{key}" if self.prefix else key
        os.environ[env_key] = json.dumps(value) if not isinstance(value, str) else value
        return True
    
    async def delete_config(self, key: str) -> bool:
        """Delete environment variable."""
        env_key = f"{self.prefix}{key}" if self.prefix else key
        if env_key in os.environ:
            del os.environ[env_key]
            return True
        return False
    
    async def watch_config(self, key: str, callback: Callable[[ConfigurationItem], None]) -> str:
        """Environment variables don't support watching."""
        return ""
    
    async def list_configs(self, prefix: str = "") -> Dict[str, ConfigurationItem]:
        """List environment configurations."""
        configs = {}
        full_prefix = f"{self.prefix}{prefix}" if self.prefix else prefix
        
        for env_key, value in os.environ.items():
            if env_key.startswith(full_prefix):
                key = env_key[len(self.prefix):] if self.prefix else env_key
                
                try:
                    parsed_value = json.loads(value)
                except:
                    parsed_value = value
                
                configs[key] = ConfigurationItem(
                    key=key,
                    value=parsed_value,
                    source="environment"
                )
        
        return configs


class FileConfigProvider(IConfigurationProvider):
    """File-based configuration provider."""
    
    def __init__(self, file_path: str):
        
    """__init__ function."""
self.file_path = file_path
        self.configs: Dict[str, Any] = {}
        self.file_watchers: Dict[str, asyncio.Task] = {}
        
    async def _load_file(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if self.file_path.endswith('.yaml') or self.file_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file {self.file_path}: {e}")
            return {}
    
    async def get_config(self, key: str) -> Optional[ConfigurationItem]:
        """Get configuration from file."""
        configs = await self._load_file()
        
        # Support dot notation for nested keys
        keys = key.split('.')
        value = configs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return ConfigurationItem(
            key=key,
            value=value,
            source=f"file:{self.file_path}"
        )
    
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration in file (not implemented for safety)."""
        logger.warning("File configuration provider does not support setting values")
        return False
    
    async def delete_config(self, key: str) -> bool:
        """Delete configuration from file (not implemented for safety)."""
        logger.warning("File configuration provider does not support deleting values")
        return False
    
    async def watch_config(self, key: str, callback: Callable[[ConfigurationItem], None]) -> str:
        """Watch file for changes."""
        watcher_id = f"file_{key}_{id(callback)}"
        
        async def watch_file():
            
    """watch_file function."""
last_mtime = 0
            while True:
                try:
                    current_mtime = os.path.getmtime(self.file_path)
                    if current_mtime > last_mtime:
                        last_mtime = current_mtime
                        config_item = await self.get_config(key)
                        if config_item:
                            await callback(config_item)
                    
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error watching file {self.file_path}: {e}")
                    await asyncio.sleep(5)
        
        task = asyncio.create_task(watch_file())
        self.file_watchers[watcher_id] = task
        return watcher_id
    
    async def list_configs(self, prefix: str = "") -> Dict[str, ConfigurationItem]:
        """List all configurations from file."""
        configs = await self._load_file()
        result = {}
        
        def flatten_dict(d: dict, parent_key: str = "", sep: str = "."):
            """Flatten nested dictionary."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_configs = flatten_dict(configs)
        
        for key, value in flat_configs.items():
            if key.startswith(prefix):
                result[key] = ConfigurationItem(
                    key=key,
                    value=value,
                    source=f"file:{self.file_path}"
                )
        
        return result


class ConfigurationManager:
    """Unified configuration manager with multiple providers."""
    
    def __init__(self) -> Any:
        self.providers: Dict[str, IConfigurationProvider] = {}
        self.cache: Dict[str, ConfigurationItem] = {}
        self.cache_ttl = 60  # seconds
        
    def add_provider(self, name: str, provider: IConfigurationProvider, priority: int = 0):
        """Add configuration provider."""
        self.providers[name] = provider
        logger.info(f"Added configuration provider: {name}")
    
    async def get_config(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get configuration value from providers (in priority order)."""
        # Check cache first
        if key in self.cache:
            cached_item = self.cache[key]
            # Simple cache expiry (in real implementation, use timestamps)
            return cached_item.value
        
        # Try providers in order
        for name, provider in self.providers.items():
            try:
                config_item = await provider.get_config(key)
                if config_item:
                    self.cache[key] = config_item
                    return config_item.value
            except Exception as e:
                logger.error(f"Error getting config from {name}: {e}")
        
        return default
    
    async def get_all_configs(self, prefix: str = "") -> Dict[str, Any]:
        """Get all configurations with optional prefix."""
        all_configs = {}
        
        for name, provider in self.providers.items():
            try:
                configs = await provider.list_configs(prefix)
                for key, config_item in configs.items():
                    if key not in all_configs:  # First provider wins
                        all_configs[key] = config_item.value
            except Exception as e:
                logger.error(f"Error listing configs from {name}: {e}")
        
        return all_configs
    
    def clear_cache(self) -> Any:
        """Clear configuration cache."""
        self.cache.clear()
        logger.info("Configuration cache cleared") 