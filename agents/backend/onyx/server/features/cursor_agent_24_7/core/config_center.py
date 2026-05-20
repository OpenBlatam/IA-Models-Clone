"""
Config Center - Configuración centralizada
==========================================

Sistema de configuración centralizada para microservicios.
Soporta múltiples backends: Consul, etcd, AWS Systems Manager.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigCenter:
    """
    Centro de configuración centralizada.
    
    Soporta:
    - Consul KV
    - etcd
    - AWS Systems Manager Parameter Store
    - Archivos locales
    """
    
    def __init__(self, backend: str = "file"):
        """
        Inicializar config center.
        
        Args:
            backend: Backend a usar ("consul", "etcd", "aws", "file").
        """
        self.backend = backend
        self._client = None
        self._cache: Dict[str, Any] = {}
        
        if backend == "consul":
            self._init_consul()
        elif backend == "etcd":
            self._init_etcd()
        elif backend == "aws":
            self._init_aws()
    
    def _init_consul(self) -> None:
        """Inicializar Consul."""
        try:
            import consul
            consul_host = os.getenv("CONSUL_HOST", "localhost")
            consul_port = int(os.getenv("CONSUL_PORT", "8500"))
            self._client = consul.Consul(host=consul_host, port=consul_port)
            logger.info("Config Center: Consul initialized")
        except ImportError:
            logger.warning("Consul not available, falling back to file")
            self.backend = "file"
        except Exception as e:
            logger.warning(f"Failed to connect to Consul: {e}, falling back to file")
            self.backend = "file"
    
    def _init_etcd(self) -> None:
        """Inicializar etcd."""
        try:
            import etcd3
            etcd_host = os.getenv("ETCD_HOST", "localhost")
            etcd_port = int(os.getenv("ETCD_PORT", "2379"))
            self._client = etcd3.client(host=etcd_host, port=etcd_port)
            logger.info("Config Center: etcd initialized")
        except ImportError:
            logger.warning("etcd3 not available, falling back to file")
            self.backend = "file"
        except Exception as e:
            logger.warning(f"Failed to connect to etcd: {e}, falling back to file")
            self.backend = "file"
    
    def _init_aws(self) -> None:
        """Inicializar AWS Systems Manager."""
        try:
            import boto3
            self._client = boto3.client('ssm', region_name=os.getenv("AWS_REGION", "us-east-1"))
            logger.info("Config Center: AWS SSM initialized")
        except ImportError:
            logger.warning("boto3 not available, falling back to file")
            self.backend = "file"
        except Exception as e:
            logger.warning(f"Failed to connect to AWS SSM: {e}, falling back to file")
            self.backend = "file"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración.
        
        Args:
            key: Clave de configuración.
            default: Valor por defecto.
        
        Returns:
            Valor de configuración.
        """
        # Verificar cache primero
        if key in self._cache:
            return self._cache[key]
        
        value = None
        
        if self.backend == "consul":
            value = self._get_consul(key)
        elif self.backend == "etcd":
            value = self._get_etcd(key)
        elif self.backend == "aws":
            value = self._get_aws(key)
        else:
            value = self._get_file(key)
        
        if value is None:
            value = default
        
        # Guardar en cache
        if value is not None:
            self._cache[key] = value
        
        return value
    
    def _get_consul(self, key: str) -> Optional[Any]:
        """Obtener de Consul."""
        try:
            _, data = self._client.kv.get(f"config/{key}")
            if data:
                return json.loads(data['Value'].decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to get from Consul: {e}")
        return None
    
    def _get_etcd(self, key: str) -> Optional[Any]:
        """Obtener de etcd."""
        try:
            value, _ = self._client.get(f"/config/{key}")
            if value:
                return json.loads(value.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to get from etcd: {e}")
        return None
    
    def _get_aws(self, key: str) -> Optional[Any]:
        """Obtener de AWS SSM."""
        try:
            response = self._client.get_parameter(
                Name=f"/cursor-agent/{key}",
                WithDecryption=True
            )
            return json.loads(response['Parameter']['Value'])
        except Exception as e:
            logger.error(f"Failed to get from AWS SSM: {e}")
        return None
    
    def _get_file(self, key: str) -> Optional[Any]:
        """Obtener de archivo local."""
        config_file = Path(os.getenv("CONFIG_FILE", "config.json"))
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    return config.get(key)
            except Exception as e:
                logger.error(f"Failed to read config file: {e}")
        return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        Establecer valor de configuración.
        
        Args:
            key: Clave de configuración.
            value: Valor a establecer.
        
        Returns:
            True si se estableció exitosamente.
        """
        # Actualizar cache
        self._cache[key] = value
        
        if self.backend == "consul":
            return self._set_consul(key, value)
        elif self.backend == "etcd":
            return self._set_etcd(key, value)
        elif self.backend == "aws":
            return self._set_aws(key, value)
        else:
            return self._set_file(key, value)
    
    def _set_consul(self, key: str, value: Any) -> bool:
        """Establecer en Consul."""
        try:
            self._client.kv.put(f"config/{key}", json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Failed to set in Consul: {e}")
            return False
    
    def _set_etcd(self, key: str, value: Any) -> bool:
        """Establecer en etcd."""
        try:
            self._client.put(f"/config/{key}", json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Failed to set in etcd: {e}")
            return False
    
    def _set_aws(self, key: str, value: Any) -> bool:
        """Establecer en AWS SSM."""
        try:
            self._client.put_parameter(
                Name=f"/cursor-agent/{key}",
                Value=json.dumps(value),
                Type="String",
                Overwrite=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set in AWS SSM: {e}")
            return False
    
    def _set_file(self, key: str, value: Any) -> bool:
        """Establecer en archivo local."""
        config_file = Path(os.getenv("CONFIG_FILE", "config.json"))
        try:
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
            else:
                config = {}
            
            config[key] = value
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to write config file: {e}")
            return False


# Config center global
_config_center: Optional[ConfigCenter] = None


def get_config_center() -> ConfigCenter:
    """Obtener config center global."""
    global _config_center
    
    if _config_center is None:
        backend = os.getenv("CONFIG_CENTER_BACKEND", "file")
        _config_center = ConfigCenter(backend=backend)
    
    return _config_center




