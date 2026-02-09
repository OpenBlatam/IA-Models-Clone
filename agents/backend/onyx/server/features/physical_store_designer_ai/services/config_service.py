"""
Config Service - Gestión de configuración avanzada (YAML)
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json

from ..core.service_base import BaseService

# Placeholder para YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigService(BaseService):
    """Servicio para gestión de configuración"""
    
    def __init__(self):
        super().__init__("ConfigService")
        self.configs: Dict[str, Dict[str, Any]] = {}
        
        if not YAML_AVAILABLE:
            self.log_warning("PyYAML no disponible")
    
    def create_config(
        self,
        config_name: str,
        config_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crear configuración"""
        
        config_id = f"config_{config_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        config = {
            "config_id": config_id,
            "name": config_name,
            "data": config_data,
            "created_at": datetime.now().isoformat()
        }
        
        self.configs[config_id] = config
        
        return config
    
    def load_config_from_yaml(
        self,
        yaml_content: str
    ) -> Dict[str, Any]:
        """Cargar configuración desde YAML"""
        
        if not YAML_AVAILABLE:
            return {"error": "PyYAML no disponible"}
        
        try:
            config_data = yaml.safe_load(yaml_content)
            config_id = f"config_yaml_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            config = {
                "config_id": config_id,
                "name": "yaml_config",
                "data": config_data,
                "source": "yaml",
                "created_at": datetime.now().isoformat()
            }
            
            self.configs[config_id] = config
            return config
        except Exception as e:
            self.log_error(f"Error cargando YAML: {e}", exc_info=True)
            return {"error": str(e)}
    
    def create_training_config(
        self,
        model_name: str,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 10,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        loss_function: str = "mse",
        **kwargs
    ) -> Dict[str, Any]:
        """Crear configuración de entrenamiento"""
        
        config_data = {
            "model": {
                "name": model_name,
                "type": "store_design"
            },
            "training": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "loss_function": loss_function,
                **kwargs
            },
            "data": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            },
            "optimization": {
                "mixed_precision": True,
                "gradient_accumulation_steps": 1,
                "gradient_clipping": 1.0
            }
        }
        
        return self.create_config(f"training_{model_name}", config_data)
    
    def create_model_config(
        self,
        model_name: str,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.2
    ) -> Dict[str, Any]:
        """Crear configuración de modelo"""
        
        config_data = {
            "model": {
                "name": model_name,
                "architecture": {
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "output_dim": output_dim,
                    "activation": activation,
                    "dropout": dropout
                }
            }
        }
        
        return self.create_config(f"model_{model_name}", config_data)
    
    def merge_configs(
        self,
        base_config_id: str,
        override_config_id: str
    ) -> Dict[str, Any]:
        """Fusionar configuraciones"""
        
        base_config = self.configs.get(base_config_id)
        override_config = self.configs.get(override_config_id)
        
        if not base_config or not override_config:
            raise ValueError("Configuración no encontrada")
        
        # Merge recursivo
        merged_data = self._deep_merge(
            base_config["data"].copy(),
            override_config["data"].copy()
        )
        
        merged_config = {
            "config_id": f"merged_{base_config_id}_{override_config_id}",
            "name": f"merged_{base_config['name']}_{override_config['name']}",
            "data": merged_data,
            "base_config": base_config_id,
            "override_config": override_config_id,
            "created_at": datetime.now().isoformat()
        }
        
        return merged_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Merge recursivo de diccionarios"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def export_config(
        self,
        config_id: str,
        format: str = "yaml"
    ) -> str:
        """Exportar configuración"""
        
        config = self.configs.get(config_id)
        
        if not config:
            raise ValueError(f"Configuración {config_id} no encontrada")
        
        if format == "yaml" and YAML_AVAILABLE:
            return yaml.dump(config["data"], default_flow_style=False)
        elif format == "json":
            return json.dumps(config["data"], indent=2)
        else:
            return str(config["data"])




