"""
MOEA Configuration Manager
==========================
Gestor de configuración para el proyecto MOEA
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class MOEAConfig:
    """Gestor de configuración MOEA"""
    
    DEFAULT_CONFIG = {
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "base_url": "http://localhost:8000"
        },
        "moea": {
            "default_population_size": 100,
            "default_generations": 100,
            "default_mutation_rate": 0.1,
            "default_crossover_rate": 0.9
        },
        "algorithms": {
            "enabled": ["nsga2", "nsga3", "moead", "spea2"],
            "default": "nsga2"
        },
        "problems": {
            "enabled": ["ZDT1", "ZDT2", "ZDT3", "DTLZ2", "DTLZ3"],
            "default": "ZDT1"
        },
        "visualization": {
            "output_dir": "moea_visualizations",
            "format": "png",
            "dpi": 300
        },
        "benchmark": {
            "default_runs": 3,
            "default_population": 100,
            "default_generations": 50
        }
    }
    
    def __init__(self, config_file: str = ".moea_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge con defaults
                config = self.DEFAULT_CONFIG.copy()
                self._deep_update(config, user_config)
                return config
            except Exception as e:
                print(f"⚠️  Error cargando configuración: {e}")
                print("   Usando configuración por defecto")
        
        return self.DEFAULT_CONFIG.copy()
    
    def _deep_update(self, base: Dict, update: Dict):
        """Actualizar diccionario recursivamente"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def save(self):
        """Guardar configuración"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Obtener valor de configuración"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Establecer valor de configuración"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def show(self):
        """Mostrar configuración actual"""
        print("📋 Configuración MOEA:")
        print(json.dumps(self.config, indent=2, ensure_ascii=False))
    
    def reset(self):
        """Resetear a configuración por defecto"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()
        print("✅ Configuración reseteada a valores por defecto")


def main():
    """CLI para gestión de configuración"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Configuration Manager")
    parser.add_argument(
        '--config',
        default='.moea_config.json',
        help='Archivo de configuración'
    )
    parser.add_argument(
        '--get',
        help='Obtener valor (ej: api.port)'
    )
    parser.add_argument(
        '--set',
        nargs=2,
        metavar=('KEY', 'VALUE'),
        help='Establecer valor (ej: api.port 8001)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar configuración completa'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Resetear a valores por defecto'
    )
    
    args = parser.parse_args()
    
    config = MOEAConfig(args.config)
    
    if args.get:
        value = config.get(args.get)
        print(f"{args.get} = {value}")
    elif args.set:
        key, value = args.set
        # Intentar convertir a número si es posible
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Mantener como string
            pass
        config.set(key, value)
        config.save()
        print(f"✅ {key} = {value}")
    elif args.show:
        config.show()
    elif args.reset:
        config.reset()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

