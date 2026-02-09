from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, TypeVar
from dataclasses import dataclass, field
import copy
from config_manager import CompleteConfig, ConfigManager
        from config_manager import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Configuration Loader
===================

This module provides utilities for loading configurations from YAML files
and merging them with command-line arguments and environment variables.

Features:
- YAML configuration loading
- Environment variable overrides
- Command-line argument merging
- Configuration validation
- Configuration templates
- Configuration inheritance
"""


# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic config classes
T = TypeVar('T', bound='BaseConfig')


class ConfigLoader:
    """Loader for configuration files with environment and command-line overrides."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_manager = ConfigManager(config_dir)
        
        logger.info(f"ConfigLoader initialized with config directory: {self.config_dir}")
    
    def load_config_from_file(self, config_file: str, config_name: Optional[str] = None) -> CompleteConfig:
        """Load configuration from YAML file."""
        filepath = self.config_dir / config_file
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_data = yaml.safe_load(f)
        
        # If config_name is specified, extract that specific configuration
        if config_name and config_name in config_data:
            config_data = config_data[config_name]
        
        # Convert to CompleteConfig object
        config = self._dict_to_config(config_data)
        
        logger.info(f"Loaded configuration from {filepath}")
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CompleteConfig:
        """Convert dictionary to CompleteConfig object."""
            SystemConfig, ModelConfig, DataConfig, 
            TrainingConfig, EvaluationConfig, CompleteConfig
        )
        
        # Extract component configurations
        system_config = SystemConfig(**config_dict.get('system', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # Create complete configuration
        complete_config = CompleteConfig(
            name=config_dict.get('name', 'default'),
            description=config_dict.get('description', ''),
            version=config_dict.get('version', '1.0.0'),
            created_at=config_dict.get('created_at', ''),
            system=system_config,
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config
        )
        
        return complete_config
    
    def load_config_with_overrides(
        self, 
        config_file: str, 
        config_name: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        cli_overrides: Optional[Dict[str, Any]] = None
    ) -> CompleteConfig:
        """Load configuration with environment and command-line overrides."""
        # Load base configuration
        config = self.load_config_from_file(config_file, config_name)
        
        # Apply environment overrides
        if env_overrides:
            config = self._apply_env_overrides(config, env_overrides)
        
        # Apply command-line overrides
        if cli_overrides:
            config = self._apply_cli_overrides(config, cli_overrides)
        
        # Validate final configuration
        if not config.validate():
            errors = config.get_validation_errors()
            logger.warning(f"Configuration validation warnings:\n" + "\n".join(errors))
        
        return config
    
    def _apply_env_overrides(self, config: CompleteConfig, env_overrides: Dict[str, str]) -> CompleteConfig:
        """Apply environment variable overrides to configuration."""
        config_dict = config.to_dict()
        
        for env_var, value in env_overrides.items():
            # Parse environment variable name (e.g., "MODEL_FRAME_SIZE" -> ["model", "frame_size"])
            parts = env_var.lower().split('_')
            
            # Find the corresponding configuration path
            if len(parts) >= 2:
                section = parts[0]
                key = '_'.join(parts[1:])
                
                if section in config_dict and key in config_dict[section]:
                    # Convert value to appropriate type
                    original_value = config_dict[section][key]
                    converted_value = self._convert_value(value, type(original_value))
                    config_dict[section][key] = converted_value
                    
                    logger.debug(f"Applied environment override: {env_var}={value}")
        
        return CompleteConfig.from_dict(config_dict)
    
    def _apply_cli_overrides(self, config: CompleteConfig, cli_overrides: Dict[str, Any]) -> CompleteConfig:
        """Apply command-line argument overrides to configuration."""
        config_dict = config.to_dict()
        
        for key_path, value in cli_overrides.items():
            # Parse key path (e.g., "model.frame_size" -> ["model", "frame_size"])
            parts = key_path.split('.')
            
            if len(parts) >= 2:
                section = parts[0]
                key = '.'.join(parts[1:])
                
                if section in config_dict:
                    # Navigate to the nested key
                    current = config_dict[section]
                    key_parts = key.split('.')
                    
                    for i, part in enumerate(key_parts[:-1]):
                        if part in current:
                            current = current[part]
                        else:
                            break
                    else:
                        # Set the final value
                        final_key = key_parts[-1]
                        if final_key in current:
                            original_value = current[final_key]
                            converted_value = self._convert_value(value, type(original_value))
                            current[final_key] = converted_value
                            
                            logger.debug(f"Applied CLI override: {key_path}={value}")
        
        return CompleteConfig.from_dict(config_dict)
    
    def _convert_value(self, value: str, target_type: Type) -> Any:
        """Convert string value to target type."""
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            # Handle list values (e.g., "1,2,3" -> [1, 2, 3])
            if isinstance(value, str):
                return [int(x.strip()) for x in value.split(',')]
            return value
        else:
            return value
    
    def create_config_from_template(
        self, 
        template_name: str, 
        output_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> CompleteConfig:
        """Create a new configuration from a template."""
        # Load template
        template_config = self.load_config_from_file("default_configs.yaml", template_name)
        
        # Apply overrides
        if overrides:
            template_config = self._apply_cli_overrides(template_config, overrides)
        
        # Update metadata
        template_config.name = output_name
        template_config.description = f"Configuration based on {template_name} template"
        
        return template_config
    
    def save_config_template(self, config: CompleteConfig, template_name: str) -> Path:
        """Save configuration as a template."""
        filepath = self.config_dir / f"{template_name}_template.yaml"
        
        # Convert to template format
        template_data = {
            template_name: config.to_dict()
        }
        
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(template_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration template to {filepath}")
        return filepath


class CommandLineConfigLoader:
    """Command-line interface for configuration loading."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_loader = ConfigLoader(config_dir)
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(description="AI Video Configuration Loader")
        
        # Basic arguments
        parser.add_argument(
            "--config", "-c",
            type=str,
            default="default_configs.yaml",
            help="Configuration file path"
        )
        parser.add_argument(
            "--config-name", "-n",
            type=str,
            help="Configuration name within the file"
        )
        parser.add_argument(
            "--output", "-o",
            type=str,
            help="Output configuration file path"
        )
        parser.add_argument(
            "--template", "-t",
            type=str,
            help="Template name to use"
        )
        parser.add_argument(
            "--validate", "-v",
            action="store_true",
            help="Validate configuration"
        )
        parser.add_argument(
            "--list", "-l",
            action="store_true",
            help="List available configurations"
        )
        
        # Model-specific overrides
        parser.add_argument(
            "--model-type",
            type=str,
            choices=["diffusion", "gan", "transformer"],
            help="Model type override"
        )
        parser.add_argument(
            "--frame-size",
            type=str,
            help="Frame size override (e.g., '256,256')"
        )
        parser.add_argument(
            "--num-frames",
            type=int,
            help="Number of frames override"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Batch size override"
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            help="Learning rate override"
        )
        parser.add_argument(
            "--num-epochs",
            type=int,
            help="Number of epochs override"
        )
        
        # System overrides
        parser.add_argument(
            "--device",
            type=str,
            choices=["cpu", "cuda", "mps"],
            help="Device override"
        )
        parser.add_argument(
            "--environment",
            type=str,
            choices=["development", "staging", "production"],
            help="Environment override"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        return self.parser.parse_args(args)
    
    def load_config_from_args(self, args: argparse.Namespace) -> CompleteConfig:
        """Load configuration from command-line arguments."""
        # Convert arguments to overrides
        overrides = {}
        
        if args.model_type:
            overrides["model.model_type"] = args.model_type
        if args.frame_size:
            overrides["model.frame_size"] = [int(x) for x in args.frame_size.split(',')]
        if args.num_frames:
            overrides["model.num_frames"] = args.num_frames
        if args.batch_size:
            overrides["training.batch_size"] = args.batch_size
            overrides["data.batch_size"] = args.batch_size
        if args.learning_rate:
            overrides["training.learning_rate"] = args.learning_rate
        if args.num_epochs:
            overrides["training.num_epochs"] = args.num_epochs
        if args.device:
            overrides["model.device"] = args.device
            overrides["evaluation.device"] = args.device
        if args.environment:
            overrides["system.environment"] = args.environment
        if args.debug:
            overrides["system.debug"] = True
        
        # Load configuration
        if args.template:
            config = self.config_loader.create_config_from_template(
                args.template, 
                f"{args.template}_custom",
                overrides
            )
        else:
            config = self.config_loader.load_config_with_overrides(
                args.config,
                args.config_name,
                cli_overrides=overrides
            )
        
        return config
    
    def run(self, args: Optional[List[str]] = None) -> CompleteConfig:
        """Run the configuration loader."""
        parsed_args = self.parse_args(args)
        
        if parsed_args.list:
            self._list_configurations()
            return None
        
        config = self.load_config_from_args(parsed_args)
        
        if parsed_args.validate:
            self._validate_configuration(config)
        
        if parsed_args.output:
            self.config_loader.config_manager.save_config(config, parsed_args.output)
        
        return config
    
    def _list_configurations(self) -> List[Any]:
        """List available configurations."""
        configs = self.config_loader.config_manager.list_configs()
        print("Available configurations:")
        for config in configs:
            print(f"  - {config}")
    
    def _validate_configuration(self, config: CompleteConfig):
        """Validate configuration and print results."""
        validation_results = self.config_loader.config_manager.validate_config(config)
        
        if validation_results:
            print("Configuration validation errors:")
            for component, errors in validation_results.items():
                for error in errors:
                    print(f"  {component}: {error}")
        else:
            print("Configuration is valid")


# Environment variable utilities
def get_env_overrides(prefix: str = "AI_VIDEO_") -> Dict[str, str]:
    """Get environment variable overrides with specified prefix."""
    overrides = {}
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):]
            overrides[config_key] = value
    
    return overrides


def load_config_with_env(
    config_file: str,
    config_name: Optional[str] = None,
    env_prefix: str = "AI_VIDEO_"
) -> CompleteConfig:
    """Load configuration with environment variable overrides."""
    loader = ConfigLoader()
    env_overrides = get_env_overrides(env_prefix)
    
    return loader.load_config_with_overrides(
        config_file,
        config_name,
        env_overrides=env_overrides
    )


# Convenience functions
def quick_load_config(config_name: str = "diffusion_default") -> CompleteConfig:
    """Quickly load a default configuration."""
    loader = ConfigLoader()
    return loader.load_config_from_file("default_configs.yaml", config_name)


def create_custom_config(
    base_config: str,
    overrides: Dict[str, Any],
    output_name: str
) -> CompleteConfig:
    """Create a custom configuration from base with overrides."""
    loader = ConfigLoader()
    config = loader.load_config_from_file("default_configs.yaml", base_config)
    
    for key_path, value in overrides.items():
        config = loader._apply_cli_overrides(config, {key_path: value})
    
    config.name = output_name
    return config


if __name__ == "__main__":
    # Example usage
    print("🔧 Configuration Loader")
    print("=" * 30)
    
    # Create loader
    loader = ConfigLoader()
    
    # Load default configuration
    config = quick_load_config("diffusion_default")
    print(f"✅ Loaded configuration: {config.name}")
    
    # Create custom configuration
    custom_config = create_custom_config(
        "diffusion_default",
        {
            "model.frame_size": [512, 512],
            "training.num_epochs": 50,
            "training.learning_rate": 5e-5
        },
        "my_custom_config"
    )
    print(f"✅ Created custom configuration: {custom_config.name}")
    
    # Save custom configuration
    loader.config_manager.save_config(custom_config, "my_custom_config.yaml")
    print("✅ Saved custom configuration")
    
    # Load with environment overrides
    env_config = load_config_with_env("default_configs.yaml", "diffusion_default")
    print(f"✅ Loaded configuration with environment overrides: {env_config.name}")
    
    # Command-line interface
    cli_loader = CommandLineConfigLoader()
    print("\n📋 Command-line usage examples:")
    print("  python config_loader.py --list")
    print("  python config_loader.py --template diffusion_default --output my_config.yaml")
    print("  python config_loader.py --config my_config.yaml --batch-size 16 --num-epochs 50") 