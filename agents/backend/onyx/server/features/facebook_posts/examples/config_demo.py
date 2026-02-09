from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import yaml
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import copy
from datetime import datetime
import shutil
from config_manager import ConfigManager, Config, create_preset_configs
    import json
    from pathlib import Path
from typing import Any, List, Dict, Optional
import asyncio
"""
⚙️ Configuration Management Demo
===============================
Demonstration of configuration management system with YAML files.
"""


# Import our configuration system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigurationDemo:
    """Demo class for configuration management."""
    
    def __init__(self) -> Any:
        self.config_manager = ConfigManager("configs")
        self.presets = create_preset_configs()
        self.results = {}
    
    def demo_config_loading(self) -> Any:
        """Demo loading different configuration files."""
        logger.info("⚙️ Demo: Configuration Loading")
        logger.info("=" * 50)
        
        results = []
        
        # Load different configurations
        config_files = [
            "configs/default.yaml",
            "configs/small_model.yaml", 
            "configs/large_model.yaml",
            "configs/high_performance.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                logger.info(f"\nLoading configuration: {config_file}")
                
                try:
                    config = self.config_manager.load_config(config_file)
                    
                    result = {
                        'config_file': config_file,
                        'name': config.name,
                        'description': config.description,
                        'model_type': config.model.model_type,
                        'input_dim': config.model.input_dim,
                        'hidden_dim': config.model.hidden_dim,
                        'batch_size': config.training.batch_size,
                        'learning_rate': config.training.learning_rate,
                        'num_epochs': config.training.num_epochs,
                        'dataset_size': config.data.dataset_size,
                        'mixed_precision': config.performance.use_mixed_precision,
                        'gradient_checkpointing': config.performance.gradient_checkpointing
                    }
                    
                    results.append(result)
                    
                    logger.info(f"  Name: {config.name}")
                    logger.info(f"  Description: {config.description}")
                    logger.info(f"  Model: {config.model.model_type} ({config.model.input_dim}→{config.model.hidden_dim})")
                    logger.info(f"  Training: {config.training.batch_size} batch, {config.training.learning_rate} lr, {config.training.num_epochs} epochs")
                    logger.info(f"  Data: {config.data.dataset_size} samples")
                    logger.info(f"  Mixed Precision: {config.performance.use_mixed_precision}")
                    logger.info(f"  Gradient Checkpointing: {config.performance.gradient_checkpointing}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {config_file}: {e}")
                    results.append({
                        'config_file': config_file,
                        'error': str(e)
                    })
            else:
                logger.warning(f"Configuration file not found: {config_file}")
        
        return results
    
    def demo_config_validation(self) -> Any:
        """Demo configuration validation."""
        logger.info("⚙️ Demo: Configuration Validation")
        logger.info("=" * 50)
        
        results = []
        
        # Test different configurations
        configs_to_test = [
            ("default", self.config_manager.create_default_config()),
            ("small", self.presets["small"]),
            ("medium", self.presets["medium"]),
            ("large", self.presets["large"]),
            ("high_performance", self.presets["high_performance"])
        ]
        
        for name, config in configs_to_test:
            logger.info(f"\nValidating {name} configuration")
            
            issues = self.config_manager.validate_config(config)
            
            result = {
                'config_name': name,
                'valid': len(issues) == 0,
                'issues': issues,
                'issue_count': len(issues)
            }
            
            results.append(result)
            
            if issues:
                logger.warning(f"  Found {len(issues)} validation issues:")
                for issue in issues:
                    logger.warning(f"    - {issue}")
            else:
                logger.info("  Configuration is valid")
        
        return results
    
    def demo_config_comparison(self) -> Any:
        """Demo configuration comparison."""
        logger.info("⚙️ Demo: Configuration Comparison")
        logger.info("=" * 50)
        
        # Load configurations
        configs = {}
        for name in ["small", "medium", "large"]:
            if name in self.presets:
                configs[name] = self.presets[name]
        
        results = []
        
        # Compare configurations
        config_names = list(configs.keys())
        for i in range(len(config_names)):
            for j in range(i + 1, len(config_names)):
                config1_name = config_names[i]
                config2_name = config_names[j]
                config1 = configs[config1_name]
                config2 = configs[config2_name]
                
                logger.info(f"\nComparing {config1_name} vs {config2_name}")
                
                differences = self.config_manager.diff_configs(config1, config2)
                
                result = {
                    'config1': config1_name,
                    'config2': config2_name,
                    'differences': differences,
                    'difference_count': len(differences)
                }
                
                results.append(result)
                
                logger.info(f"  Found {len(differences)} differences")
                
                # Show key differences
                key_diffs = []
                for key, diff in differences.items():
                    if any(metric in key for metric in ['input_dim', 'hidden_dim', 'batch_size', 'learning_rate', 'num_epochs']):
                        key_diffs.append((key, diff))
                
                for key, diff in key_diffs[:5]:  # Show top 5 key differences
                    if diff['type'] == 'modified':
                        logger.info(f"    {key}: {diff['old_value']} → {diff['new_value']}")
        
        return results
    
    def demo_experiment_configs(self) -> Any:
        """Demo creating experiment configurations."""
        logger.info("⚙️ Demo: Experiment Configuration Creation")
        logger.info("=" * 50)
        
        # Start with default configuration
        base_config = self.config_manager.create_default_config()
        
        # Define experiments
        experiments = {
            "high_lr": {
                "description": "High learning rate experiment",
                "overrides": {
                    "training.learning_rate": 5e-4,
                    "training.batch_size": 64,
                    "custom.experiment_type": "learning_rate_study"
                }
            },
            "large_batch": {
                "description": "Large batch size experiment", 
                "overrides": {
                    "training.batch_size": 128,
                    "training.gradient_accumulation_steps": 2,
                    "performance.use_mixed_precision": True,
                    "custom.experiment_type": "batch_size_study"
                }
            },
            "memory_optimized": {
                "description": "Memory optimization experiment",
                "overrides": {
                    "performance.gradient_checkpointing": True,
                    "performance.memory_efficient_attention": True,
                    "training.gradient_accumulation_steps": 4,
                    "custom.experiment_type": "memory_optimization"
                }
            },
            "fast_training": {
                "description": "Fast training experiment",
                "overrides": {
                    "training.num_epochs": 20,
                    "training.batch_size": 256,
                    "performance.compile_model": True,
                    "performance.use_mixed_precision": True,
                    "custom.experiment_type": "fast_training"
                }
            }
        }
        
        results = []
        
        for exp_name, exp_config in experiments.items():
            logger.info(f"\nCreating experiment: {exp_name}")
            logger.info(f"Description: {exp_config['description']}")
            
            try:
                # Create experiment configuration
                experiment_config = self.config_manager.create_experiment_config(
                    base_config, exp_name, exp_config['overrides']
                )
                
                result = {
                    'experiment_name': exp_name,
                    'description': exp_config['description'],
                    'overrides': exp_config['overrides'],
                    'config_hash': experiment_config.get_hash(),
                    'success': True
                }
                
                results.append(result)
                
                logger.info(f"  Created experiment configuration")
                logger.info(f"  Config hash: {experiment_config.get_hash()}")
                
                # Show key changes
                for key, value in exp_config['overrides'].items():
                    logger.info(f"  {key}: {value}")
                
            except Exception as e:
                logger.error(f"Failed to create experiment {exp_name}: {e}")
                results.append({
                    'experiment_name': exp_name,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def demo_config_backup_restore(self) -> Any:
        """Demo configuration backup and restore."""
        logger.info("⚙️ Demo: Configuration Backup and Restore")
        logger.info("=" * 50)
        
        # Create a test configuration
        test_config = self.config_manager.create_default_config()
        test_config.name = "test_config"
        test_config.description = "Test configuration for backup/restore demo"
        
        # Modify the configuration
        test_config.training.learning_rate = 2e-4
        test_config.training.batch_size = 64
        test_config.custom['backup_test'] = True
        
        logger.info("Creating test configuration")
        logger.info(f"  Learning rate: {test_config.training.learning_rate}")
        logger.info(f"  Batch size: {test_config.training.batch_size}")
        
        # Create backup
        logger.info("\nCreating backup...")
        backup_path = self.config_manager.backup_config(test_config, "demo_backup")
        logger.info(f"Backup created at: {backup_path}")
        
        # Modify configuration again
        test_config.training.learning_rate = 3e-4
        test_config.training.batch_size = 128
        test_config.custom['modified'] = True
        
        logger.info("Modified configuration")
        logger.info(f"  Learning rate: {test_config.training.learning_rate}")
        logger.info(f"  Batch size: {test_config.training.batch_size}")
        
        # Load backup
        logger.info("\nLoading backup...")
        backup_config = self.config_manager.load_config(backup_path)
        
        logger.info("Backup configuration loaded")
        logger.info(f"  Learning rate: {backup_config.training.learning_rate}")
        logger.info(f"  Batch size: {backup_config.training.batch_size}")
        
        # Compare
        differences = self.config_manager.diff_configs(test_config, backup_config)
        
        result = {
            'backup_path': str(backup_path),
            'differences': differences,
            'difference_count': len(differences)
        }
        
        logger.info(f"\nFound {len(differences)} differences between current and backup")
        
        return result
    
    def demo_yaml_json_conversion(self) -> Any:
        """Demo YAML and JSON conversion."""
        logger.info("⚙️ Demo: YAML and JSON Conversion")
        logger.info("=" * 50)
        
        # Create test configuration
        config = self.config_manager.create_default_config()
        config.name = "conversion_test"
        
        results = []
        
        # Convert to YAML
        logger.info("Converting to YAML...")
        yaml_str = config.to_yaml()
        yaml_path = Path("configs/conversion_test.yaml")
        
        with open(yaml_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(yaml_str)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"YAML saved to: {yaml_path}")
        
        # Convert to JSON
        logger.info("Converting to JSON...")
        json_str = config.to_json()
        json_path = Path("configs/conversion_test.json")
        
        with open(json_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(json_str)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"JSON saved to: {json_path}")
        
        # Load back and compare
        logger.info("Loading configurations back...")
        
        yaml_config = self.config_manager.load_config(yaml_path)
        json_config = self.config_manager.load_config(json_path)
        
        # Compare
        yaml_differences = self.config_manager.diff_configs(config, yaml_config)
        json_differences = self.config_manager.diff_configs(config, json_config)
        
        result = {
            'yaml_path': str(yaml_path),
            'json_path': str(json_path),
            'yaml_differences': yaml_differences,
            'json_differences': json_differences,
            'yaml_identical': len(yaml_differences) == 0,
            'json_identical': len(json_differences) == 0
        }
        
        results.append(result)
        
        logger.info(f"YAML conversion identical: {result['yaml_identical']}")
        logger.info(f"JSON conversion identical: {result['json_identical']}")
        
        # Clean up
        yaml_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        
        return results
    
    def demo_config_templates(self) -> Any:
        """Demo configuration templates."""
        logger.info("⚙️ Demo: Configuration Templates")
        logger.info("=" * 50)
        
        results = []
        
        # Create different templates
        templates = [
            ("basic_template", "Basic configuration template"),
            ("training_template", "Training-focused template"),
            ("inference_template", "Inference-optimized template")
        ]
        
        for template_name, description in templates:
            logger.info(f"\nCreating template: {template_name}")
            
            # Create template configuration
            template_config = self.config_manager.create_default_config()
            template_config.name = template_name
            template_config.description = description
            
            # Customize template
            if "training" in template_name:
                template_config.training.num_epochs = 200
                template_config.training.learning_rate = 5e-5
                template_config.performance.use_mixed_precision = True
                template_config.logging.use_tensorboard = True
            elif "inference" in template_name:
                template_config.performance.optimize_for_inference = True
                template_config.performance.compile_model = True
                template_config.training.num_epochs = 0  # No training
                template_config.custom['inference_optimized'] = True
            
            # Save template
            template_path = self.config_manager.save_config(
                template_config, 
                f"configs/{template_name}.yaml"
            )
            
            result = {
                'template_name': template_name,
                'description': description,
                'template_path': str(template_path),
                'config_hash': template_config.get_hash()
            }
            
            results.append(result)
            
            logger.info(f"  Template saved to: {template_path}")
            logger.info(f"  Config hash: {template_config.get_hash()}")
        
        return results
    
    def run_all_demos(self) -> Any:
        """Run all configuration management demos."""
        logger.info("⚙️ Starting Configuration Management Demos")
        logger.info("=" * 80)
        
        # Run demos
        demos = [
            ("Configuration Loading", self.demo_config_loading),
            ("Configuration Validation", self.demo_config_validation),
            ("Configuration Comparison", self.demo_config_comparison),
            ("Experiment Configuration Creation", self.demo_experiment_configs),
            ("Configuration Backup and Restore", self.demo_config_backup_restore),
            ("YAML and JSON Conversion", self.demo_yaml_json_conversion),
            ("Configuration Templates", self.demo_config_templates)
        ]
        
        for demo_name, demo_func in demos:
            try:
                logger.info(f"\n{'='*20} {demo_name} {'='*20}")
                result = demo_func()
                self.results[demo_name] = result
                logger.info(f"✅ {demo_name} completed successfully")
            except Exception as e:
                logger.error(f"❌ {demo_name} failed: {e}")
                self.results[demo_name] = None
        
        # Summary
        logger.info("\n🎉 All configuration demos completed!")
        logger.info("=" * 80)
        
        successful_demos = sum(1 for result in self.results.values() if result is not None)
        total_demos = len(self.results)
        
        logger.info(f"Successful demos: {successful_demos}/{total_demos}")
        
        # List available configurations
        configs = self.config_manager.list_configs()
        logger.info(f"Available configuration files: {len(configs)}")
        for config_file in configs:
            logger.info(f"  - {config_file.name}")
        
        return self.results

def main():
    """Main function to run the demo."""
    demo = ConfigurationDemo()
    results = demo.run_all_demos()
    
    # Save results
    
    results_path = Path("config_demo_results.json")
    
    # Convert results to serializable format
    serializable_results = {}
    for demo_name, result in results.items():
        if result is not None:
            if isinstance(result, list):
                serializable_results[demo_name] = [
                    {k: v for k, v in r.items() if not k.startswith('_')}
                    for r in result
                ]
            else:
                serializable_results[demo_name] = {
                    k: v for k, v in result.items() if not k.startswith('_')
                }
        else:
            serializable_results[demo_name] = None
    
    with open(results_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")

match __name__:
    case "__main__":
    main() 