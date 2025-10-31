"""
⚙️ YAML Configuration Integration with Experiment Tracking

This script demonstrates how to integrate YAML-based configuration with the existing experiment tracking system.
Implements the convention: "Use configuration files (e.g., YAML) for hyperparameters and model settings."
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add modular structure to path
modular_path = Path(__file__).parent / "modular_structure"
sys.path.append(str(modular_path))

from configs.config_manager import ConfigManager
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from experiment_tracking import ExperimentConfig, create_experiment_tracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_yaml_enhanced_experiment_config(
    model_config_path: str = "configs/examples/model_classification.yaml",
    training_config_path: str = "configs/examples/training_standard.yaml"
) -> ExperimentConfig:
    """
    Create an enhanced experiment configuration using YAML files.
    
    Args:
        model_config_path: Path to model configuration YAML file
        training_config_path: Path to training configuration YAML file
        
    Returns:
        Enhanced ExperimentConfig with YAML-loaded settings
    """
    logger.info("🔧 Creating YAML-Enhanced Experiment Configuration")
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager(modular_path / "configs")
        
        # Load model configuration from YAML
        model_config = config_manager.load_config_class(ModelConfig, model_config_path)
        logger.info(f"✅ Model config loaded: {model_config.architecture}")
        
        # Load training configuration from YAML  
        training_config = config_manager.load_config_class(TrainingConfig, training_config_path)
        logger.info(f"✅ Training config loaded: {training_config.optimizer}")
        
        # Create enhanced experiment configuration
        experiment_config = ExperimentConfig(
            # Basic experiment info
            experiment_name=f"yaml_enhanced_{model_config.architecture}_{training_config.optimizer}",
            project_name=training_config.project_name,
            run_name=f"run_{model_config.model_type}_{training_config.num_epochs}epochs",
            tags=training_config.tags + [model_config.architecture, training_config.optimizer],
            notes=f"YAML-configured experiment: {model_config.architecture} with {training_config.optimizer}",
            
            # Tracking settings from training config
            enable_tensorboard=True,
            enable_wandb=True,
            log_interval=training_config.log_interval,
            save_interval=training_config.save_interval,
            
            # Enhanced logging based on configs
            log_metrics=True,
            log_hyperparameters=True,
            log_model_architecture=True,
            log_gradients=training_config.gradient_clip_norm is not None,
            log_images=True,
            log_text=True,
            
            # File paths
            tensorboard_dir="runs/yaml_enhanced_tensorboard",
            model_save_dir="models/yaml_enhanced",
            config_save_dir="configs/yaml_enhanced",
            
            # Advanced settings based on training config
            log_gradient_norms=training_config.gradient_clip_norm is not None,
            log_nan_inf_counts=training_config.detect_anomaly,
            log_clipping_stats=training_config.gradient_clip_norm is not None,
        )
        
        logger.info(f"✅ Enhanced experiment config created: {experiment_config.experiment_name}")
        return experiment_config
        
    except Exception as e:
        logger.error(f"❌ Error creating YAML-enhanced config: {e}")
        raise


def demonstrate_yaml_experiment_workflow():
    """Demonstrate complete workflow with YAML configurations."""
    logger.info("🚀 Demonstrating YAML Configuration Workflow")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create YAML-enhanced experiment configuration
        logger.info("\n📋 Step 1: Creating YAML-Enhanced Configuration")
        experiment_config = create_yaml_enhanced_experiment_config()
        
        # Step 2: Create experiment tracker with enhanced config
        logger.info("\n🔬 Step 2: Creating Enhanced Experiment Tracker")
        tracker = create_experiment_tracker(experiment_config)
        logger.info("✅ Experiment tracker created with YAML configuration")
        
        # Step 3: Load and log YAML configurations as hyperparameters
        logger.info("\n📊 Step 3: Logging YAML Configurations as Hyperparameters")
        
        config_manager = ConfigManager(modular_path / "configs")
        
        # Load configurations as dictionaries for logging
        model_config_dict = config_manager.load_yaml("examples/model_classification.yaml")
        training_config_dict = config_manager.load_yaml("examples/training_standard.yaml")
        
        # Create comprehensive hyperparameters dictionary
        hyperparameters = {
            # Model hyperparameters
            "model_type": model_config_dict["model_type"],
            "architecture": model_config_dict["architecture"],
            "input_size": model_config_dict["input_size"],
            "output_size": model_config_dict["output_size"],
            "hidden_size": model_config_dict["hidden_size"],
            "dropout_rate": model_config_dict["dropout_rate"],
            "pretrained": model_config_dict["pretrained"],
            "mixed_precision": model_config_dict["mixed_precision"],
            
            # Training hyperparameters
            "num_epochs": training_config_dict["num_epochs"],
            "batch_size": training_config_dict["batch_size"],
            "learning_rate": training_config_dict["learning_rate"],
            "weight_decay": training_config_dict["weight_decay"],
            "optimizer": training_config_dict["optimizer"],
            "scheduler": training_config_dict["scheduler"],
            "gradient_clip_norm": training_config_dict["gradient_clip_norm"],
            "early_stopping": training_config_dict["early_stopping"],
            "patience": training_config_dict["patience"],
            
            # Configuration metadata
            "config_source": "yaml_files",
            "model_config_file": "model_classification.yaml",
            "training_config_file": "training_standard.yaml"
        }
        
        # Log hyperparameters to experiment tracker
        tracker.log_hyperparameters(hyperparameters)
        logger.info("✅ YAML configurations logged as hyperparameters")
        
        # Step 4: Demonstrate environment-specific configuration loading
        logger.info("\n🌍 Step 4: Environment-Specific Configuration")
        
        # Load development overrides
        try:
            dev_config = config_manager.load_environment_config(
                "examples/training_standard.yaml",
                environment="dev"
            )
            logger.info("✅ Development configuration loaded:")
            logger.info(f"   Epochs: {dev_config['num_epochs']} (dev override)")
            logger.info(f"   Batch Size: {dev_config['batch_size']} (dev override)")
            logger.info(f"   Learning Rate: {dev_config['learning_rate']} (dev override)")
        except Exception as e:
            logger.warning(f"⚠️ Could not load dev overrides: {e}")
        
        # Step 5: Save complete experiment configuration
        logger.info("\n💾 Step 5: Saving Complete Experiment Configuration")
        
        complete_config = {
            "experiment": {
                "name": experiment_config.experiment_name,
                "project": experiment_config.project_name,
                "tags": experiment_config.tags,
                "notes": experiment_config.notes
            },
            "model": model_config_dict,
            "training": training_config_dict,
            "tracking": {
                "tensorboard_enabled": experiment_config.enable_tensorboard,
                "wandb_enabled": experiment_config.enable_wandb,
                "log_interval": experiment_config.log_interval,
                "save_interval": experiment_config.save_interval
            }
        }
        
        # Save complete configuration
        complete_config_path = "configs/complete_experiment_config.yaml"
        config_manager.save_yaml(complete_config, complete_config_path)
        logger.info(f"✅ Complete experiment configuration saved: {complete_config_path}")
        
        # Step 6: Generate configuration summary
        logger.info("\n📋 Step 6: Configuration Summary")
        summary = config_manager.get_config_summary(complete_config)
        logger.info(f"Configuration Summary:\n{summary}")
        
        # Clean up
        tracker.close()
        logger.info("✅ Experiment tracker closed")
        
        return complete_config
        
    except Exception as e:
        logger.error(f"❌ Error in YAML workflow: {e}")
        raise


def create_environment_specific_configs():
    """Create environment-specific configuration examples."""
    logger.info("\n🌍 Creating Environment-Specific Configurations")
    logger.info("=" * 50)
    
    config_manager = ConfigManager(modular_path / "configs")
    
    try:
        # Development environment configuration
        dev_overrides = {
            "num_epochs": 5,
            "batch_size": 16,
            "log_interval": 5,
            "eval_interval": 1,
            "early_stopping": False,
            "num_workers": 2,
            "experiment_name": "dev_quick_test",
            "tags": ["development", "quick_test"]
        }
        
        config_manager.save_yaml(
            dev_overrides, 
            "examples/training_standard_dev.yaml"
        )
        logger.info("✅ Development configuration created")
        
        # Production environment configuration
        prod_overrides = {
            "num_epochs": 200,
            "batch_size": 64,
            "log_interval": 200,
            "eval_interval": 5,
            "save_interval": 20,
            "early_stopping": True,
            "patience": 30,
            "num_workers": 8,
            "mixed_precision": True,
            "experiment_name": "prod_full_training",
            "tags": ["production", "full_training"]
        }
        
        config_manager.save_yaml(
            prod_overrides,
            "examples/training_standard_prod.yaml"
        )
        logger.info("✅ Production configuration created")
        
        # Hyperparameter tuning configuration
        tuning_overrides = {
            "num_epochs": 50,
            "early_stopping": True,
            "patience": 10,
            "save_best_only": True,
            "experiment_name": "hyperparameter_tuning",
            "tags": ["tuning", "optimization"]
        }
        
        config_manager.save_yaml(
            tuning_overrides,
            "examples/training_tuning.yaml"
        )
        logger.info("✅ Hyperparameter tuning configuration created")
        
    except Exception as e:
        logger.error(f"❌ Error creating environment configs: {e}")
        raise


def main():
    """Main function to demonstrate YAML configuration integration."""
    logger.info("⚙️ YAML Configuration Integration with Experiment Tracking")
    logger.info("=" * 70)
    
    try:
        # Ensure configs directory exists
        configs_dir = modular_path / "configs" / "examples"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment-specific configurations
        create_environment_specific_configs()
        
        # Run the complete workflow demonstration
        complete_config = demonstrate_yaml_experiment_workflow()
        
        logger.info("\n🎉 YAML Configuration Integration Demonstration Completed!")
        logger.info("\n💡 Key Benefits Achieved:")
        logger.info("  ✅ Centralized hyperparameter management via YAML")
        logger.info("  ✅ Environment-specific configuration overrides")
        logger.info("  ✅ Integration with existing experiment tracking")
        logger.info("  ✅ Version-controllable configuration files")
        logger.info("  ✅ Professional ML workflow with YAML configs")
        logger.info("  ✅ Easy experimentation and hyperparameter tuning")
        
        logger.info("\n📁 Generated Files:")
        logger.info("  - configs/examples/training_standard_dev.yaml")
        logger.info("  - configs/examples/training_standard_prod.yaml")
        logger.info("  - configs/examples/training_tuning.yaml")
        logger.info("  - configs/complete_experiment_config.yaml")
        
        logger.info(f"\n📊 Experiment Name: {complete_config['experiment']['name']}")
        logger.info(f"📊 Project: {complete_config['experiment']['project']}")
        logger.info(f"📊 Tags: {', '.join(complete_config['experiment']['tags'])}")
        
    except Exception as e:
        logger.error(f"❌ Error during integration demonstration: {e}")
        raise


if __name__ == "__main__":
    main()






