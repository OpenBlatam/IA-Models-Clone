"""
⚙️ YAML Configuration System Example

This script demonstrates how to use YAML-based configuration management.
Implements the convention: "Use configuration files (e.g., YAML) for hyperparameters and model settings."
"""

import os
import sys
from pathlib import Path
import logging

# Add the modular_structure directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config_manager import ConfigManager
from configs.model_config import ModelConfig, CLASSIFICATION_CONFIGS, TRANSFORMER_CONFIGS
from configs.training_config import TrainingConfig, TRAINING_CONFIGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_yaml_loading():
    """Demonstrate loading configurations from YAML files."""
    logger.info("🔧 Demonstrating YAML Configuration Loading")
    logger.info("=" * 50)
    
    # Initialize configuration manager
    config_manager = ConfigManager("configs")
    
    try:
        # Load model configuration from YAML
        logger.info("\n📁 Loading Model Configuration from YAML...")
        model_config = config_manager.load_config_class(
            ModelConfig, 
            "examples/model_classification.yaml"
        )
        logger.info(f"✅ Model Config Loaded: {model_config.architecture}")
        logger.info(f"   Input Size: {model_config.input_size}")
        logger.info(f"   Output Size: {model_config.output_size}")
        logger.info(f"   Device: {model_config.device}")
        
        # Load training configuration from YAML
        logger.info("\n📁 Loading Training Configuration from YAML...")
        training_config = config_manager.load_config_class(
            TrainingConfig,
            "examples/training_standard.yaml"
        )
        logger.info(f"✅ Training Config Loaded: {training_config.optimizer}")
        logger.info(f"   Epochs: {training_config.num_epochs}")
        logger.info(f"   Batch Size: {training_config.batch_size}")
        logger.info(f"   Learning Rate: {training_config.learning_rate}")
        
        # Load complete experiment configuration
        logger.info("\n📁 Loading Complete Experiment Configuration...")
        experiment_config = config_manager.load_yaml("examples/experiment_complete.yaml")
        logger.info(f"✅ Experiment Config Loaded: {experiment_config['experiment']['name']}")
        logger.info(f"   Model Type: {experiment_config['model']['model_type']}")
        logger.info(f"   Dataset: {experiment_config['data']['dataset_name']}")
        
        return model_config, training_config, experiment_config
        
    except Exception as e:
        logger.error(f"❌ Error loading configurations: {e}")
        raise


def demonstrate_environment_specific_configs():
    """Demonstrate environment-specific configuration loading."""
    logger.info("\n🌍 Demonstrating Environment-Specific Configurations")
    logger.info("=" * 50)
    
    config_manager = ConfigManager("configs")
    
    try:
        # Set environment variable for demonstration
        os.environ['ML_ENVIRONMENT'] = 'dev'
        
        # Load base configuration with environment overrides
        logger.info("Loading configuration with development overrides...")
        config = config_manager.load_environment_config(
            "examples/training_standard.yaml",
            environment="dev"
        )
        
        logger.info(f"✅ Environment Config Loaded:")
        logger.info(f"   Epochs: {config['num_epochs']} (dev override)")
        logger.info(f"   Batch Size: {config['batch_size']} (dev override)")
        logger.info(f"   Learning Rate: {config['learning_rate']} (dev override)")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error loading environment config: {e}")
        raise


def demonstrate_config_merging():
    """Demonstrate configuration merging capabilities."""
    logger.info("\n🔀 Demonstrating Configuration Merging")
    logger.info("=" * 50)
    
    config_manager = ConfigManager("configs")
    
    try:
        # Base configuration
        base_config = {
            "model": {
                "architecture": "resnet18",
                "num_layers": 18,
                "dropout_rate": 0.1
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        }
        
        # Override configuration
        override_config = {
            "model": {
                "dropout_rate": 0.2,  # Override this
                "new_param": "added"   # Add this
            },
            "training": {
                "epochs": 50,          # Override this
                "new_setting": True    # Add this
            },
            "evaluation": {            # Add new section
                "metrics": ["accuracy", "f1"]
            }
        }
        
        # Merge configurations
        merged_config = config_manager.merge_configs(base_config, override_config)
        
        logger.info("✅ Configuration Merged Successfully:")
        logger.info(f"   Model Architecture: {merged_config['model']['architecture']} (kept)")
        logger.info(f"   Model Dropout: {merged_config['model']['dropout_rate']} (overridden)")
        logger.info(f"   Model New Param: {merged_config['model']['new_param']} (added)")
        logger.info(f"   Training Epochs: {merged_config['training']['epochs']} (overridden)")
        logger.info(f"   Training Batch Size: {merged_config['training']['batch_size']} (kept)")
        logger.info(f"   Evaluation Section: {merged_config['evaluation']} (added)")
        
        return merged_config
        
    except Exception as e:
        logger.error(f"❌ Error merging configurations: {e}")
        raise


def demonstrate_config_saving():
    """Demonstrate saving configurations to YAML files."""
    logger.info("\n💾 Demonstrating Configuration Saving")
    logger.info("=" * 50)
    
    config_manager = ConfigManager("configs")
    
    try:
        # Create a custom model configuration
        custom_config = ModelConfig(
            model_type="classification",
            architecture="custom_cnn",
            input_size=(3, 64, 64),
            output_size=5,
            hidden_size=256,
            num_layers=4,
            dropout_rate=0.25,
            pretrained=False,
            mixed_precision=True
        )
        
        # Save to YAML file
        save_path = "examples/custom_model_config.yaml"
        config_manager.save_config_class(custom_config, save_path)
        
        logger.info(f"✅ Custom configuration saved to: {save_path}")
        
        # Load it back to verify
        loaded_config = config_manager.load_config_class(ModelConfig, save_path)
        logger.info(f"✅ Configuration loaded back successfully:")
        logger.info(f"   Architecture: {loaded_config.architecture}")
        logger.info(f"   Input Size: {loaded_config.input_size}")
        logger.info(f"   Hidden Size: {loaded_config.hidden_size}")
        
        return loaded_config
        
    except Exception as e:
        logger.error(f"❌ Error saving/loading configuration: {e}")
        raise


def demonstrate_environment_variables():
    """Demonstrate environment variable substitution in YAML."""
    logger.info("\n🌐 Demonstrating Environment Variable Substitution")
    logger.info("=" * 50)
    
    config_manager = ConfigManager("configs")
    
    try:
        # Set some environment variables
        os.environ['CUSTOM_BATCH_SIZE'] = '64'
        os.environ['CUSTOM_LEARNING_RATE'] = '0.005'
        os.environ['MODEL_SAVE_PATH'] = './custom_models'
        
        # Create a YAML content with environment variables
        yaml_content = """
# Configuration with environment variables
training:
  batch_size: ${CUSTOM_BATCH_SIZE:32}
  learning_rate: ${CUSTOM_LEARNING_RATE:0.001}
  epochs: ${CUSTOM_EPOCHS:100}  # Will use default since not set

model:
  save_path: "${MODEL_SAVE_PATH:./models}"
  
experiment:
  name: "${EXPERIMENT_NAME:default_experiment}"
        """
        
        # Save to temporary file
        temp_config_path = Path("configs/examples/temp_env_config.yaml")
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_config_path, 'w') as f:
            f.write(yaml_content)
        
        # Load with environment variable substitution
        config = config_manager.load_yaml("examples/temp_env_config.yaml")
        
        logger.info("✅ Environment variables substituted:")
        logger.info(f"   Batch Size: {config['training']['batch_size']} (from env)")
        logger.info(f"   Learning Rate: {config['training']['learning_rate']} (from env)")
        logger.info(f"   Epochs: {config['training']['epochs']} (default used)")
        logger.info(f"   Model Save Path: {config['model']['save_path']} (from env)")
        logger.info(f"   Experiment Name: {config['experiment']['name']} (default used)")
        
        # Clean up
        temp_config_path.unlink()
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error with environment variables: {e}")
        raise


def demonstrate_predefined_configs():
    """Demonstrate using predefined configuration templates."""
    logger.info("\n📋 Demonstrating Predefined Configuration Templates")
    logger.info("=" * 50)
    
    try:
        # Show available model configurations
        logger.info("Available Model Configurations:")
        for name, config in CLASSIFICATION_CONFIGS.items():
            logger.info(f"  - {name}: {config.architecture} ({config.input_size})")
        
        # Show available training configurations
        logger.info("\nAvailable Training Configurations:")
        for name, config in TRAINING_CONFIGS.items():
            logger.info(f"  - {name}: {config.num_epochs} epochs, LR={config.learning_rate}")
        
        # Use a predefined configuration
        logger.info("\n✅ Using predefined 'fine_tuning' configuration:")
        fine_tune_config = TRAINING_CONFIGS["fine_tuning"]
        logger.info(f"   Epochs: {fine_tune_config.num_epochs}")
        logger.info(f"   Learning Rate: {fine_tune_config.learning_rate}")
        logger.info(f"   Optimizer: {fine_tune_config.optimizer}")
        logger.info(f"   Warmup Epochs: {fine_tune_config.warmup_epochs}")
        
        return fine_tune_config
        
    except Exception as e:
        logger.error(f"❌ Error with predefined configs: {e}")
        raise


def main():
    """Main function to demonstrate YAML configuration system."""
    logger.info("⚙️ YAML Configuration System Demonstration")
    logger.info("=" * 60)
    
    try:
        # Create configs directory structure
        configs_dir = Path("configs/examples")
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Run all demonstrations
        demonstrate_yaml_loading()
        demonstrate_environment_specific_configs()
        demonstrate_config_merging()
        demonstrate_config_saving()
        demonstrate_environment_variables()
        demonstrate_predefined_configs()
        
        logger.info("\n🎉 All YAML configuration demonstrations completed successfully!")
        logger.info("\n💡 Key Benefits of YAML Configuration System:")
        logger.info("  ✅ Centralized hyperparameter management")
        logger.info("  ✅ Environment-specific overrides")
        logger.info("  ✅ Version control friendly")
        logger.info("  ✅ Human-readable format")
        logger.info("  ✅ Environment variable support")
        logger.info("  ✅ Configuration validation")
        logger.info("  ✅ Easy experimentation")
        logger.info("  ✅ Professional ML workflow")
        
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()






