from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from ..config_manager import (
        import shutil
        import shutil
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Tests for Configuration Manager
"""

    ConfigManager,
    ConfigValidationError,
    load_config,
    get_model_config,
    get_training_config,
    get_data_config,
    get_evaluation_config
)

class TestConfigManager:
    """Test ConfigManager class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir()
        (self.config_dir / "environments").mkdir()
    
    def teardown_method(self) -> Any:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_config(self, config_data: dict, filename: str = "config.yaml"):
        """Create a test configuration file."""
        config_path = self.config_dir / filename
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        return config_path
    
    def create_env_config(self, env_name: str, config_data: dict):
        """Create an environment-specific configuration file."""
        env_path = self.config_dir / "environments" / f"{env_name}.yaml"
        with open(env_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        return env_path
    
    def test_config_manager_initialization(self) -> Any:
        """Test ConfigManager initialization."""
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        assert config_manager.config_dir == self.config_dir
        assert config_manager.environment == "development"  # Default
        assert config_manager.config_cache == {}
    
    def test_config_manager_initialization_with_environment(self) -> Any:
        """Test ConfigManager initialization with specific environment."""
        config_manager = ConfigManager(
            config_dir=str(self.config_dir),
            environment="production"
        )
        
        assert config_manager.environment == "production"
    
    def test_config_manager_initialization_with_env_var(self) -> Any:
        """Test ConfigManager initialization with environment variable."""
        with patch.dict(os.environ, {"ML_ENVIRONMENT": "staging"}):
            config_manager = ConfigManager(config_dir=str(self.config_dir))
            assert config_manager.environment == "staging"
    
    def test_config_manager_invalid_config_dir(self) -> Any:
        """Test ConfigManager initialization with invalid config directory."""
        with pytest.raises(FileNotFoundError):
            ConfigManager(config_dir="nonexistent_dir")
    
    def test_load_config_basic(self) -> Any:
        """Test basic configuration loading."""
        # Create test configuration
        test_config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        self.create_test_config(test_config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        config = config_manager.load_config()
        
        assert config["app"]["name"] == "test_pipeline"
        assert config["models"]["gpt2"]["model_name"] == "gpt2"
        assert config["training"]["default"]["batch_size"] == 16
        assert config["data"]["default"]["batch_size"] == 32
        assert config["evaluation"]["default"]["batch_size"] == 32
    
    def test_load_config_with_environment_overrides(self) -> Any:
        """Test configuration loading with environment overrides."""
        # Create main configuration
        main_config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        # Create development environment overrides
        dev_config = {
            "training": {
                "default": {
                    "batch_size": 4,  # Override
                    "num_epochs": 2,  # Override
                    "use_mixed_precision": False  # Override
                }
            },
            "data": {
                "default": {
                    "batch_size": 8,  # Override
                    "num_workers": 2  # Override
                }
            }
        }
        
        self.create_test_config(main_config)
        self.create_env_config("development", dev_config)
        
        config_manager = ConfigManager(
            config_dir=str(self.config_dir),
            environment="development"
        )
        config = config_manager.load_config()
        
        # Check that overrides were applied
        assert config["training"]["default"]["batch_size"] == 4
        assert config["training"]["default"]["num_epochs"] == 2
        assert config["training"]["default"]["use_mixed_precision"] == False
        assert config["data"]["default"]["batch_size"] == 8
        assert config["data"]["default"]["num_workers"] == 2
        
        # Check that non-overridden values remain
        assert config["training"]["default"]["learning_rate"] == 1.0e-4
        assert config["models"]["gpt2"]["model_name"] == "gpt2"
    
    def test_load_config_with_main_env_overrides(self) -> Any:
        """Test configuration loading with environment overrides in main config."""
        # Create main configuration with environment overrides
        main_config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            },
            "environments": {
                "development": {
                    "training": {
                        "default": {
                            "batch_size": 4,
                            "num_epochs": 2
                        }
                    }
                }
            }
        }
        
        self.create_test_config(main_config)
        
        config_manager = ConfigManager(
            config_dir=str(self.config_dir),
            environment="development"
        )
        config = config_manager.load_config()
        
        # Check that overrides were applied
        assert config["training"]["default"]["batch_size"] == 4
        assert config["training"]["default"]["num_epochs"] == 2
        
        # Check that non-overridden values remain
        assert config["training"]["default"]["learning_rate"] == 1.0e-4
    
    def test_validation_missing_required_sections(self) -> Any:
        """Test validation with missing required sections."""
        # Create incomplete configuration
        incomplete_config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            }
            # Missing models, training, data, evaluation sections
        }
        
        self.create_test_config(incomplete_config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        with pytest.raises(ConfigValidationError, match="Required configuration section"):
            config_manager.load_config()
    
    def test_validation_missing_app_fields(self) -> Any:
        """Test validation with missing app fields."""
        # Create configuration with missing app fields
        config = {
            "app": {
                "name": "test_pipeline"
                # Missing version and environment
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        self.create_test_config(config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        with pytest.raises(ConfigValidationError, match="Required app field"):
            config_manager.load_config()
    
    def test_validation_invalid_environment(self) -> Any:
        """Test validation with invalid environment."""
        config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "invalid_env"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        self.create_test_config(config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        with pytest.raises(ConfigValidationError, match="Invalid environment"):
            config_manager.load_config()
    
    def test_validation_invalid_model_config(self) -> Any:
        """Test validation with invalid model configuration."""
        config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": "invalid",  # Should be numeric
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        self.create_test_config(config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        with pytest.raises(ConfigValidationError, match="must be numeric"):
            config_manager.load_config()
    
    def test_get_model_config(self) -> Optional[Dict[str, Any]]:
        """Test getting specific model configuration."""
        config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                },
                "bert": {
                    "model_name": "bert-base-uncased",
                    "max_length": 512,
                    "temperature": 1.0,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        self.create_test_config(config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        full_config = config_manager.load_config()
        
        # Get specific model config
        gpt2_config = config_manager.get_model_config("gpt2", full_config)
        assert gpt2_config["model_name"] == "gpt2"
        assert gpt2_config["max_length"] == 512
        
        bert_config = config_manager.get_model_config("bert", full_config)
        assert bert_config["model_name"] == "bert-base-uncased"
        
        # Test non-existent model
        with pytest.raises(ConfigValidationError, match="not found"):
            config_manager.get_model_config("nonexistent", full_config)
    
    def test_get_training_config(self) -> Optional[Dict[str, Any]]:
        """Test getting specific training configuration."""
        config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                },
                "fast": {
                    "model_type": "gpt2",
                    "batch_size": 8,
                    "learning_rate": 2.0e-4,
                    "num_epochs": 2,
                    "use_mixed_precision": False,
                    "use_wandb": False,
                    "use_tensorboard": False
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        self.create_test_config(config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        full_config = config_manager.load_config()
        
        # Get specific training config
        default_config = config_manager.get_training_config("default", full_config)
        assert default_config["batch_size"] == 16
        assert default_config["use_mixed_precision"] == True
        
        fast_config = config_manager.get_training_config("fast", full_config)
        assert fast_config["batch_size"] == 8
        assert fast_config["use_mixed_precision"] == False
        
        # Test non-existent config
        with pytest.raises(ConfigValidationError, match="not found"):
            config_manager.get_training_config("nonexistent", full_config)
    
    def test_resolve_device(self) -> Any:
        """Test device resolution."""
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        # Test auto resolution
        device = config_manager.resolve_device("auto")
        assert device in ["cuda", "cpu"]
        
        # Test explicit devices
        assert config_manager.resolve_device("cuda") == "cuda"
        assert config_manager.resolve_device("cpu") == "cpu"
    
    def test_resolve_torch_dtype(self) -> Any:
        """Test torch dtype resolution."""
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        # Test auto resolution
        dtype = config_manager.resolve_torch_dtype("auto")
        assert dtype in [torch.float16, torch.float32]
        
        # Test explicit dtypes
        assert config_manager.resolve_torch_dtype("float16") == torch.float16
        assert config_manager.resolve_torch_dtype("float32") == torch.float32
        
        # Test invalid dtype
        with pytest.raises(ConfigValidationError, match="Invalid torch_dtype"):
            config_manager.resolve_torch_dtype("invalid")
    
    def test_update_config(self) -> Any:
        """Test configuration updates."""
        config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        self.create_test_config(config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        original_config = config_manager.load_config()
        
        # Update configuration
        updates = {
            "training": {
                "default": {
                    "batch_size": 32,
                    "learning_rate": 2.0e-4
                }
            }
        }
        
        updated_config = config_manager.update_config(updates, original_config)
        
        # Check that updates were applied
        assert updated_config["training"]["default"]["batch_size"] == 32
        assert updated_config["training"]["default"]["learning_rate"] == 2.0e-4
        
        # Check that other values remain unchanged
        assert updated_config["training"]["default"]["num_epochs"] == 5
        assert updated_config["models"]["gpt2"]["model_name"] == "gpt2"
    
    def test_save_config(self) -> Any:
        """Test configuration saving."""
        config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            }
        }
        
        self.create_test_config(config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        original_config = config_manager.load_config()
        
        # Save configuration
        output_filename = "saved_config.yaml"
        config_manager.save_config(original_config, output_filename)
        
        # Check that file was created
        output_path = self.config_dir / output_filename
        assert output_path.exists()
        
        # Load and verify saved configuration
        with open(output_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            saved_config = yaml.safe_load(f)
        
        assert saved_config["app"]["name"] == "test_pipeline"
        assert saved_config["models"]["gpt2"]["model_name"] == "gpt2"
    
    def test_get_config_summary(self) -> Optional[Dict[str, Any]]:
        """Test configuration summary generation."""
        config = {
            "app": {
                "name": "test_pipeline",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "gpt2": {
                    "model_name": "gpt2",
                    "max_length": 512,
                    "temperature": 0.7,
                    "device": "auto"
                },
                "bert": {
                    "model_name": "bert-base-uncased",
                    "max_length": 512,
                    "temperature": 1.0,
                    "device": "auto"
                }
            },
            "training": {
                "default": {
                    "model_type": "gpt2",
                    "batch_size": 16,
                    "learning_rate": 1.0e-4,
                    "num_epochs": 5,
                    "use_mixed_precision": True,
                    "use_wandb": False,
                    "use_tensorboard": True
                },
                "fast": {
                    "model_type": "gpt2",
                    "batch_size": 8,
                    "learning_rate": 2.0e-4,
                    "num_epochs": 2,
                    "use_mixed_precision": False,
                    "use_wandb": False,
                    "use_tensorboard": False
                }
            },
            "data": {
                "default": {
                    "max_length": 512,
                    "batch_size": 32,
                    "num_workers": 4,
                    "pin_memory": True
                }
            },
            "evaluation": {
                "default": {
                    "batch_size": 32,
                    "num_workers": 4,
                    "device": "auto",
                    "save_predictions": True,
                    "save_metrics": True,
                    "generate_plots": True,
                    "generate_report": True
                }
            },
            "ensemble": {
                "default": {
                    "models": [
                        {"name": "gpt2", "weight": 0.6},
                        {"name": "bert", "weight": 0.4}
                    ],
                    "method": "weighted_average"
                }
            },
            "experiment_tracking": {
                "tensorboard": {"enabled": True},
                "wandb": {"enabled": False},
                "mlflow": {"enabled": False}
            },
            "performance": {
                "gpu": {
                    "device": "auto",
                    "mixed_precision": True
                }
            }
        }
        
        self.create_test_config(config)
        
        config_manager = ConfigManager(config_dir=str(self.config_dir))
        full_config = config_manager.load_config()
        
        summary = config_manager.get_config_summary(full_config)
        
        # Check summary structure
        assert "app" in summary
        assert "models" in summary
        assert "training_configs" in summary
        assert "data_configs" in summary
        assert "evaluation_configs" in summary
        assert "ensemble_configs" in summary
        assert "experiment_tracking" in summary
        assert "performance" in summary
        
        # Check specific values
        assert summary["app"]["name"] == "test_pipeline"
        assert "gpt2" in summary["models"]
        assert "bert" in summary["models"]
        assert "default" in summary["training_configs"]
        assert "fast" in summary["training_configs"]
        assert summary["experiment_tracking"]["tensorboard"] == True
        assert summary["experiment_tracking"]["wandb"] == False

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir()
        (self.config_dir / "environments").mkdir()
    
    def teardown_method(self) -> Any:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_config(self, config_data: dict, filename: str = "config.yaml"):
        """Create a test configuration file."""
        config_path = self.config_dir / filename
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        return config_path
    
    @patch('ml.config.config_manager.ConfigManager')
    def test_load_config_function(self, mock_config_manager_class) -> Any:
        """Test load_config convenience function."""
        # Mock config manager
        mock_config_manager = mock_config_manager_class.return_value
        mock_config_manager.load_config.return_value = {"test": "config"}
        
        # Test with default environment
        result = load_config()
        assert result == {"test": "config"}
        mock_config_manager_class.assert_called_with(environment=None)
        
        # Test with specific environment
        result = load_config(environment="production")
        assert result == {"test": "config"}
        mock_config_manager_class.assert_called_with(environment="production")
    
    @patch('ml.config.config_manager.ConfigManager')
    def test_get_model_config_function(self, mock_config_manager_class) -> Optional[Dict[str, Any]]:
        """Test get_model_config convenience function."""
        # Mock config manager
        mock_config_manager = mock_config_manager_class.return_value
        mock_config_manager.load_config.return_value = {"models": {"gpt2": {"test": "config"}}}
        mock_config_manager.get_model_config.return_value = {"test": "config"}
        
        result = get_model_config("gpt2", environment="production")
        assert result == {"test": "config"}
        
        mock_config_manager_class.assert_called_with(environment="production")
        mock_config_manager.load_config.assert_called_once()
        mock_config_manager.get_model_config.assert_called_once_with("gpt2", {"models": {"gpt2": {"test": "config"}}})
    
    @patch('ml.config.config_manager.ConfigManager')
    def test_get_training_config_function(self, mock_config_manager_class) -> Optional[Dict[str, Any]]:
        """Test get_training_config convenience function."""
        # Mock config manager
        mock_config_manager = mock_config_manager_class.return_value
        mock_config_manager.load_config.return_value = {"training": {"default": {"test": "config"}}}
        mock_config_manager.get_training_config.return_value = {"test": "config"}
        
        result = get_training_config("default", environment="production")
        assert result == {"test": "config"}
        
        mock_config_manager_class.assert_called_with(environment="production")
        mock_config_manager.load_config.assert_called_once()
        mock_config_manager.get_training_config.assert_called_once_with("default", {"training": {"default": {"test": "config"}}})
    
    @patch('ml.config.config_manager.ConfigManager')
    def test_get_data_config_function(self, mock_config_manager_class) -> Optional[Dict[str, Any]]:
        """Test get_data_config convenience function."""
        # Mock config manager
        mock_config_manager = mock_config_manager_class.return_value
        mock_config_manager.load_config.return_value = {"data": {"default": {"test": "config"}}}
        mock_config_manager.get_data_config.return_value = {"test": "config"}
        
        result = get_data_config("default", environment="production")
        assert result == {"test": "config"}
        
        mock_config_manager_class.assert_called_with(environment="production")
        mock_config_manager.load_config.assert_called_once()
        mock_config_manager.get_data_config.assert_called_once_with("default", {"data": {"default": {"test": "config"}}})
    
    @patch('ml.config.config_manager.ConfigManager')
    def test_get_evaluation_config_function(self, mock_config_manager_class) -> Optional[Dict[str, Any]]:
        """Test get_evaluation_config convenience function."""
        # Mock config manager
        mock_config_manager = mock_config_manager_class.return_value
        mock_config_manager.load_config.return_value = {"evaluation": {"default": {"test": "config"}}}
        mock_config_manager.get_evaluation_config.return_value = {"test": "config"}
        
        result = get_evaluation_config("default", environment="production")
        assert result == {"test": "config"}
        
        mock_config_manager_class.assert_called_with(environment="production")
        mock_config_manager.load_config.assert_called_once()
        mock_config_manager.get_evaluation_config.assert_called_once_with("default", {"evaluation": {"default": {"test": "config"}}})

match __name__:
    case "__main__":
    pytest.main([__file__]) 