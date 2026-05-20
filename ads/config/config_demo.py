"""
Configuration System Demo for the ads feature.

This demo showcases the entire unified configuration system, including:
- Basic and optimized settings
- Configuration models and management
- Provider configurations
- YAML-based configuration persistence
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from .settings import Settings, OptimizedSettings, get_settings, get_optimized_settings
from .manager import ConfigManager, ConfigType
from .models import (
    ModelConfig, TrainingConfig, DataConfig, ExperimentConfig,
    OptimizationConfig, DeploymentConfig, ProjectConfig
)
from .providers import (
    get_llm_config, get_embeddings_config, get_redis_config, 
    get_database_config, get_storage_config, get_api_config
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationSystemDemo:
    """Demonstrates the unified configuration system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.config_manager = ConfigManager("./demo_configs")
        self.demo_project = "ads_demo_project"
    
    async def run_comprehensive_demo(self):
        """Run the complete configuration system demonstration."""
        print("🚀 Starting Configuration System Demo")
        print("=" * 60)
        
        try:
            # Demo 1: Basic and Optimized Settings
            await self._demo_settings()
            
            # Demo 2: Configuration Models
            await self._demo_configuration_models()
            
            # Demo 3: Configuration Management
            await self._demo_configuration_management()
            
            # Demo 4: Provider Configurations
            await self._demo_provider_configurations()
            
            # Demo 5: Advanced Configuration Features
            await self._demo_advanced_features()
            
            # Demo 6: System Integration
            await self._demo_system_integration()
            
            print("\n✅ Configuration System Demo Completed Successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")
    
    async def _demo_settings(self):
        """Demonstrate basic and optimized settings."""
        print("\n📋 Demo 1: Basic and Optimized Settings")
        print("-" * 40)
        
        # Basic settings
        basic_settings = get_settings()
        print(f"Basic Settings:")
        print(f"  - Environment: {basic_settings.environment}")
        print(f"  - Host: {basic_settings.host}")
        print(f"  - Port: {basic_settings.port}")
        print(f"  - Database URL: {basic_settings.database_url}")
        print(f"  - Cache TTL: {basic_settings.cache_ttl}s")
        
        # Optimized settings
        optimized_settings = get_optimized_settings()
        print(f"\nOptimized Settings:")
        print(f"  - Environment: {optimized_settings.environment}")
        print(f"  - Workers: {optimized_settings.workers}")
        print(f"  - Database Pool Size: {optimized_settings.database_pool_size}")
        print(f"  - Redis Max Connections: {optimized_settings.redis_max_connections}")
        print(f"  - Rate Limits: {optimized_settings.rate_limits}")
        print(f"  - Background Task Workers: {optimized_settings.background_task_workers}")
    
    async def _demo_configuration_models(self):
        """Demonstrate configuration models."""
        print("\n🏗️  Demo 2: Configuration Models")
        print("-" * 40)
        
        # Create sample configurations
        model_config = ModelConfig(
            name="demo_model",
            type="transformer",
            architecture="bert-base",
            input_size=768,
            output_size=10,
            hidden_sizes=[512, 256],
            dropout_rate=0.1
        )
        
        training_config = TrainingConfig(
            batch_size=64,
            learning_rate=1e-4,
            epochs=50,
            mixed_precision=True,
            gradient_accumulation_steps=2
        )
        
        data_config = DataConfig(
            train_data_path="./data/train",
            val_data_path="./data/val",
            batch_size=64,
            num_workers=8,
            augmentation_enabled=True
        )
        
        print(f"Model Config:")
        print(f"  - Name: {model_config.name}")
        print(f"  - Architecture: {model_config.architecture}")
        print(f"  - Input Size: {model_config.input_size}")
        print(f"  - Hidden Sizes: {model_config.hidden_sizes}")
        
        print(f"\nTraining Config:")
        print(f"  - Batch Size: {training_config.batch_size}")
        print(f"  - Learning Rate: {training_config.learning_rate}")
        print(f"  - Mixed Precision: {training_config.mixed_precision}")
        
        print(f"\nData Config:")
        print(f"  - Train Path: {data_config.train_data_path}")
        print(f"  - Batch Size: {data_config.batch_size}")
        print(f"  - Augmentation: {data_config.augmentation_enabled}")
    
    async def _demo_configuration_management(self):
        """Demonstrate configuration management."""
        print("\n⚙️  Demo 3: Configuration Management")
        print("-" * 40)
        
        # Create default configurations
        print("Creating default configurations...")
        created_files = self.config_manager.create_default_configs(self.demo_project)
        print(f"Created {len(created_files)} configuration files:")
        for config_type, filepath in created_files.items():
            print(f"  - {config_type}: {filepath}")
        
        # Load all configurations
        print("\nLoading all configurations...")
        configs = self.config_manager.load_all_configs(self.demo_project)
        print(f"Loaded {len(configs)} configurations:")
        for config_type, config in configs.items():
            print(f"  - {config_type}: {config.name}")
        
        # Update a configuration
        print("\nUpdating training configuration...")
        updates = {
            "batch_size": 128,
            "learning_rate": 5e-5,
            "epochs": 100
        }
        success = self.config_manager.update_config(
            self.demo_project, ConfigType.TRAINING, updates
        )
        print(f"Update successful: {success}")
        
        # Get configuration info
        print("\nGetting configuration information...")
        config_info = self.config_manager.get_config_info(self.demo_project)
        print(f"Project: {config_info['project_name']}")
        print(f"Config Directory: {config_info['config_dir']}")
        print(f"Last Updated: {config_info['last_updated']}")
    
    async def _demo_provider_configurations(self):
        """Demonstrate provider configurations."""
        print("\n🔌 Demo 4: Provider Configurations")
        print("-" * 40)
        
        try:
            # LLM configuration
            print("LLM Configuration:")
            llm_config = get_llm_config()
            print(f"  - Model: {llm_config.model_name}")
            print(f"  - Temperature: {llm_config.temperature}")
            print(f"  - Max Tokens: {llm_config.max_tokens}")
        except ImportError:
            print("  - LLM configuration not available (langchain_openai not installed)")
        
        # Redis configuration
        print("\nRedis Configuration:")
        redis_config = get_redis_config()
        print(f"  - URL: {redis_config['url']}")
        print(f"  - Max Connections: {redis_config['max_connections']}")
        print(f"  - Socket Timeout: {redis_config['socket_timeout']}s")
        
        # Database configuration
        print("\nDatabase Configuration:")
        db_config = get_database_config()
        print(f"  - URL: {db_config['url']}")
        print(f"  - Pool Size: {db_config['pool_size']}")
        print(f"  - Max Overflow: {db_config['max_overflow']}")
        
        # Storage configuration
        print("\nStorage Configuration:")
        storage_config = get_storage_config()
        print(f"  - Path: {storage_config['storage_path']}")
        print(f"  - Max File Size: {storage_config['max_file_size'] / (1024*1024):.1f}MB")
        print(f"  - Allowed Types: {', '.join(storage_config['allowed_file_types'])}")
        
        # API configuration
        print("\nAPI Configuration:")
        api_config = get_api_config()
        print(f"  - Host: {api_config['host']}")
        print(f"  - Port: {api_config['port']}")
        print(f"  - Workers: {api_config['workers']}")
        print(f"  - Max Requests: {api_config['max_requests']}")
    
    async def _demo_advanced_features(self):
        """Demonstrate advanced configuration features."""
        print("\n🚀 Demo 5: Advanced Configuration Features")
        print("-" * 40)
        
        # Configuration validation
        print("Validating configurations...")
        configs = self.config_manager.load_all_configs(self.demo_project)
        
        for config_type, config in configs.items():
            validation_result = self.config_manager.validate_config(config, config_type)
            print(f"\n{config_type.title()} Validation:")
            print(f"  - Valid: {validation_result['is_valid']}")
            if validation_result['errors']:
                print(f"  - Errors: {validation_result['errors']}")
            if validation_result['warnings']:
                print(f"  - Warnings: {validation_result['warnings']}")
        
        # Configuration caching
        print("\nTesting configuration caching...")
        start_time = asyncio.get_event_loop().time()
        configs1 = self.config_manager.load_all_configs(self.demo_project)
        time1 = asyncio.get_event_loop().time() - start_time
        
        start_time = asyncio.get_event_loop().time()
        configs2 = self.config_manager.load_all_configs(self.demo_project)
        time2 = asyncio.get_event_loop().time() - start_time
        
        print(f"  - First load: {time1:.4f}s")
        print(f"  - Cached load: {time2:.4f}s")
        print(f"  - Cache speedup: {time1/time2:.1f}x")
    
    async def _demo_system_integration(self):
        """Demonstrate system integration."""
        print("\n🔗 Demo 6: System Integration")
        print("-" * 40)
        
        # Create a complete project configuration
        project_config = ProjectConfig(
            name=self.demo_project,
            version="1.0.0",
            description="Demo project for configuration system",
            author="Configuration Demo",
            project_root="./demo_project",
            config_dir="./demo_configs",
            output_dir="./demo_outputs",
            logs_dir="./demo_logs"
        )
        
        # Save project configuration
        project_file = Path(f"./demo_configs/{self.demo_project}/project_config.yaml")
        success = self.config_manager.save_config(
            project_config, project_file, ConfigType.PROJECT
        )
        print(f"Project configuration saved: {success}")
        
        # Demonstrate configuration hierarchy
        print("\nConfiguration Hierarchy:")
        print("  - Project Level: Project configuration")
        print("  - Model Level: Model, training, data configurations")
        print("  - System Level: API, deployment, optimization configurations")
        print("  - Provider Level: LLM, database, storage configurations")
        
        # Show configuration file structure
        print("\nConfiguration File Structure:")
        config_dir = Path("./demo_configs")
        if config_dir.exists():
            for item in config_dir.rglob("*.yaml"):
                relative_path = item.relative_to(config_dir)
                print(f"  - {relative_path}")
    
    def cleanup_demo(self):
        """Clean up demo files and directories."""
        try:
            # Clean up configuration files
            self.config_manager.cleanup_project(self.demo_project)
            
            # Remove demo directories
            demo_dirs = ["./demo_configs", "./demo_project", "./demo_outputs", "./demo_logs"]
            for demo_dir in demo_dirs:
                demo_path = Path(demo_dir)
                if demo_path.exists():
                    import shutil
                    shutil.rmtree(demo_path)
                    print(f"Cleaned up: {demo_dir}")
            
            print("Demo cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            print(f"Cleanup failed: {e}")


async def main():
    """Main demo function."""
    demo = ConfigurationSystemDemo()
    
    try:
        await demo.run_comprehensive_demo()
    finally:
        # Clean up demo files
        demo.cleanup_demo()


if __name__ == "__main__":
    asyncio.run(main())
