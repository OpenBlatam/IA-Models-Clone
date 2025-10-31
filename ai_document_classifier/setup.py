"""
Setup Script for AI Document Classifier
=======================================

Setup and configuration script for the enhanced AI Document Classifier system.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDocumentClassifierSetup:
    """Setup and configuration manager for AI Document Classifier"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize setup manager
        
        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.config_dir = self.base_dir / "config"
        self.templates_dir = self.base_dir / "templates"
        self.models_dir = self.base_dir / "models"
        self.cache_dir = self.base_dir / "cache"
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config_dir,
            self.templates_dir,
            self.models_dir,
            self.cache_dir,
            self.base_dir / "logs",
            self.base_dir / "data"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def install_dependencies(self, requirements_file: str = "requirements.txt"):
        """
        Install Python dependencies
        
        Args:
            requirements_file: Path to requirements file
        """
        requirements_path = self.base_dir / requirements_file
        
        if not requirements_path.exists():
            logger.error(f"Requirements file not found: {requirements_path}")
            return False
        
        try:
            logger.info("Installing Python dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ], check=True)
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            import nltk
            
            logger.info("Downloading NLTK data...")
            nltk_data = [
                'punkt',
                'stopwords',
                'wordnet',
                'averaged_perceptron_tagger',
                'maxent_ne_chunker',
                'words'
            ]
            
            for data in nltk_data:
                try:
                    nltk.download(data, quiet=True)
                    logger.info(f"Downloaded NLTK data: {data}")
                except Exception as e:
                    logger.warning(f"Failed to download NLTK data {data}: {e}")
            
            return True
        except ImportError:
            logger.warning("NLTK not available, skipping NLTK data download")
            return False
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
            return False
    
    def download_spacy_model(self, model_name: str = "en_core_web_sm"):
        """
        Download SpaCy model
        
        Args:
            model_name: Name of the SpaCy model to download
        """
        try:
            import spacy
            
            logger.info(f"Downloading SpaCy model: {model_name}")
            subprocess.run([
                sys.executable, "-m", "spacy", "download", model_name
            ], check=True)
            logger.info(f"SpaCy model {model_name} downloaded successfully")
            return True
        except ImportError:
            logger.warning("SpaCy not available, skipping model download")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download SpaCy model: {e}")
            return False
    
    def create_environment_file(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Create environment configuration file
        
        Args:
            api_keys: Dictionary of API keys to configure
        """
        env_file = self.base_dir / ".env"
        
        env_content = [
            "# AI Document Classifier Environment Configuration",
            "# Copy this file and add your API keys",
            "",
            "# OpenAI API Key (for AI classification)",
            "OPENAI_API_KEY=your_openai_api_key_here",
            "",
            "# Hugging Face API Key (for alternative AI models)",
            "HUGGINGFACE_API_KEY=your_huggingface_api_key_here",
            "",
            "# Google Translate API Key (for translation)",
            "GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key_here",
            "",
            "# Other service API keys",
            "GRAMMARLY_API_KEY=your_grammarly_api_key_here",
            "COPYSCAPE_API_KEY=your_copyscape_api_key_here",
            "",
            "# Application settings",
            "DEBUG=false",
            "HOST=0.0.0.0",
            "PORT=8000",
            "LOG_LEVEL=INFO",
            "",
            "# Database settings",
            "DATABASE_URL=sqlite:///./ai_document_classifier.db",
            "",
            "# Cache settings",
            "CACHE_TTL_HOURS=24",
            "MAX_CACHE_SIZE=1000",
            "",
            "# Performance settings",
            "MAX_WORKERS=4",
            "BATCH_SIZE=100",
            "REQUEST_TIMEOUT=30"
        ]
        
        if api_keys:
            for key, value in api_keys.items():
                env_content.append(f"{key}={value}")
        
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(env_content))
            
            logger.info(f"Environment file created: {env_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create environment file: {e}")
            return False
    
    def configure_services(self, service_config: Optional[Dict[str, Any]] = None):
        """
        Configure external services
        
        Args:
            service_config: Service configuration dictionary
        """
        config_file = self.config_dir / "services.json"
        
        if service_config:
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(service_config, f, indent=2)
                logger.info(f"Service configuration saved: {config_file}")
                return True
            except Exception as e:
                logger.error(f"Failed to save service configuration: {e}")
                return False
        else:
            # Use default configuration
            if config_file.exists():
                logger.info("Service configuration already exists")
                return True
            else:
                logger.warning("No service configuration provided and default not found")
                return False
    
    def initialize_database(self):
        """Initialize SQLite database for batch processing"""
        try:
            from utils.batch_processor import BatchProcessor
            from document_classifier_engine import DocumentClassifierEngine
            
            # Initialize components to create database
            classifier = DocumentClassifierEngine()
            processor = BatchProcessor(classifier, str(self.cache_dir))
            
            logger.info("Database initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def train_initial_models(self):
        """Train initial machine learning models"""
        try:
            from models.advanced_classifier import AdvancedDocumentClassifier
            
            classifier = AdvancedDocumentClassifier(str(self.models_dir))
            
            # Create and train models
            logger.info("Training initial models...")
            results = classifier.train_models()
            
            logger.info("Model training results:")
            for method, accuracy in results.items():
                logger.info(f"  {method}: {accuracy:.3f}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to train initial models: {e}")
            return False
    
    def run_tests(self):
        """Run basic tests to verify installation"""
        try:
            logger.info("Running basic tests...")
            
            # Test imports
            from document_classifier_engine import DocumentClassifierEngine
            from models.advanced_classifier import AdvancedDocumentClassifier
            from templates.dynamic_template_generator import DynamicTemplateGenerator
            from utils.batch_processor import BatchProcessor
            
            # Test basic classification
            classifier = DocumentClassifierEngine()
            result = classifier.classify_document("I want to write a novel", use_ai=False)
            
            if result and result.document_type:
                logger.info("✓ Basic classification test passed")
            else:
                logger.error("✗ Basic classification test failed")
                return False
            
            # Test template generation
            generator = DynamicTemplateGenerator()
            template = generator.generate_template("novel", "intermediate")
            
            if template and template.sections:
                logger.info("✓ Template generation test passed")
            else:
                logger.error("✗ Template generation test failed")
                return False
            
            logger.info("All tests passed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Tests failed: {e}")
            return False
    
    def setup_complete(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Run complete setup process
        
        Args:
            api_keys: Optional API keys for external services
            
        Returns:
            True if setup completed successfully
        """
        logger.info("Starting AI Document Classifier setup...")
        
        steps = [
            ("Installing dependencies", lambda: self.install_dependencies()),
            ("Downloading NLTK data", lambda: self.download_nltk_data()),
            ("Downloading SpaCy model", lambda: self.download_spacy_model()),
            ("Creating environment file", lambda: self.create_environment_file(api_keys)),
            ("Configuring services", lambda: self.configure_services()),
            ("Initializing database", lambda: self.initialize_database()),
            ("Training initial models", lambda: self.train_initial_models()),
            ("Running tests", lambda: self.run_tests())
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            try:
                if step_func():
                    success_count += 1
                    logger.info(f"✓ {step_name} completed")
                else:
                    logger.warning(f"⚠ {step_name} failed or skipped")
            except Exception as e:
                logger.error(f"✗ {step_name} failed: {e}")
        
        logger.info(f"Setup completed: {success_count}/{len(steps)} steps successful")
        
        if success_count == len(steps):
            logger.info("🎉 Setup completed successfully!")
            logger.info("You can now start the server with: python main.py")
            return True
        else:
            logger.warning("⚠ Setup completed with some issues. Check the logs above.")
            return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup AI Document Classifier")
    parser.add_argument("--api-keys", type=str, help="JSON file with API keys")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-models", action="store_true", help="Skip model training")
    parser.add_argument("--test-only", action="store_true", help="Run tests only")
    
    args = parser.parse_args()
    
    setup = AIDocumentClassifierSetup()
    
    # Load API keys if provided
    api_keys = None
    if args.api_keys:
        try:
            with open(args.api_keys, 'r') as f:
                api_keys = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return 1
    
    if args.test_only:
        success = setup.run_tests()
    else:
        success = setup.setup_complete(api_keys)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())



























