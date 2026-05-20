#!/usr/bin/env python3
"""
Real AI Document Processor Installation Script
Installs and configures the real AI document processing system
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log the result"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {description}")
        logger.error(f"Command: {command}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_requirements():
    """Install Python requirements"""
    if not os.path.exists("real_requirements.txt"):
        logger.error("real_requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r real_requirements.txt",
        "Installing Python requirements"
    )

def install_spacy_model():
    """Install spaCy English model"""
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Installing spaCy English model"
    )

def install_nltk_data():
    """Install NLTK data"""
    nltk_script = """
import nltk
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data installed successfully")
except Exception as e:
    print(f"Error installing NLTK data: {e}")
    exit(1)
"""
    
    return run_command(
        f"{sys.executable} -c \"{nltk_script}\"",
        "Installing NLTK data"
    )

def create_env_file():
    """Create .env file with default configuration"""
    env_content = """# Real AI Document Processor Configuration
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# AI Model settings
SPACY_MODEL=en_core_web_sm
MAX_TEXT_LENGTH=5120
MAX_SUMMARY_LENGTH=150
MIN_SUMMARY_LENGTH=30

# Processing settings
ENABLE_SPACY=true
ENABLE_NLTK=true
ENABLE_TRANSFORMERS=true
ENABLE_SENTIMENT=true
ENABLE_CLASSIFICATION=true
ENABLE_SUMMARIZATION=true
ENABLE_QA=true

# API settings
RATE_LIMIT_PER_MINUTE=100
MAX_FILE_SIZE_MB=10

# Optional: Add your API keys here
# OPENAI_API_KEY=your_openai_api_key_here
# AWS_ACCESS_KEY_ID=your_aws_key_here
# AWS_SECRET_ACCESS_KEY=your_aws_secret_here
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        logger.info("Created .env file with default configuration")
    else:
        logger.info(".env file already exists")

def test_installation():
    """Test the installation"""
    test_script = """
try:
    from real_ai_processor import RealAIDocumentProcessor
    print("✓ Real AI processor imported successfully")
    
    import spacy
    print("✓ spaCy imported successfully")
    
    import nltk
    print("✓ NLTK imported successfully")
    
    from transformers import pipeline
    print("✓ Transformers imported successfully")
    
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)
"""
    
    return run_command(
        f"{sys.executable} -c \"{test_script}\"",
        "Testing installation"
    )

def main():
    """Main installation function"""
    logger.info("Starting Real AI Document Processor installation...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        sys.exit(1)
    
    # Install spaCy model
    if not install_spacy_model():
        logger.error("Failed to install spaCy model")
        sys.exit(1)
    
    # Install NLTK data
    if not install_nltk_data():
        logger.error("Failed to install NLTK data")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Test installation
    if not test_installation():
        logger.error("Installation test failed")
        sys.exit(1)
    
    logger.info("✓ Real AI Document Processor installed successfully!")
    logger.info("You can now run: python real_app.py")

if __name__ == "__main__":
    main()













