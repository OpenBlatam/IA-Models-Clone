#!/usr/bin/env python3
"""
Improved Real AI Document Processor Installation Script
Installs and configures the enhanced AI document processing system
"""

import subprocess
import sys
import os
import logging
import platform

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

def check_system_requirements():
    """Check system requirements"""
    logger.info("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    logger.info(f"✓ Python version: {sys.version}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
            logger.warning("Warning: Less than 4GB RAM available. Performance may be affected.")
        else:
            logger.info(f"✓ Available memory: {memory.total // (1024**3)}GB")
    except ImportError:
        logger.warning("psutil not available, cannot check memory")
    
    # Check disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free
        if free_space < 2 * 1024 * 1024 * 1024:  # 2GB
            logger.warning("Warning: Less than 2GB free disk space. May not be enough for models.")
        else:
            logger.info(f"✓ Free disk space: {free_space // (1024**3)}GB")
    except Exception as e:
        logger.warning(f"Cannot check disk space: {e}")
    
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
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    print("NLTK data installed successfully")
except Exception as e:
    print(f"Error installing NLTK data: {e}")
    exit(1)
"""
    
    return run_command(
        f"{sys.executable} -c \"{nltk_script}\"",
        "Installing NLTK data"
    )

def install_tesseract():
    """Install Tesseract OCR"""
    system = platform.system().lower()
    
    if system == "windows":
        logger.info("For Windows, please install Tesseract manually from:")
        logger.info("https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("Make sure to add it to your PATH")
        return True
    elif system == "darwin":  # macOS
        return run_command(
            "brew install tesseract",
            "Installing Tesseract OCR (macOS)"
        )
    elif system == "linux":
        return run_command(
            "sudo apt-get update && sudo apt-get install -y tesseract-ocr",
            "Installing Tesseract OCR (Linux)"
        )
    else:
        logger.warning(f"Unknown system: {system}. Please install Tesseract manually.")
        return True

def create_env_file():
    """Create .env file with enhanced configuration"""
    env_content = """# Improved Real AI Document Processor Configuration
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

# Advanced features
ENABLE_SENTENCE_TRANSFORMERS=true
ENABLE_SKLEARN=true
ENABLE_CACHING=true
ENABLE_REDIS=false

# API settings
RATE_LIMIT_PER_MINUTE=100
MAX_FILE_SIZE_MB=10

# Cache settings
CACHE_TTL=3600
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Performance settings
ENABLE_COMPRESSION=true
ENABLE_METRICS=true
ENABLE_MONITORING=true

# Optional: Add your API keys here
# OPENAI_API_KEY=your_openai_api_key_here
# AWS_ACCESS_KEY_ID=your_aws_key_here
# AWS_SECRET_ACCESS_KEY=your_aws_secret_here
# GOOGLE_CLOUD_PROJECT=your_google_project_here
# AZURE_STORAGE_ACCOUNT=your_azure_account_here
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        logger.info("Created .env file with enhanced configuration")
    else:
        logger.info(".env file already exists")

def test_installation():
    """Test the installation"""
    test_script = """
try:
    # Test basic imports
    from real_ai_processor import RealAIDocumentProcessor
    print("✓ Real AI processor imported successfully")
    
    from advanced_ai_processor import AdvancedAIProcessor
    print("✓ Advanced AI processor imported successfully")
    
    from document_parser import RealDocumentParser
    print("✓ Document parser imported successfully")
    
    # Test core libraries
    import spacy
    print("✓ spaCy imported successfully")
    
    import nltk
    print("✓ NLTK imported successfully")
    
    from transformers import pipeline
    print("✓ Transformers imported successfully")
    
    # Test advanced libraries
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence transformers imported successfully")
    except ImportError:
        print("⚠ Sentence transformers not available")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ scikit-learn imported successfully")
    except ImportError:
        print("⚠ scikit-learn not available")
    
    try:
        import redis
        print("✓ Redis imported successfully")
    except ImportError:
        print("⚠ Redis not available")
    
    try:
        import pytesseract
        print("✓ Tesseract OCR imported successfully")
    except ImportError:
        print("⚠ Tesseract OCR not available")
    
    print("✓ All core imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)
"""
    
    return run_command(
        f"{sys.executable} -c \"{test_script}\"",
        "Testing installation"
    )

def create_startup_script():
    """Create startup script"""
    startup_script = """#!/bin/bash
# Improved Real AI Document Processor Startup Script

echo "Starting Improved Real AI Document Processor..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
python improved_app.py
"""
    
    with open("start.sh", "w") as f:
        f.write(startup_script)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("start.sh", 0o755)
    
    logger.info("Created startup script: start.sh")

def create_docker_file():
    """Create Dockerfile for containerization"""
    dockerfile_content = """FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    tesseract-ocr-eng \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY real_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r real_requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "improved_app.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("Created Dockerfile for containerization")

def main():
    """Main installation function"""
    logger.info("Starting Improved Real AI Document Processor installation...")
    
    # Check system requirements
    if not check_system_requirements():
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
    
    # Install Tesseract
    if not install_tesseract():
        logger.warning("Tesseract installation failed, OCR features may not work")
    
    # Create .env file
    create_env_file()
    
    # Create startup script
    create_startup_script()
    
    # Create Dockerfile
    create_docker_file()
    
    # Test installation
    if not test_installation():
        logger.error("Installation test failed")
        sys.exit(1)
    
    logger.info("✓ Improved Real AI Document Processor installed successfully!")
    logger.info("You can now run:")
    logger.info("  python improved_app.py")
    logger.info("  or")
    logger.info("  ./start.sh")
    logger.info("  or")
    logger.info("  docker build -t ai-doc-processor .")
    logger.info("  docker run -p 8000:8000 ai-doc-processor")

if __name__ == "__main__":
    main()













