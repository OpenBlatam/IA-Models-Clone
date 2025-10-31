#!/usr/bin/env python3
"""
Enhanced AI Document Processor Installation Script
Installs real, working dependencies with enhanced features
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
    if not os.path.exists("practical_requirements.txt"):
        logger.error("practical_requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r practical_requirements.txt",
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
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("NLTK data installed successfully")
except Exception as e:
    print(f"Error installing NLTK data: {e}")
    exit(1)
"""
    
    return run_command(
        f"{sys.executable} -c \"{nltk_script}\"",
        "Installing NLTK data"
    )

def test_installation():
    """Test the installation"""
    test_script = """
try:
    # Test basic imports
    from practical_ai_processor import PracticalAIProcessor
    print("✓ Practical AI processor imported successfully")
    
    from enhanced_ai_processor import EnhancedAIProcessor
    print("✓ Enhanced AI processor imported successfully")
    
    # Test core libraries
    import spacy
    print("✓ spaCy imported successfully")
    
    import nltk
    print("✓ NLTK imported successfully")
    
    from transformers import pipeline
    print("✓ Transformers imported successfully")
    
    import fastapi
    print("✓ FastAPI imported successfully")
    
    # Test enhanced libraries
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
        import psutil
        print("✓ psutil imported successfully")
    except ImportError:
        print("⚠ psutil not available")
    
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
# Enhanced AI Document Processor Startup Script

echo "Starting Enhanced AI Document Processor..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
python enhanced_app.py
"""
    
    with open("start_enhanced.sh", "w") as f:
        f.write(startup_script)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("start_enhanced.sh", 0o755)
    
    logger.info("Created startup script: start_enhanced.sh")

def create_docker_file():
    """Create Dockerfile for containerization"""
    dockerfile_content = """FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
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
COPY practical_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r practical_requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "enhanced_app.py"]
"""
    
    with open("Dockerfile.enhanced", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("Created Dockerfile.enhanced for containerization")

def main():
    """Main installation function"""
    logger.info("Starting Enhanced AI Document Processor installation...")
    
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
    
    # Create startup script
    create_startup_script()
    
    # Create Dockerfile
    create_docker_file()
    
    # Test installation
    if not test_installation():
        logger.error("Installation test failed")
        sys.exit(1)
    
    logger.info("✓ Enhanced AI Document Processor installed successfully!")
    logger.info("You can now run:")
    logger.info("  python enhanced_app.py")
    logger.info("  or")
    logger.info("  ./start_enhanced.sh")
    logger.info("  or")
    logger.info("  docker build -f Dockerfile.enhanced -t enhanced-ai-doc-processor .")
    logger.info("  docker run -p 8000:8000 enhanced-ai-doc-processor")

if __name__ == "__main__":
    main()













