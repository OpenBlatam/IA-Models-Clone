#!/usr/bin/env python3
"""
Complete Real AI Document Processor Installation Script
Installs all real, working dependencies with advanced features
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
    if not os.path.exists("real_working_requirements.txt"):
        logger.error("real_working_requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r real_working_requirements.txt",
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
    from real_working_processor import RealWorkingProcessor
    print("✓ Real working processor imported successfully")
    
    from advanced_real_processor import AdvancedRealProcessor
    print("✓ Advanced real processor imported successfully")
    
    # Test core libraries
    import spacy
    print("✓ spaCy imported successfully")
    
    import nltk
    print("✓ NLTK imported successfully")
    
    from transformers import pipeline
    print("✓ Transformers imported successfully")
    
    import fastapi
    print("✓ FastAPI imported successfully")
    
    # Test advanced libraries
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ scikit-learn imported successfully")
    except ImportError:
        print("⚠ scikit-learn not available")
    
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

def create_startup_scripts():
    """Create startup scripts"""
    # Basic startup script
    basic_script = """#!/bin/bash
# Real Working AI Document Processor Startup Script

echo "Starting Real Working AI Document Processor..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
python improved_real_app.py
"""
    
    with open("start_basic.sh", "w") as f:
        f.write(basic_script)
    
    # Complete startup script
    complete_script = """#!/bin/bash
# Complete Real AI Document Processor Startup Script

echo "Starting Complete Real AI Document Processor..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
python complete_real_app.py
"""
    
    with open("start_complete.sh", "w") as f:
        f.write(complete_script)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("start_basic.sh", 0o755)
        os.chmod("start_complete.sh", 0o755)
    
    logger.info("Created startup scripts: start_basic.sh, start_complete.sh")

def create_docker_files():
    """Create Dockerfiles for containerization"""
    # Basic Dockerfile
    basic_dockerfile = """FROM python:3.9-slim

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
COPY real_working_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r real_working_requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "improved_real_app.py"]
"""
    
    with open("Dockerfile.basic", "w") as f:
        f.write(basic_dockerfile)
    
    # Complete Dockerfile
    complete_dockerfile = """FROM python:3.9-slim

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
COPY real_working_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r real_working_requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "complete_real_app.py"]
"""
    
    with open("Dockerfile.complete", "w") as f:
        f.write(complete_dockerfile)
    
    logger.info("Created Dockerfiles: Dockerfile.basic, Dockerfile.complete")

def main():
    """Main installation function"""
    logger.info("Starting Complete Real AI Document Processor installation...")
    
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
    
    # Create startup scripts
    create_startup_scripts()
    
    # Create Dockerfiles
    create_docker_files()
    
    # Test installation
    if not test_installation():
        logger.error("Installation test failed")
        sys.exit(1)
    
    logger.info("✓ Complete Real AI Document Processor installed successfully!")
    logger.info("You can now run:")
    logger.info("  Basic version:")
    logger.info("    python improved_real_app.py")
    logger.info("    or ./start_basic.sh")
    logger.info("  Complete version:")
    logger.info("    python complete_real_app.py")
    logger.info("    or ./start_complete.sh")
    logger.info("  Docker:")
    logger.info("    docker build -f Dockerfile.basic -t basic-ai-doc-processor .")
    logger.info("    docker build -f Dockerfile.complete -t complete-ai-doc-processor .")

if __name__ == "__main__":
    main()













