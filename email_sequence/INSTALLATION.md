# Installation Guide

This guide provides comprehensive instructions for installing the Email Sequence AI System with all its dependencies and optional features.

## Quick Start

### Basic Installation

For basic functionality with minimal dependencies:

```bash
# Clone the repository
git clone https://github.com/blatamacademy/email-sequence-ai.git
cd email-sequence-ai

# Install with minimal dependencies
pip install -e .[minimal]
```

### Full Installation

For complete functionality with all features:

```bash
# Install with all dependencies
pip install -e .[all]
```

## Installation Options

### 1. Minimal Installation

Basic functionality with essential dependencies only:

```bash
pip install -e .[minimal]
```

**Includes:**
- Core PyTorch and transformers
- Basic data processing (numpy, pandas, scikit-learn)
- Gradio web interface
- Essential configuration and logging
- Basic testing framework

### 2. Development Installation

For developers with testing and development tools:

```bash
pip install -e .[dev]
```

**Includes:**
- All minimal dependencies
- Testing frameworks (pytest, coverage)
- Code quality tools (black, flake8, mypy)
- Documentation tools (sphinx)
- Debugging tools (debugpy, icecream)
- Profiling tools (memory-profiler, line-profiler)

### 3. GPU Support

For GPU-accelerated training and inference:

```bash
pip install -e .[gpu]
```

**Includes:**
- CUDA-enabled PyTorch
- Apex for mixed precision training
- GPU monitoring tools

### 4. Distributed Training

For multi-GPU and distributed training:

```bash
pip install -e .[distributed]
```

**Includes:**
- Horovod for distributed training
- Ray for hyperparameter tuning
- Multi-GPU support

### 5. Cloud Integration

For cloud deployment and storage:

```bash
pip install -e .[cloud]
```

**Includes:**
- AWS (boto3, s3fs)
- Google Cloud (storage, AI platform)
- Azure (storage, ML services)

### 6. Monitoring and Observability

For production monitoring:

```bash
pip install -e .[monitoring]
```

**Includes:**
- Prometheus client
- Grafana API
- Sentry for error tracking
- Structured logging

### 7. Performance Profiling

For performance analysis and optimization:

```bash
pip install -e .[profiling]
```

**Includes:**
- Memory profiling
- Line profiling
- CPU profiling (py-spy, pyinstrument)
- Visualization tools

### 8. Hyperparameter Optimization

For advanced model optimization:

```bash
pip install -e .[optimization]
```

**Includes:**
- Optuna for hyperparameter tuning
- Hyperopt for Bayesian optimization
- Ray Tune for distributed optimization
- Weights & Biases integration
- MLflow for experiment tracking
- TensorBoard for visualization

### 9. NLP and Text Processing

For advanced text processing:

```bash
pip install -e .[nlp]
```

**Includes:**
- NLTK for natural language processing
- spaCy for advanced NLP
- TextBlob for sentiment analysis
- Gensim for word embeddings
- WordCloud for visualization
- Jieba for Chinese text processing

### 10. Web Framework Support

For web application development:

```bash
pip install -e .[web]
```

**Includes:**
- Flask for web applications
- FastAPI for high-performance APIs
- Streamlit for data apps
- Dash for interactive dashboards

### 11. Database Support

For database integration:

```bash
pip install -e .[database]
```

**Includes:**
- SQLAlchemy for ORM
- Alembic for migrations
- Redis for caching
- MongoDB support
- Elasticsearch for search

## Environment Setup

### Python Version

The system requires Python 3.8 or higher:

```bash
# Check Python version
python --version

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### CUDA Setup (Optional)

For GPU support, install CUDA toolkit:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    unzip \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libxft-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config
```

#### CentOS/RHEL

```bash
sudo yum update
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    python3-devel \
    python3-pip \
    git \
    curl \
    wget \
    unzip \
    openssl-devel \
    libffi-devel \
    libjpeg-devel \
    libpng-devel \
    freetype-devel \
    blas-devel \
    lapack-devel \
    atlas-devel \
    gcc-gfortran \
    pkgconfig
```

#### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install \
    python3 \
    git \
    curl \
    wget \
    openssl \
    pkg-config \
    libjpeg \
    libpng \
    freetype \
    openblas \
    gcc
```

#### Windows

1. Install Visual Studio Build Tools
2. Install Git for Windows
3. Install Python from python.org
4. Install required Visual C++ redistributables

## Installation Verification

### Basic Verification

```bash
# Test basic installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import gradio; print(f'Gradio version: {gradio.__version__}')"
```

### GPU Verification

```bash
# Test GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### System Test

```bash
# Run basic tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/email_sequence_db
REDIS_URL=redis://localhost:6379

# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Cloud Storage
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-west-2

# Monitoring
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_PORT=9090

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/email_sequence.log

# Model Configuration
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./data
```

### Configuration Files

The system uses YAML configuration files:

```yaml
# config/config.yaml
model:
  name: "email-sequence-model"
  type: "transformer"
  max_length: 512
  batch_size: 16

training:
  epochs: 10
  learning_rate: 1e-4
  warmup_steps: 1000
  gradient_accumulation_steps: 4

optimization:
  mixed_precision: true
  gradient_clipping: 1.0
  early_stopping_patience: 5

profiling:
  enabled: true
  memory_tracking: true
  performance_monitoring: true
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation Issues

```bash
# Clear pip cache
pip cache purge

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. CUDA Issues

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"
```

#### 3. Memory Issues

```bash
# Reduce batch size in configuration
# Enable gradient checkpointing
# Use mixed precision training
```

#### 4. Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall in development mode
pip install -e . --force-reinstall
```

### Getting Help

1. Check the [Documentation](https://email-sequence-ai.readthedocs.io/)
2. Search [Issues](https://github.com/blatamacademy/email-sequence-ai/issues)
3. Create a new issue with detailed information
4. Join our [Discord Community](https://discord.gg/blatamacademy)

## Next Steps

After installation:

1. **Quick Start**: Run the basic demo
   ```bash
   python examples/basic_demo.py
   ```

2. **Training**: Start with a simple training example
   ```bash
   python examples/training_example.py
   ```

3. **Profiling**: Analyze performance
   ```bash
   python examples/profiling_demo.py
   ```

4. **Web Interface**: Launch Gradio app
   ```bash
   python examples/gradio_app.py
   ```

5. **Documentation**: Read the comprehensive guides
   - [User Guide](docs/USER_GUIDE.md)
   - [API Reference](docs/API_REFERENCE.md)
   - [Training Guide](docs/TRAINING_GUIDE.md)
   - [Optimization Guide](docs/OPTIMIZATION_GUIDE.md) 