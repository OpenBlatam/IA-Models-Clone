# 🚀 OS Content System

> Part of the [Blatam Academy Integrated Platform](../README.md)

**Advanced content generation and management system with AI, optimized for high performance and scalability.**

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API](#-api)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🎯 **Key Functionalities**
- **AI Content Generation**: Integration with advanced language models
- **Video Processing**: Optimized pipeline for video generation and editing
- **Math System**: Unified platform for mathematical calculations and analysis
- **Audio Processing**: AI audio generation and manipulation
- **RESTful API**: Complete interface with automatic documentation

### 🚀 **Technical Features**
- **Clean Architecture**: Implementation of Clean Architecture and SOLID principles
- **Asynchronous Processing**: Efficient handling of concurrent tasks
- **Multi-Level Cache System**: L1 (memory), L2 (Redis), L3 (disk)
- **Advanced Monitoring**: Real-time metrics with Prometheus and Grafana
- **Structured Logging**: Logging system with context and performance analysis
- **Automated Testing**: Complete suite of unit and integration tests

### 🔧 **Performance Optimizations**
- **GPU Acceleration**: Support for CUDA and hardware optimizations
- **Intelligent Compression**: Multiple compression algorithms (ZSTD, LZ4, Brotli)
- **Load Balancing**: Intelligent load distribution
- **CDN Integration**: Distributed content management
- **Memory Management**: Automatic memory management and cleanup

## 🏗️ Architecture

### **Architecture Diagram**
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Auth      │  │   Content   │  │   Media     │        │
│  │  Service    │  │  Generator  │  │  Processor  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Cache     │  │   Queue     │  │   Storage   │        │
│  │  Manager    │  │  Manager    │  │   Service   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Database Layer                           │
└─────────────────────────────────────────────────────────────┘
```

### **Core Components**

#### **1. Core Services**
- **ContentGenerator**: Intelligent content generation
- **MediaProcessor**: Audio and video processing
- **MathEngine**: Unified math engine
- **NLPService**: Natural language processing

#### **2. Infrastructure**
- **CacheManager**: Multi-level cache management
- **AsyncProcessor**: Asynchronous task processing
- **LoadBalancer**: Intelligent load balancing
- **PerformanceMonitor**: Performance monitoring

#### **3. Data Layer**
- **PostgreSQL**: Main database
- **Redis**: Cache and sessions
- **Elasticsearch**: Search and analysis
- **MongoDB**: Document storage

## 🛠️ Installation

### **System Requirements**
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: Optional, NVIDIA CUDA compatible
- **System**: Linux, macOS, or Windows 10+

### **Quick Install**

#### **1. Clone Repository**
```bash
git clone <repository-url>
cd os-content-system
```

#### **2. Run Automatic Setup**
```bash
# Full setup with tests
python setup.py

# Setup without tests (faster)
python setup.py --skip-tests
```

#### **3. Manual Configuration (Alternative)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Unix/Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp env.example .env
# Edit .env with your configurations
```

### **Verify Installation**
```bash
# Verify installation
python -c "import fastapi, torch, transformers; print('✅ Successful installation')"

# Run basic tests
python test_suite.py
```

## ⚙️ Configuration

### **Main Environment Variables**

```bash
# Application Configuration
ENVIRONMENT=development
APP_NAME="OS Content System"
APP_VERSION=1.0.0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true

# Database
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=your_password
DB_NAME=os_content

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
SECRET_KEY=your-super-secret-key-here
```

### **Configuration Files**

#### **config.py**
```python
from config import get_config

config = get_config()
print(f"Database: {config.database.connection_string}")
print(f"Redis: {config.redis.connection_string}")
```

#### **logger.py**
```python
from logger import get_logger, log_with_context

logger = get_logger("my_service")
log_with_context("INFO", "Operation completed", user_id=123, duration=0.5)
```

## 🚀 Usage

### **Start System**

#### **Windows**
```bash
# Option 1: Automatic script
start.bat

# Option 2: Manual
venv\Scripts\activate
python main.py
```

#### **Unix/Linux/macOS**
```bash
# Option 1: Automatic script
./start.sh

# Option 2: Manual
source venv/bin/activate
python main.py
```

### **System Access**
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:9090

### **Usage Examples**

#### **Generate Content**
```python
import requests

# Generate text
response = requests.post("http://localhost:8000/api/generate/text", json={
    "prompt": "Write an article about AI",
    "max_length": 500
})

# Generate video
response = requests.post("http://localhost:8000/api/generate/video", json={
    "script": "Script for the video",
    "duration": 60
})
```

#### **Process Media**
```python
# Process audio
response = requests.post("http://localhost:8000/api/process/audio", json={
    "operation": "enhance",
    "file_path": "/path/to/audio.mp3"
})

# Process video
response = requests.post("http://localhost:8000/api/process/video", json={
    "operation": "compress",
    "file_path": "/path/to/video.mp4"
})
```

## 🔌 API

### **Main Endpoints**

#### **Content Generation**
- `POST /api/generate/text` - Generate text
- `POST /api/generate/video` - Generate video
- `POST /api/generate/audio` - Generate audio
- `POST /api/generate/image` - Generate image

#### **Media Processing**
- `POST /api/process/video` - Process video
- `POST /api/process/audio` - Process audio
- `POST /api/process/image` - Process image

#### **Math System**
- `POST /api/math/calculate` - Mathematical calculations
- `POST /api/math/optimize` - Mathematical optimization
- `POST /api/math/analyze` - Mathematical analysis

#### **System Management**
- `GET /health` - System status
- `GET /metrics` - Performance metrics
- `GET /status` - Detailed status

### **Response Examples**

#### **Text Generation**
```json
{
  "success": true,
  "data": {
    "text": "Generated article about AI...",
    "tokens_used": 150,
    "processing_time": 2.5
  },
  "metadata": {
    "model": "gpt-4",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### **System Status**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "2h 15m 30s",
  "services": {
    "database": "connected",
    "redis": "connected",
    "gpu": "available"
  },
  "performance": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "active_requests": 12
  }
}
```

## 🧪 Testing

### **Run Tests**

#### **Full Suite**
```bash
python test_suite.py
```

#### **Specific Tests**
```bash
# Unit tests only
pytest test_suite.py::TestConfig -v

# Performance tests only
pytest test_suite.py::TestPerformance -v

# Integration tests
pytest test_suite.py::TestIntegration -v
```

#### **Tests with Coverage**
```bash
pytest test_suite.py --cov=. --cov-report=html
```

### **Test Types**

- **Unit Tests**: Tests of individual components
- **Integration Tests**: Tests of integration between services
- **Performance Tests**: Performance tests and benchmarks
- **Error Handling Tests**: Error handling tests

## 🚀 Deployment

### **Local Deployment**

#### **Docker Compose**
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### **Kubernetes**
```bash
# Apply configuration
kubectl apply -f k8s/

# Verify status
kubectl get pods
kubectl get services
```

### **Production Deployment**

#### **Production Environment Variables**
```bash
ENVIRONMENT=production
API_DEBUG=false
API_WORKERS=4
LOG_LEVEL=WARNING
PROMETHEUS_ENABLED=true
```

#### **Production Monitoring**
- **Prometheus**: System metrics
- **Grafana**: Dashboards and visualization
- **Sentry**: Error monitoring
- **ELK Stack**: Logs and analysis

## 📊 Monitoring and Logs

### **Log System**

#### **Log Types**
- **Application Logs**: `logs/os_content.log`
- **Error Logs**: `logs/os_content_error.log`
- **Structured Logs**: `logs/os_content_structured.log`

#### **Log Levels**
- **DEBUG**: Detailed information for development
- **INFO**: General system information
- **WARNING**: Warnings that do not prevent operation
- **ERROR**: Errors that affect functionality
- **CRITICAL**: Critical system errors

### **Performance Metrics**

#### **System Metrics**
- CPU and memory usage
- Disk and network I/O
- API response time
- Error rate

#### **Application Metrics**
- Processing time
- Cache usage
- Task queue
- Services status

## 🔧 Development

### **Project Structure**
```
os-content-system/
├── main.py                 # Main entry point
├── config.py              # Configuration system
├── logger.py              # Logging system
├── requirements.txt       # Python dependencies
├── setup.py              # Installation script
├── test_suite.py         # Test suite
├── env.example           # Example environment variables
├── README.md             # This file
├── logs/                 # Logs directory
├── data/                 # Data directory
├── models/               # AI models
├── cache/                # Cache directory
└── docs/                 # Additional documentation
```

### **Development Workflow**

#### **1. Setup Environment**
```bash
# Clone and setup
git clone <repo>
cd os-content-system
python setup.py --skip-tests
```

#### **2. Activate Virtual Environment**
```bash
# Windows
venv\Scripts\activate

# Unix/Linux/macOS
source venv/bin/activate
```

#### **3. Install Development Dependencies**
```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov
```

#### **4. Run Tests**
```bash
python test_suite.py
```

#### **5. Local Development**
```bash
python main.py
```

### **Code Conventions**

#### **Code Style**
- **PEP 8**: Python code style
- **Type Hints**: Mandatory type annotations
- **Docstrings**: Function and class documentation
- **Error Handling**: Robust error handling

#### **Commit Structure**
```
feat: new video generation feature
fix: fix in cache error handling
docs: API documentation update
test: added tests for logging system
refactor: cache engine refactor
```

## 🤝 Contributing

### **How to Contribute**

#### **1. Fork Repository**
- Fork the project on GitHub
- Clone your fork locally

#### **2. Create Feature Branch**
```bash
git checkout -b feature/new-feature
```

#### **3. Development**
- Implement your feature
- Add appropriate tests
- Verify that all tests pass

#### **4. Commit and Push**
```bash
git add .
git commit -m "feat: feature description"
git push origin feature/new-feature
```

#### **5. Pull Request**
- Create a Pull Request on GitHub
- Describe the changes made
- Wait for code review

### **Contribution Guidelines**

#### **Code**
- Follow established style conventions
- Include tests for new functionality
- Keep test coverage high
- Document new functions and classes

#### **Documentation**
- Update README.md if necessary
- Document new APIs
- Include usage examples
- Keep documentation updated

#### **Testing**
- All tests must pass
- New functionality must have tests
- Keep test coverage >90%
- Include integration tests where appropriate

## 📚 Additional Resources

### **Documentation**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Python AsyncIO](https://docs.python.org/3/library/asyncio.html)

### **Related Tools**
- **MLflow**: ML experiment management
- **Ray**: Distributed computing
- **Dask**: Parallel data processing
- **Optuna**: Hyperparameter optimization

### **Community**
- **Discord**: [Link to server]
- **GitHub Issues**: [Link to issues]
- **Documentation**: [Link to docs]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI**: Modern and fast web framework
- **Pydantic**: Data validation with Python
- **Transformers**: Hugging Face AI models
- **PyTorch**: Deep learning framework
- **Open Source Community**: For all contributions

---

**Need help?**
- 📧 Email: support@oscontent.com
- 💬 Discord: [Link to server]
- 📖 Docs: [Link to documentation]
- 🐛 Issues: [Link to GitHub Issues]

---

**⭐ If this project is useful to you, leave us a star on GitHub!**
