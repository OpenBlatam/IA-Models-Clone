# 🚀 Facebook Posts API - Quick Start Guide

## ⚡ Get Started in 5 Minutes

### 1. **Setup the System**
```bash
# Run the automated setup script
python setup_improved_system.py

# Or manually:
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

pip install -r requirements_improved.txt
```

### 2. **Configure Environment**
```bash
# Edit the .env file with your settings
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 3. **Start the Server**
```bash
# Using the startup script
./start_server.sh        # Linux/macOS
start_server.bat         # Windows

# Or manually
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. **Test the API**
```bash
# Run the demo
python demo_improved_api.py

# Or test manually
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Interactive API docs
```

## 🎯 Key Features

### **Enhanced API Endpoints**
- ✅ **POST** `/api/v1/posts/generate` - Generate single post
- ✅ **POST** `/api/v1/posts/generate/batch` - Generate multiple posts
- ✅ **GET** `/api/v1/posts` - List posts with filtering
- ✅ **GET** `/api/v1/posts/{id}` - Get specific post
- ✅ **PUT** `/api/v1/posts/{id}` - Update post
- ✅ **DELETE** `/api/v1/posts/{id}` - Delete post
- ✅ **POST** `/api/v1/posts/{id}/optimize` - Optimize post
- ✅ **GET** `/api/v1/health` - Health check
- ✅ **GET** `/api/v1/metrics` - Performance metrics

### **Advanced Features**
- 🔒 **Rate Limiting**: Per-user rate limiting
- 🔐 **Authentication**: JWT and API key support
- 📊 **Analytics**: Background analytics processing
- 🚀 **Async Operations**: Non-blocking I/O
- 📝 **Comprehensive Logging**: Structured logging with request IDs
- 🧪 **Full Test Coverage**: Comprehensive test suite
- 📚 **Auto Documentation**: OpenAPI/Swagger docs

## 💡 Usage Examples

### **Generate a Post**
```python
import httpx

async def generate_post():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/posts/generate",
            json={
                "topic": "AI in Business",
                "audience_type": "professionals",
                "content_type": "educational",
                "tone": "professional",
                "optimization_level": "advanced"
            }
        )
        return response.json()
```

### **Batch Generation**
```python
async def generate_batch():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/posts/generate/batch",
            json={
                "requests": [
                    {
                        "topic": "Digital Marketing",
                        "audience_type": "professionals",
                        "content_type": "educational"
                    },
                    {
                        "topic": "Remote Work",
                        "audience_type": "general",
                        "content_type": "educational"
                    }
                ],
                "parallel_processing": True
            }
        )
        return response.json()
```

### **Filter Posts**
```python
async def list_posts():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/api/v1/posts",
            params={
                "status": "published",
                "content_type": "educational",
                "audience_type": "professionals",
                "limit": 10,
                "skip": 0
            }
        )
        return response.json()
```

## 🔧 Configuration

### **Environment Variables**
```bash
# API Configuration
API_TITLE=Ultimate Facebook Posts API
API_VERSION=4.0.0
DEBUG=true

# Server
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=sqlite:///./facebook_posts.db

# Redis
REDIS_URL=redis://localhost:6379

# AI Service
AI_API_KEY=your_openai_api_key_here
AI_MODEL=gpt-3.5-turbo

# Security
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

## 🧪 Testing

### **Run Tests**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_improved_api.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### **Test Categories**
- ✅ **Unit Tests**: Individual function testing
- ✅ **Integration Tests**: API endpoint testing
- ✅ **Error Tests**: Error scenario testing
- ✅ **Performance Tests**: Response time validation
- ✅ **Security Tests**: Authentication testing

## 📊 Monitoring

### **Health Check**
```bash
curl http://localhost:8000/health
```

### **Metrics**
```bash
curl http://localhost:8000/api/v1/metrics
```

### **Logs**
```bash
# View logs
tail -f logs/facebook_posts.log

# Or check console output for structured logs
```

## 🚀 Production Deployment

### **Environment Setup**
```bash
# Set production environment
export DEBUG=false
export API_KEY=your_production_api_key
export SECRET_KEY=your_production_secret_key
export DATABASE_URL=postgresql://user:pass@host:port/db
export REDIS_URL=redis://production-redis:6379
```

### **Start Production Server**
```bash
# Using Gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or using Uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🔍 Troubleshooting

### **Common Issues**

**1. Port Already in Use**
```bash
# Find process using port 8000
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn app:app --port 8001
```

**2. Import Errors**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements_improved.txt
```

**3. Database Connection Issues**
```bash
# Check database URL in .env file
# For SQLite, ensure directory exists
mkdir -p data
```

**4. Redis Connection Issues**
```bash
# Install and start Redis
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Windows
# Download Redis for Windows or use Docker
```

## 📚 Documentation

### **API Documentation**
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### **Additional Resources**
- **README**: README_IMPROVEMENTS.md
- **Improvements Summary**: IMPROVEMENTS_SUMMARY.md
- **Demo Script**: demo_improved_api.py
- **Test Suite**: tests/test_improved_api.py

## 🎉 Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Run the Demo**: `python demo_improved_api.py`
3. **Read the Documentation**: Check README_IMPROVEMENTS.md
4. **Run Tests**: `pytest tests/ -v`
5. **Customize Configuration**: Edit .env file
6. **Deploy to Production**: Follow production deployment guide

## 🆘 Support

### **Getting Help**
- Check the documentation in README_IMPROVEMENTS.md
- Review the test cases in tests/test_improved_api.py
- Run the demo script to see examples
- Check logs for error details

### **Contributing**
- Follow the code style in existing files
- Add tests for new features
- Update documentation
- Use type hints and proper error handling

Happy coding! 🚀






























