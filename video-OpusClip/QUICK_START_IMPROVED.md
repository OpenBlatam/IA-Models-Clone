# 🚀 Quick Start Guide - Improved Video-OpusClip API

## 🎯 **Get Started in 5 Minutes**

This guide will help you quickly get started with the improved Video-OpusClip API that follows FastAPI best practices.

---

## 📋 **Prerequisites**

- Python 3.8+
- FastAPI
- Redis (optional, falls back to in-memory cache)
- All dependencies from `requirements_opus_clip.txt`

---

## ⚡ **Quick Installation**

```bash
# Install dependencies
pip install -r requirements_opus_clip.txt

# Run the improved API
python improved_api.py
```

The API will be available at: `http://localhost:8000`

---

## 🔧 **Basic Usage**

### **1. Video Processing**

```python
import requests

# Process a single video
response = requests.post("http://localhost:8000/api/v1/video/process", json={
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "language": "en",
    "max_clip_length": 60,
    "quality": "high",
    "format": "mp4"
})

print(response.json())
```

### **2. Batch Processing**

```python
# Process multiple videos
response = requests.post("http://localhost:8000/api/v1/video/batch", json={
    "requests": [
        {
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "language": "en",
            "max_clip_length": 60
        },
        {
            "youtube_url": "https://www.youtube.com/watch?v=example2",
            "language": "es",
            "max_clip_length": 45
        }
    ],
    "max_workers": 4
})

print(response.json())
```

### **3. Viral Video Generation**

```python
# Generate viral variants
response = requests.post("http://localhost:8000/api/v1/viral/process", json={
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "n_variants": 5,
    "use_langchain": True,
    "platform": "tiktok"
})

print(response.json())
```

### **4. LangChain Analysis**

```python
# Analyze content with LangChain
response = requests.post("http://localhost:8000/api/v1/langchain/analyze", json={
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "analysis_type": "comprehensive",
    "platform": "youtube"
})

print(response.json())
```

---

## 🏥 **Health Check**

```python
# Check API health
response = requests.get("http://localhost:8000/health")
print(response.json())
```

---

## 📊 **Key Features**

### **✅ Early Returns & Guard Clauses**
- Fast error detection and handling
- Security validation at entry points
- System health checks before processing

### **✅ Enhanced Caching**
- Redis primary cache with in-memory fallback
- Automatic cache invalidation
- Performance optimization

### **✅ Comprehensive Validation**
- Input sanitization and validation
- Security threat detection
- Type safety with Pydantic models

### **✅ Performance Monitoring**
- Real-time metrics collection
- Health monitoring
- Error tracking and statistics

### **✅ Modular Architecture**
- Clean separation of concerns
- Easy to maintain and extend
- Production-ready structure

---

## 🔍 **API Documentation**

Visit `http://localhost:8000/docs` for interactive API documentation with:
- Request/response schemas
- Example requests
- Error codes and responses
- Try-it-out functionality

---

## 🧪 **Testing**

```bash
# Run the demo
python demo_improved_api.py

# Run tests
python test_improved_api.py

# Or with pytest
pytest test_improved_api.py -v
```

---

## 🚀 **Production Deployment**

### **Environment Variables**
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export LOG_LEVEL=info
```

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_opus_clip.txt .
RUN pip install -r requirements_opus_clip.txt

COPY . .
EXPOSE 8000

CMD ["python", "improved_api.py"]
```

---

## 📈 **Performance Tips**

1. **Enable Redis**: For better caching performance
2. **Use Batch Processing**: For multiple videos
3. **Monitor Health**: Check `/health` endpoint regularly
4. **Optimize Workers**: Adjust `max_workers` based on your system

---

## 🔒 **Security Features**

- **Input Validation**: All inputs are validated and sanitized
- **URL Sanitization**: Malicious URLs are blocked
- **Rate Limiting**: Built-in protection against abuse
- **Request Tracking**: All requests are tracked with IDs

---

## 🆘 **Troubleshooting**

### **Common Issues**

1. **Redis Connection Error**
   - Solution: API falls back to in-memory cache automatically

2. **Validation Errors**
   - Check request format and required fields
   - Ensure YouTube URLs are valid

3. **Performance Issues**
   - Check system resources at `/health`
   - Adjust `max_workers` parameter

### **Getting Help**

- Check API documentation at `/docs`
- Review error responses for detailed information
- Monitor logs for debugging information

---

## 🎉 **You're Ready!**

The improved Video-OpusClip API is now running with:
- ✅ FastAPI best practices
- ✅ Comprehensive error handling
- ✅ Performance optimizations
- ✅ Security enhancements
- ✅ Production-ready features

**Happy coding! 🚀**






























