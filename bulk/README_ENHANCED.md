# 🚀 BUL - Business Universal Language (Enhanced)

**Advanced AI-powered document generation system for SMEs with enterprise-grade features**

BUL Enhanced is a comprehensive, production-ready system that provides AI-powered document generation with advanced features including authentication, rate limiting, caching, metrics, and real-time monitoring.

## ✨ Enhanced Features

### 🔐 **Authentication & Security**
- User authentication with API keys
- Session management
- Permission-based access control
- Secure endpoints

### ⚡ **Performance & Scalability**
- Redis caching for improved performance
- Rate limiting (10 requests/minute per IP)
- Background task processing
- Async/await support

### 📊 **Monitoring & Metrics**
- Prometheus metrics integration
- Real-time dashboard with Dash
- System health monitoring
- Performance analytics

### 🔧 **Advanced Functionality**
- File upload/download support
- Task cancellation
- Enhanced error handling
- Comprehensive logging
- WebSocket support (ready)

### 📈 **Dashboard & Visualization**
- Real-time system monitoring
- Task status visualization
- Performance metrics
- Interactive charts

## 🏗️ Architecture

```
bulk/
├── bul_enhanced.py          # Enhanced API server
├── dashboard.py             # Real-time dashboard
├── start_enhanced.py        # System launcher
├── test_enhanced_api.py     # Comprehensive tests
├── requirements.txt         # Enhanced dependencies
├── uploads/                 # File upload directory
├── downloads/              # File download directory
└── logs/                   # System logs
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone or navigate to the directory
cd C:\blatam-academy\agents\backend\onyx\server\features\bulk

# Install dependencies
pip install -r requirements.txt
```

### 2. **Start the System**

```bash
# Option 1: Use the launcher (recommended)
python start_enhanced.py

# Option 2: Start components separately
python bul_enhanced.py --host 0.0.0.0 --port 8000
python dashboard.py
```

### 3. **Access the Services**

- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8050
- **API Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

## 📋 API Endpoints

### 🔐 **Authentication**
- `POST /auth/login` - User login
- `GET /auth/logout` - User logout

### 📄 **Documents**
- `POST /documents/generate` - Generate document (rate limited)
- `GET /download/{task_id}` - Download generated document

### 📊 **Tasks**
- `GET /tasks` - List tasks (with filtering & pagination)
- `GET /tasks/{task_id}/status` - Get task status
- `POST /tasks/{task_id}/cancel` - Cancel task
- `DELETE /tasks/{task_id}` - Delete task

### 📁 **Files**
- `POST /upload` - Upload file
- `GET /download/{task_id}` - Download document

### 🔧 **System**
- `GET /` - System information
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /metrics` - Prometheus metrics

## 🎯 Usage Examples

### **Generate Document**

```javascript
// Frontend JavaScript
async function generateDocument(query, businessArea, documentType) {
    const response = await fetch('http://localhost:8000/documents/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer your_session_token'
        },
        body: JSON.stringify({
            query: query,
            business_area: businessArea,
            document_type: documentType,
            priority: 1,
            user_id: 'user123',
            session_id: 'session456'
        })
    });
    return await response.json();
}
```

### **Monitor Task Status**

```javascript
async function monitorTask(taskId) {
    const response = await fetch(`http://localhost:8000/tasks/${taskId}/status`);
    const data = await response.json();
    
    console.log(`Task ${taskId}: ${data.status} (${data.progress}%)`);
    
    if (data.status === 'completed') {
        console.log('Document ready for download!');
    }
}
```

### **Upload File**

```javascript
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
    });
    return await response.json();
}
```

## 🔧 Configuration

### **Environment Variables**

```bash
# Optional: Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Optional: API configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# Optional: Rate limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60
```

### **Default Users**

The system comes with default users for testing:

- **Admin**: `user_id: admin`, `api_key: admin_key_123`
- **User**: `user_id: user1`, `api_key: user_key_456`

## 📊 Dashboard Features

The real-time dashboard provides:

- **System Status**: API health, uptime, active tasks
- **Performance Metrics**: Request count, success rate, processing time
- **Task Monitoring**: Real-time task status and progress
- **Visualizations**: Charts and graphs for system metrics
- **Interactive Tables**: Sortable and filterable task lists

## 🧪 Testing

### **Run Tests**

```bash
# Run comprehensive test suite
python test_enhanced_api.py

# Run specific test categories
pytest test_enhanced_api.py::TestEnhancedBULAPI::test_document_generation
```

### **Test Coverage**

The test suite covers:
- ✅ API health and status
- ✅ Authentication and authorization
- ✅ Document generation
- ✅ Task management
- ✅ Rate limiting
- ✅ File upload/download
- ✅ Error handling
- ✅ Caching
- ✅ Dashboard integration

## 🔒 Security Features

- **Rate Limiting**: Prevents abuse with configurable limits
- **Authentication**: API key and session-based auth
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error responses
- **CORS**: Configurable cross-origin resource sharing

## 📈 Performance Features

- **Caching**: Redis-based caching for improved response times
- **Async Processing**: Non-blocking background tasks
- **Metrics**: Prometheus integration for monitoring
- **Logging**: Comprehensive logging with rotation
- **Connection Pooling**: Efficient database connections

## 🚀 Production Deployment

### **Docker Deployment** (Coming Soon)

```bash
# Build and run with Docker
docker build -t bul-enhanced .
docker run -p 8000:8000 -p 8050:8050 bul-enhanced
```

### **Environment Setup**

```bash
# Production environment
export DEBUG_MODE=false
export REDIS_HOST=your-redis-host
export API_HOST=0.0.0.0
export API_PORT=8000
```

## 🔄 Migration from Basic API

If migrating from the basic BUL API:

1. **Backup existing data**
2. **Install enhanced dependencies**
3. **Update frontend code** to use new endpoints
4. **Configure authentication** if needed
5. **Test thoroughly** with the enhanced test suite

## 📚 API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of the Blatam Academy system.

## 🆘 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the logs in `bul_enhanced.log`
- Check system status at `/health`
- Monitor performance at `/stats`

---

**BUL Enhanced**: Empowering SMEs with enterprise-grade AI-driven document generation.

## 🎉 What's New in Enhanced Version

- ✅ **Authentication & Authorization**
- ✅ **Rate Limiting & Security**
- ✅ **Real-time Dashboard**
- ✅ **File Upload/Download**
- ✅ **Task Cancellation**
- ✅ **Enhanced Error Handling**
- ✅ **Prometheus Metrics**
- ✅ **Redis Caching**
- ✅ **Comprehensive Testing**
- ✅ **Production Ready**
