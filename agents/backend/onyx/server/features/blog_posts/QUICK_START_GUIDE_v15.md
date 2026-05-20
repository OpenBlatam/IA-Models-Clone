# Quick Start Guide - Enhanced Blog System v15.0.0

## 🚀 Getting Started in 5 Minutes

This guide will help you set up and run the Enhanced Blog System v15.0.0 with all its advanced features including real-time collaboration, AI content generation, and advanced analytics.

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **PostgreSQL**: 14 or higher
- **Redis**: 6 or higher
- **Elasticsearch**: 8 or higher
- **OpenAI API Key**: For AI content generation

### Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install python3.11 python3.11-pip python3.11-venv

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Install Redis
sudo apt install redis-server

# Install Elasticsearch
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
sudo apt update
sudo apt install elasticsearch
```

#### macOS
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 postgresql redis elasticsearch
```

#### Windows
```bash
# Install Python from python.org
# Install PostgreSQL from postgresql.org
# Install Redis from redis.io
# Install Elasticsearch from elastic.co
```

## Installation Steps

### 1. Clone and Setup Project
```bash
# Navigate to your project directory
cd agents/backend/onyx/server/features/blog_posts

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements-enhanced-v15.txt
```

### 2. Environment Configuration
```bash
# Create environment file
cp .env.example .env

# Edit environment variables
nano .env
```

#### Required Environment Variables
```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost/blog_db
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-super-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI/ML Configuration
OPENAI_API_KEY=your-openai-api-key-here
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Search Configuration
ELASTICSEARCH_URL=http://localhost:9200
SEARCH_INDEX_NAME=blog_posts

# Performance Configuration
CACHE_TTL=3600
MAX_CONCURRENT_REQUESTS=100
BATCH_SIZE=32

# Real-time Configuration
WEBSOCKET_PING_INTERVAL=20
WEBSOCKET_PING_TIMEOUT=20

# Monitoring
SENTRY_DSN=your-sentry-dsn-here
ENABLE_METRICS=true
```

### 3. Database Setup
```bash
# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql

# In PostgreSQL prompt:
CREATE DATABASE blog_db;
CREATE USER blog_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE blog_db TO blog_user;
\q

# Initialize database tables
python -c "
from ENHANCED_BLOG_SYSTEM_v15 import Base, engine
Base.metadata.create_all(bind=engine)
print('Database tables created successfully!')
"
```

### 4. Start Services
```bash
# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Start Elasticsearch
sudo systemctl start elasticsearch
sudo systemctl enable elasticsearch

# Verify services are running
redis-cli ping  # Should return PONG
curl http://localhost:9200  # Should return Elasticsearch info
```

### 5. Run the Application
```bash
# Start the FastAPI application
python ENHANCED_BLOG_SYSTEM_v15.py
```

The application will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## 🎯 Quick Demo

### Run the Comprehensive Demo
```bash
# Run the interactive demo
python enhanced_demo_v15.py
```

This demo will showcase:
- 🤖 AI content generation
- 👥 Real-time collaboration
- 📊 Advanced analytics
- ⚡ Performance features
- 🔒 Security features
- 🔗 Integration capabilities

## 🧪 Testing the System

### 1. Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "15.0.0"
}
```

### 2. AI Content Generation
```bash
curl -X POST "http://localhost:8000/ai/generate-content" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "artificial intelligence trends",
    "style": "professional",
    "length": "medium",
    "tone": "informative"
  }'
```

### 3. Real-time Collaboration
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/collaborate/1');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// Send collaboration message
ws.send(JSON.stringify({
    type: 'content_update',
    content: 'Updated content here',
    user_id: 'user123'
}));
```

### 4. Analytics
```bash
curl -X POST "http://localhost:8000/analytics" \
  -H "Content-Type: application/json" \
  -d '{
    "date_from": "2024-01-01T00:00:00Z",
    "date_to": "2024-01-31T23:59:59Z"
  }'
```

## 📊 Monitoring

### Prometheus Metrics
```bash
# View metrics
curl http://localhost:8000/metrics
```

### Key Metrics to Monitor
- `blog_posts_created_total`: Total posts created
- `blog_posts_read_total`: Total posts read
- `real_time_collaborators`: Active collaborators
- `ai_content_generated_total`: AI content generations
- `blog_posts_processing_seconds`: Processing time

### Grafana Dashboard
```bash
# Install Grafana
sudo apt install grafana

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

## 🔧 Configuration Options

### Performance Tuning
```python
# In your .env file
CACHE_TTL=3600  # Cache time-to-live in seconds
MAX_CONCURRENT_REQUESTS=100  # Max concurrent requests
BATCH_SIZE=32  # Batch size for operations
```

### AI Configuration
```python
# OpenAI API settings
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4
OPENAI_MAX_TOKENS=1000
```

### Real-time Settings
```python
# WebSocket configuration
WEBSOCKET_PING_INTERVAL=20  # Ping interval in seconds
WEBSOCKET_PING_TIMEOUT=20   # Ping timeout in seconds
MAX_COLLABORATORS_PER_POST=10  # Max collaborators per post
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Error
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -h localhost -U blog_user -d blog_db
```

#### 2. Redis Connection Error
```bash
# Check Redis status
sudo systemctl status redis

# Test Redis connection
redis-cli ping
```

#### 3. Elasticsearch Connection Error
```bash
# Check Elasticsearch status
sudo systemctl status elasticsearch

# Test Elasticsearch
curl http://localhost:9200
```

#### 4. OpenAI API Error
```bash
# Verify API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.openai.com/v1/models
```

### Performance Issues

#### High Response Times
```python
# Increase cache TTL
CACHE_TTL=7200  # 2 hours

# Optimize database queries
# Add database indexes
```

#### Memory Issues
```python
# Reduce batch size
BATCH_SIZE=16

# Increase worker processes
WORKERS=4
```

## 📚 Next Steps

### 1. Production Deployment
- Set up proper SSL certificates
- Configure load balancer
- Set up monitoring and alerting
- Implement backup strategies

### 2. Customization
- Customize AI prompts
- Add custom analytics
- Implement custom authentication
- Extend API endpoints

### 3. Scaling
- Set up database replication
- Implement Redis clustering
- Configure Elasticsearch cluster
- Deploy with Kubernetes

## 🆘 Support

### Documentation
- **API Docs**: http://localhost:8000/docs
- **System Summary**: `ENHANCED_BLOG_SYSTEM_SUMMARY_v15.md`
- **Demo Script**: `enhanced_demo_v15.py`

### Community
- **GitHub Issues**: Report bugs and feature requests
- **Discord**: Real-time support and discussions
- **Documentation**: Comprehensive guides and tutorials

### Enterprise Support
- **Email**: support@enhancedblog.com
- **Phone**: +1-555-ENHANCED
- **Slack**: Enterprise support channel

---

**🎉 Congratulations!** You've successfully set up the Enhanced Blog System v15.0.0 with all its advanced features. Start exploring the real-time collaboration, AI content generation, and advanced analytics capabilities! 