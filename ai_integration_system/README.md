# AI Integration System

A comprehensive system for integrating AI-generated content directly with CMS, CRM, and marketing platforms including Salesforce, Mailchimp, WordPress, HubSpot, and more.

## 🚀 Features

- **Multi-Platform Integration**: Seamlessly distribute content across multiple platforms
- **Automated Workflows**: Intelligent content routing and processing
- **Real-time Monitoring**: Track integration status and performance
- **Webhook Support**: Handle platform events and notifications
- **Error Handling**: Robust retry mechanisms and error recovery
- **Scalable Architecture**: Built for high-volume content processing
- **RESTful API**: Easy integration with existing systems

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Content    │───▶│  Integration     │───▶│   Platforms     │
│   Generator     │    │     Engine       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Monitoring &   │
                       │   Analytics      │
                       └──────────────────┘
```

## 📋 Supported Platforms

### CRM Systems
- **Salesforce**: Content documents, campaigns, leads, opportunities
- **HubSpot**: Blog posts, email campaigns, landing pages, contacts

### Email Marketing
- **Mailchimp**: Email campaigns, templates, automations

### Content Management
- **WordPress**: Posts, pages, custom post types

### Communication
- **Slack**: Messages, channels, file sharing

### Productivity Suites
- **Google Workspace**: Docs, Sheets, Drive, Gmail
- **Microsoft 365**: Word, Excel, Teams, SharePoint

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 13+
- Redis 6+
- Docker & Docker Compose (optional)

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd ai_integration_system

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start the application
uvicorn api_endpoints:app --host 0.0.0.0 --port 8000
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ai_integration

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here

# Salesforce
SALESFORCE__ENABLED=true
SALESFORCE__BASE_URL=https://your-instance.salesforce.com
SALESFORCE__CLIENT_ID=your_client_id
SALESFORCE__CLIENT_SECRET=your_client_secret
SALESFORCE__USERNAME=your_username
SALESFORCE__PASSWORD=your_password
SALESFORCE__SECURITY_TOKEN=your_security_token

# Mailchimp
MAILCHIMP__ENABLED=true
MAILCHIMP__API_KEY=your_api_key
MAILCHIMP__SERVER_PREFIX=us1
MAILCHIMP__LIST_ID=your_list_id

# WordPress
WORDPRESS__ENABLED=true
WORDPRESS__BASE_URL=https://your-site.com
WORDPRESS__USERNAME=your_username
WORDPRESS__PASSWORD=your_password

# HubSpot
HUBSPOT__ENABLED=true
HUBSPOT__API_KEY=your_api_key
HUBSPOT__PORTAL_ID=your_portal_id
```

## 📖 Usage

### Basic Integration

```python
from ai_integration_system import IntegrationRequest, ContentType, integration_engine

# Create an integration request
request = IntegrationRequest(
    content_id="blog_post_001",
    content_type=ContentType.BLOG_POST,
    content_data={
        "title": "AI Integration Best Practices",
        "content": "Your blog post content here...",
        "author": "AI Assistant",
        "tags": ["AI", "Integration", "Best Practices"]
    },
    target_platforms=["wordpress", "hubspot", "salesforce"],
    priority=1
)

# Add to integration queue
await integration_engine.add_integration_request(request)

# Process the request
await integration_engine.process_single_request(request)

# Check status
status = await integration_engine.get_integration_status("blog_post_001")
print(status)
```

### API Usage

#### Create Integration Request

```bash
curl -X POST "http://localhost:8000/ai-integration/integrate" \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "blog_post_001",
    "content_type": "blog_post",
    "content_data": {
      "title": "AI Integration Best Practices",
      "content": "Your content here...",
      "author": "AI Assistant",
      "tags": ["AI", "Integration"]
    },
    "target_platforms": ["wordpress", "hubspot"],
    "priority": 1
  }'
```

#### Check Integration Status

```bash
curl -X GET "http://localhost:8000/ai-integration/status/blog_post_001"
```

#### Get Available Platforms

```bash
curl -X GET "http://localhost:8000/ai-integration/platforms"
```

#### Test Platform Connection

```bash
curl -X POST "http://localhost:8000/ai-integration/platforms/salesforce/test"
```

### Bulk Integration

```python
# Create multiple requests
bulk_requests = [
    IntegrationRequest(
        content_id=f"content_{i}",
        content_type=ContentType.BLOG_POST,
        content_data={"title": f"Article {i}", "content": f"Content {i}"},
        target_platforms=["wordpress"]
    )
    for i in range(1, 11)
]

# Add all requests
for request in bulk_requests:
    await integration_engine.add_integration_request(request)

# Process all requests
await integration_engine.process_integration_queue()
```

## 🔧 Platform-Specific Configuration

### Salesforce Setup

1. Create a Connected App in Salesforce
2. Generate OAuth credentials
3. Configure the following in your `.env`:
   ```env
   SALESFORCE__BASE_URL=https://your-instance.salesforce.com
   SALESFORCE__CLIENT_ID=your_connected_app_client_id
   SALESFORCE__CLIENT_SECRET=your_connected_app_secret
   SALESFORCE__USERNAME=your_salesforce_username
   SALESFORCE__PASSWORD=your_salesforce_password
   SALESFORCE__SECURITY_TOKEN=your_security_token
   ```

### Mailchimp Setup

1. Get your API key from Mailchimp
2. Find your server prefix (e.g., "us1", "us2")
3. Get your list/audience ID
4. Configure in your `.env`:
   ```env
   MAILCHIMP__API_KEY=your_api_key
   MAILCHIMP__SERVER_PREFIX=us1
   MAILCHIMP__LIST_ID=your_list_id
   ```

### WordPress Setup

1. Enable REST API in WordPress
2. Create an application password
3. Configure in your `.env`:
   ```env
   WORDPRESS__BASE_URL=https://your-site.com
   WORDPRESS__USERNAME=your_username
   WORDPRESS__APPLICATION_PASSWORD=your_app_password
   ```

### HubSpot Setup

1. Create a private app in HubSpot
2. Generate an API key
3. Get your portal ID
4. Configure in your `.env`:
   ```env
   HUBSPOT__API_KEY=your_api_key
   HUBSPOT__PORTAL_ID=your_portal_id
   ```

## 📊 Monitoring and Analytics

### Health Check

```bash
curl -X GET "http://localhost:8000/ai-integration/health"
```

### Queue Status

```bash
curl -X GET "http://localhost:8000/ai-integration/queue/status"
```

### Prometheus Metrics

Access metrics at: `http://localhost:9090`

### Grafana Dashboard

Access dashboard at: `http://localhost:3000` (admin/admin)

## 🔄 Webhook Handling

The system supports webhooks from integrated platforms:

```python
# Webhook endpoint
POST /ai-integration/webhooks/{platform}

# Example webhook payload
{
  "event_type": "content.created",
  "object_id": "12345",
  "platform": "salesforce",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_integration_system

# Run specific test file
pytest tests/test_integration_engine.py
```

### Demo Script

```bash
# Run the demo
python examples/demo.py
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**:
   ```bash
   export ENVIRONMENT=production
   export DATABASE_URL=postgresql://user:pass@prod-db:5432/ai_integration
   export REDIS_URL=redis://prod-redis:6379/0
   ```

2. **Docker Deployment**:
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

3. **Kubernetes Deployment**:
   ```bash
   kubectl apply -f k8s/
   ```

### Scaling

- **Horizontal Scaling**: Run multiple API instances behind a load balancer
- **Worker Scaling**: Increase Celery worker processes
- **Database Scaling**: Use read replicas for better performance

## 🔒 Security

- **Authentication**: JWT-based authentication
- **Authorization**: Role-based access control
- **API Security**: Rate limiting and request validation
- **Data Encryption**: TLS for data in transit, encryption at rest
- **Secrets Management**: Environment variables and secret stores

## 📈 Performance

- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Caching**: Redis for session and data caching
- **Queue Management**: Celery for background task processing
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: [Wiki](link-to-wiki)
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: support@aiintegration.com

## 🗺️ Roadmap

- [ ] Additional platform connectors (Shopify, WooCommerce, etc.)
- [ ] Advanced workflow automation
- [ ] Machine learning for content optimization
- [ ] Real-time collaboration features
- [ ] Advanced analytics and reporting
- [ ] Mobile application
- [ ] Enterprise SSO integration

---

**Made with ❤️ by the AI Integration Team**



























