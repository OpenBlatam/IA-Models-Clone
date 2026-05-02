# Document Workflow Chain

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 🚀 Description

The **Document Workflow Chain** is a professional and optimized AI system that allows continuous document generation through intelligent prompt chaining. Each generated document automatically becomes the input for generating the next document, creating efficient and scalable workflows.

### ✨ Key Features

- **Continuous Generation**: Create documents uninterruptedly using a single initial query
- **Intelligent Chaining**: Each document automatically feeds the generation of the next
- **Workflow Management**: Complete control over workflow status and progress
- **Title Generation**: Automatically create attractive blog titles
- **Full REST API**: Endpoints for integration with other systems
- **State Persistence**: Maintains the complete history of each document chain
- **Multiple AI Clients**: Support for OpenAI, Anthropic, Cohere, and custom clients
- **Advanced Database**: Full persistence with PostgreSQL and SQLAlchemy
- **Web Dashboard**: Visual interface for workflow monitoring and management
- **Intelligent Integrations**: Automatic connection with other Blatam Academy services
- **Quality Analysis**: Automatic content quality scoring system
- **SEO Optimization**: Automatic integration with SEO services
- **Redundancy Detection**: Automatic verification of unique content
- **Brand Consistency**: Automatic analysis and adjustment of brand tone
- **Content Templates**: Predefined template system for different types of content
- **Multi-language Support**: Content generation in 6 main languages
- **Advanced Analytics**: Predictive analytics and trend system
- **Sentiment Analysis**: Automatic evaluation of content tone and sentiment
- **Prompt Optimization**: Intelligent prompt optimization system
- **Performance Metrics**: Detailed tracking of quality and performance metrics

## 🏗️ Architecture

```
Document Workflow Chain
├── main.py                     # Main FastAPI application
├── start.py                    # Optimized startup script
├── system_config.py            # Centralized system configuration
├── workflow_chain_engine.py    # Main system engine
├── api_endpoints.py            # REST API Endpoints
├── ai_clients.py              # AI client integration
├── database.py                # Database models and operations
├── dashboard.py               # Web monitoring dashboard
├── integrations.py            # Blatam service integrations
├── external_integrations.py   # External service integrations
├── content_analyzer.py        # Advanced content analysis
├── content_templates.py       # Content template system
├── multilang_support.py       # Multi-language support
├── advanced_analytics.py      # Analytics and predictive analysis
├── advanced_analysis.py       # Advanced document analysis
├── content_quality_control.py # Content quality control
├── content_versioning.py      # Versioning system
├── workflow_scheduler.py      # Workflow scheduler
├── workflow_automation.py     # Workflow automation
├── intelligent_generation.py  # Intelligent generation
├── trend_analysis.py          # Trend analysis
├── ai_optimization.py         # AI optimization
├── intelligent_cache.py       # Intelligent cache system
├── test_workflow.py           # Test suite
├── examples/                   # Examples and demos
├── templates/                  # HTML templates for dashboard
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Service orchestration
├── nginx.conf                  # Load balancer configuration
├── requirements.txt            # Python dependencies
├── env.example                 # Environment variables
└── README.md                   # This file
```

## 🚀 Installation and Configuration

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- AI Client (OpenAI, Anthropic, Cohere, etc.)

### Local Installation

```bash
# Clone or navigate to directory
cd document_workflow_chain

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
export AI_API_KEY="your_api_key_here"
export AI_CLIENT_TYPE="openai"  # or anthropic, cohere
export AI_MODEL="gpt-4"         # or claude-3, etc.
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run only the main service
docker build -t document-workflow-chain .
docker run -p 8001:8000 -e AI_API_KEY=your_key document-workflow-chain
```

## 📖 Basic Usage

### 1. Create a Workflow Chain

```python
from workflow_chain_engine import WorkflowChainEngine

# Initialize engine
engine = WorkflowChainEngine(ai_client=your_ai_client)

# Create a new chain
chain = await engine.create_workflow_chain(
    name="Digital Marketing Guide",
    description="Series of articles on digital marketing",
    initial_prompt="Write an introduction to modern digital marketing"
)
```

### 2. Continue the Chain

```python
# Generate the next document automatically
next_doc = await engine.continue_workflow_chain(chain.id)

# Or with a custom prompt
next_doc = await engine.continue_workflow_chain(
    chain.id, 
    continuation_prompt="Now write about SEO strategies"
)
```

### 3. Generate Blog Titles

```python
# Generate an attractive title
title = await engine.generate_blog_title(
    "Blog content about artificial intelligence..."
)
print(title)  # "Artificial Intelligence: The Future of Digital Marketing"
```

## 🎛️ Web Dashboard

### Dashboard Access
- **URL**: `http://localhost:8020/dashboard/`
- **Features**:
  - Real-time visualization of active workflows
  - Performance and quality statistics
  - Chain management (pause, resume, complete)
  - Quality trend charts
  - Token usage and generation time metrics

### Dashboard Functionalities
- **Real-Time Monitoring**: See the status of all your workflows
- **Visual Management**: Control your chains with simple clicks
- **Advanced Analytics**: Performance and quality charts
- **Data Export**: Download statistics and reports

## 🔗 Intelligent Integrations

### Integrated Services
- **Content Redundancy Detector**: Verifies content uniqueness
- **SEO Optimizer**: Automatically optimizes for search engines
- **Brand Voice Analyzer**: Maintains brand consistency
- **Blog Publisher**: Automatically publishes to your blog

### Integration Flow
1. **Generation**: Create content with AI
2. **Verification**: Detect redundancy automatically
3. **Optimization**: Improve SEO and brand consistency
4. **Publication**: Publish or schedule automatically

## 🎨 Content Templates

### Available Template Types
- **Blog Post**: Structured articles with introduction, body, and conclusion
- **Tutorial**: Step-by-step guides with detailed instructions
- **Product Description**: Sales-optimized product descriptions
- **News Article**: News articles with journalistic structure
- **Social Media Post**: Optimized posts for social media

### Template Usage
```python
# Create workflow with template
chain = await engine.create_workflow_with_template(
    template_id="blog_post",
    topic="Artificial Intelligence in Marketing",
    name="AI Marketing Series",
    description="Articles about AI in marketing",
    language_code="en",
    word_count=1500,
    tone="professional",
    audience="marketing professionals"
)
```

## 🌍 Multi-language Support

### Supported Languages
- **English (en)**: Professional and conversational content
- **Spanish (es)**: Warm and friendly content
- **French (fr)**: Elegant and formal content
- **Portuguese (pt)**: Warm and welcoming content
- **German (de)**: Precise and professional content
- **Italian (it)**: Passionate and expressive content

### Cultural Adaptation
- Localized date format
- Regional number format
- Culturally appropriate writing style
- Adapted formality level
- Localized keywords

## 📊 Advanced Analytics

### Performance Metrics
- **Quality Score**: Automatic content quality evaluation
- **Generation Time**: Generation speed metrics
- **Token Usage**: Cost and efficiency tracking
- **Engagement Score**: Engagement potential measurement
- **SEO Score**: Search engine optimization evaluation
- **Readability**: Reading ease analysis

### Predictive Analysis
- **Quality Trends**: Quality trend prediction
- **Cost Optimization**: Recommendations to reduce costs
- **Performance Improvement**: Optimization suggestions
- **Engagement Insights**: Engagement predictions

### Analytics Dashboard
- Real-time metric visualization
- Interactive trend charts
- Automatic performance alerts
- Personalized optimization reports

## 🌐 REST API

### Main Endpoints

#### Create Workflow Chain
```http
POST /api/v1/document-workflow-chain/create
Content-Type: application/json

{
    "name": "My Blog Series",
    "description": "Series about technology",
    "initial_prompt": "Write about the future of AI"
}
```

#### Continue Workflow Chain
```http
POST /api/v1/document-workflow-chain/continue
Content-Type: application/json

{
    "chain_id": "uuid-of-workflow",
    "continuation_prompt": "Continue with the next topic"
}
```

#### Get Chain History
```http
GET /api/v1/document-workflow-chain/chain/{chain_id}/history
```

#### Generate Blog Title
```http
POST /api/v1/document-workflow-chain/generate-title
Content-Type: application/json

{
    "content": "Blog content to generate title..."
}
```

#### Create Workflow with Template
```http
POST /api/v1/document-workflow-chain/create-with-template
Content-Type: application/json

{
    "template_id": "blog_post",
    "topic": "Artificial Intelligence in Marketing",
    "name": "AI Marketing Series",
    "description": "Articles about AI in marketing",
    "language_code": "en",
    "word_count": 1500,
    "tone": "professional",
    "audience": "marketing professionals"
}
```

#### Analyze Content
```http
POST /api/v1/document-workflow-chain/analyze-content
Content-Type: application/json

{
    "content": "Content to analyze...",
    "title": "Document Title",
    "language_code": "en"
}
```

#### Get Performance Analysis
```http
GET /api/v1/document-workflow-chain/chain/{chain_id}/performance
```

#### Get Workflow Insights
```http
GET /api/v1/document-workflow-chain/chain/{chain_id}/insights
```

#### Get Available Templates
```http
GET /api/v1/document-workflow-chain/templates?category=blogging
```

#### Get Supported Languages
```http
GET /api/v1/document-workflow-chain/languages
```

#### Get Analytics Summary
```http
GET /api/v1/document-workflow-chain/analytics/summary
```

### Workflow Management

```http
# Pause workflow
POST /api/v1/document-workflow-chain/chain/{chain_id}/pause

# Resume workflow
POST /api/v1/document-workflow-chain/chain/{chain_id}/resume

# Complete workflow
POST /api/v1/document-workflow-chain/chain/{chain_id}/complete

# Export workflow
GET /api/v1/document-workflow-chain/chain/{chain_id}/export
```

## 🎯 Use Cases

### 1. Automatic Blog Series
```python
# Create a complete blog post series
chain = await engine.create_workflow_chain(
    name="Complete SEO Guide",
    description="Series of 10 articles about SEO",
    initial_prompt="Write a complete introduction to SEO"
)

# Generate 9 more articles automatically
for i in range(9):
    await engine.continue_workflow_chain(chain.id)
```

### 2. Staggered Educational Content
```python
# Create progressive educational content
chain = await engine.create_workflow_chain(
    name="Python Course",
    description="Progressive Python lessons",
    initial_prompt="Write lesson 1: Introduction to Python"
)

# Each lesson builds on the previous one
for lesson in range(2, 11):
    await engine.continue_workflow_chain(
        chain.id,
        f"Write lesson {lesson} based on the previous one"
    )
```

### 3. Technical Documentation
```python
# Generate complete technical documentation
chain = await engine.create_workflow_chain(
    name="API Documentation",
    description="Complete API documentation",
    initial_prompt="Write the introduction to our REST API"
)
```

## 🔧 Advanced Configuration

### Environment Variables

```bash
# AI Configuration
AI_CLIENT_TYPE=openai          # openai, anthropic, cohere
AI_API_KEY=your_api_key
AI_MODEL=gpt-4                 # gpt-4, claude-3, etc.

# Database Configuration
POSTGRES_PASSWORD=your_password
REDIS_URL=redis://localhost:6379

# Logging Configuration
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
```

### Engine Customization

```python
# Custom settings
settings = {
    "max_chain_length": 50,
    "auto_title_generation": True,
    "content_quality_threshold": 0.8,
    "continuation_strategy": "semantic_similarity"
}

chain = await engine.create_workflow_chain(
    name="My Workflow",
    description="Custom workflow",
    initial_prompt="Initial prompt",
    settings=settings
)
```

## 📊 Monitoring and Logs

### Health Check
```http
GET /api/v1/document-workflow-chain/health
```

Response:
```json
{
    "status": "healthy",
    "active_chains": 5,
    "completed_chains": 12,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Structured Logs
The system generates detailed logs for:
- Workflow creation
- Document generation
- Errors and exceptions
- Performance metrics

## 🧪 Testing

### Run Demo
```bash
python examples/demo.py
```

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
pytest tests/integration/
```

## 🚀 Production Deployment

### Docker Compose
```bash
# Complete deployment with database and Redis
docker-compose -f docker-compose.yml up -d
```

### Kubernetes
```yaml
# Example deployment for Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-workflow-chain
spec:
  replicas: 3
  selector:
    matchLabels:
      app: document-workflow-chain
  template:
    metadata:
      labels:
        app: document-workflow-chain
    spec:
      containers:
      - name: document-workflow-chain
        image: document-workflow-chain:latest
        ports:
        - containerPort: 8000
        env:
        - name: AI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-secrets
              key: api-key
```

## 🔒 Security

- JWT Authentication for endpoints
- Rate limiting to prevent abuse
- Input validation on all endpoints
- Audit logs for tracking
- Sensitive data encryption

## 📈 Scalability

- Stateless architecture for horizontal scaling
- Cache with Redis for optimization
- PostgreSQL database for persistence
- Load balancing with Nginx
- Monitoring with Prometheus

## 🤝 Contribution

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is under the MIT License. See the `LICENSE` file for more details.

## 🆘 Support

- **Documentation**: [Project Wiki]
- **Issues**: [GitHub Issues]
- **Discussions**: [GitHub Discussions]
- **Email**: support@blatam-academy.com

## 🎉 Acknowledgments

- Blatam Academy Team
- Developer Community
- Open Source Contributors

---

**Document Workflow Chain** - Transforming content creation with AI 🚀

[← Back to Main README](../README.md)
