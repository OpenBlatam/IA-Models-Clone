# BUL - Business Unlimited

**Advanced AI-powered document generation system for SMEs using OpenRouter and LangChain**

BUL (Business Unlimited) is a comprehensive system that continuously processes business queries and generates professional documents across all business areas for Small and Medium Enterprises (SMEs). The system keeps working until manually stopped, providing real-time document generation capabilities.

## 🚀 Features

- **Continuous Processing**: Keeps working until manually stopped
- **OpenRouter Integration**: Uses multiple AI models via OpenRouter
- **LangChain Integration**: Advanced prompt engineering and document generation
- **SME-Focused**: Specialized agents for different business areas
- **Multi-Format Output**: Supports Markdown, HTML, PDF, DOCX, and more
- **Real-time API**: RESTful API for query submission and document retrieval
- **Intelligent Analysis**: Automatic query analysis and business area detection
- **Comprehensive Templates**: Pre-built templates for various document types

## 🏗️ Architecture

```
bul/
├── core/                    # Core system components
│   ├── bul_engine.py       # Main BUL engine
│   └── continuous_processor.py  # Continuous processing system
├── agents/                  # Business area agents
│   └── sme_agent_manager.py # SME agent management
├── api/                     # API endpoints
│   └── bul_api.py          # FastAPI REST API
├── config/                  # Configuration
│   ├── bul_config.py       # Main configuration
│   └── openrouter_config.py # OpenRouter configuration
├── utils/                   # Utility classes
│   ├── document_processor.py # Document processing
│   └── query_analyzer.py    # Query analysis
├── templates/               # Document templates
├── main.py                  # Main entry point
└── requirements.txt         # Dependencies
```

## 🎯 Business Areas Supported

- **Marketing**: Strategy, campaigns, social media, brand guidelines
- **Sales**: Strategy, proposals, playbooks, customer management
- **Operations**: Manuals, workflows, quality management
- **HR**: Policies, training, recruitment, performance
- **Finance**: Planning, budgets, analysis, reporting
- **Legal**: Compliance, contracts, policies, risk management
- **Technical**: Documentation, systems, security, automation
- **Content**: Writing, blogs, manuals, training materials
- **Strategy**: Business plans, roadmaps, initiatives
- **Customer Service**: Support, satisfaction, retention

## 📋 Document Types Generated

- **Strategy Documents**: Business plans, marketing strategies, operational plans
- **Manuals**: Procedures, workflows, training materials
- **Templates**: Proposals, contracts, forms, checklists
- **Analysis Reports**: Market analysis, financial reports, performance reviews
- **Policies**: HR policies, operational procedures, compliance documents

## 🛠️ Installation

1. **Clone or navigate to the BUL directory**:
   ```bash
   cd C:\blatam-academy\agents\backend\onyx\server\features\bul
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your OpenRouter API key
   ```

4. **Set up OpenRouter API key**:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

## 🚀 Usage

### Command Line Interface

```bash
# Start full system (processor + API)
python main.py --mode full

# Start only continuous processor
python main.py --mode processor

# Start only API server
python main.py --mode api --port 8000

# Start with debug mode
python main.py --mode full --debug
```

### API Usage

1. **Start the system**:
   ```bash
   python main.py --mode full
   ```

2. **Submit a query**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{
          "query": "Create a marketing strategy for a new restaurant",
          "priority": 1
        }'
   ```

3. **Check task status**:
   ```bash
   curl "http://localhost:8000/task/{task_id}/status"
   ```

4. **Get generated documents**:
   ```bash
   curl "http://localhost:8000/task/{task_id}/documents"
   ```

### Python API

```python
from bul import BULEngine, ContinuousProcessor

# Initialize system
engine = BULEngine()
processor = ContinuousProcessor()

# Start continuous processing
await processor.start()

# Submit query
task_id = await engine.submit_query(
    query="Create a sales strategy for B2B software",
    priority=1
)

# Get results
documents = engine.get_completed_documents(task_id)
```

## 🔧 Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `BUL_DEBUG`: Enable debug mode (default: false)
- `BUL_API_HOST`: API server host (default: 0.0.0.0)
- `BUL_API_PORT`: API server port (default: 8000)
- `BUL_MAX_CONCURRENT_TASKS`: Max concurrent processing tasks (default: 5)

### Business Area Configuration

Enable/disable specific business areas in `config/bul_config.py`:

```python
enabled_areas = [
    "marketing", "sales", "operations", "hr", "finance",
    "legal", "technical", "content", "strategy", "customer_service"
]
```

## 📊 API Endpoints

### Core Endpoints

- `POST /query` - Submit a business query
- `GET /task/{task_id}/status` - Get task status
- `GET /task/{task_id}/documents` - Get generated documents
- `GET /documents` - List all documents
- `GET /search` - Search documents
- `GET /stats` - Get processing statistics

### Management Endpoints

- `GET /agents` - Get available business area agents
- `POST /processor/start` - Start continuous processor
- `POST /processor/stop` - Stop continuous processor
- `GET /processor/status` - Get processor status
- `GET /health` - Health check

## 🔄 Continuous Processing

The system operates in continuous mode by default:

1. **Query Submission**: Users submit business queries via API
2. **Query Analysis**: System analyzes query to determine business area and document types
3. **Task Creation**: Creates processing tasks with appropriate priority
4. **Document Generation**: Uses AI models to generate comprehensive documents
5. **Storage**: Saves documents in organized directory structure
6. **Continuous Loop**: Keeps processing until manually stopped

## 🎨 Document Templates

The system includes pre-built templates for:

- **Business Plans**: Executive summary, market analysis, financial projections
- **Marketing Strategies**: Target audience, marketing mix, budget allocation
- **Sales Proposals**: Client solutions, pricing, implementation plans
- **Operational Manuals**: Procedures, workflows, quality standards
- **HR Policies**: Employee guidelines, procedures, compliance

## 🔍 Query Analysis

The system automatically analyzes queries to:

- Determine primary business area
- Identify secondary relevant areas
- Select appropriate document types
- Assess complexity level
- Set processing priority
- Estimate processing time

## 📈 Monitoring and Statistics

Track system performance with:

- Total tasks processed
- Success/failure rates
- Average processing times
- Active/queued tasks
- Document generation statistics
- Business area distribution

## 🛡️ Security and Rate Limiting

- API key authentication (optional)
- Rate limiting per IP address
- Input validation and sanitization
- Secure document storage
- Error handling and logging

## 🚀 Deployment

### Development
```bash
python main.py --mode full --debug
```

### Production
```bash
python main.py --mode full --host 0.0.0.0 --port 8000
```

### Docker (Future)
```bash
docker build -t bul-system .
docker run -p 8000:8000 bul-system
```

## 📝 Example Queries

- "Create a marketing strategy for a new e-commerce store"
- "Develop a sales process for B2B software sales"
- "Write an operational manual for customer service"
- "Create HR policies for remote work"
- "Generate a financial plan for a startup"
- "Develop a content strategy for social media"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the Blatam Academy system.

## 🆘 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the logs in `bul.log`
- Check system status at `/health`

---

**BUL - Business Unlimited**: Empowering SMEs with AI-driven document generation.

