# BUL - Business Universal Language (Optimized)

**Modular AI-powered document generation system for SMEs**

BUL is a clean, optimized system that processes business queries and generates professional documents across multiple business areas. The system features a modular architecture for better maintainability and scalability.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Business Area Specialization**: Dedicated agents for different business areas
- **Query Analysis**: Intelligent query processing and routing
- **Document Generation**: Multi-format document creation
- **RESTful API**: Clean API endpoints for integration
- **Real-time Processing**: Asynchronous task processing
- **Configurable**: Flexible configuration system

## 🏗️ Architecture

```
bulk/
├── modules/                    # Core modules
│   ├── __init__.py
│   ├── document_processor.py   # Document generation
│   ├── query_analyzer.py       # Query analysis
│   ├── business_agents.py      # Business area agents
│   └── api_handler.py          # API request handling
├── config_optimized.py         # Configuration management
├── bul_optimized.py           # Main application
├── requirements_optimized.txt  # Dependencies
└── README_OPTIMIZED.md        # This file
```

## 🎯 Business Areas Supported

- **Marketing**: Strategy, campaigns, content, analysis
- **Sales**: Proposals, presentations, playbooks, forecasts
- **Operations**: Manuals, procedures, workflows, reports
- **HR**: Policies, training, job descriptions, evaluations
- **Finance**: Budgets, forecasts, analysis, reports

## 📋 Document Types Generated

- **Strategy Documents**: Business plans, marketing strategies
- **Manuals**: Procedures, workflows, training materials
- **Proposals**: Sales proposals, project proposals
- **Reports**: Analysis reports, performance reviews
- **Policies**: HR policies, operational procedures

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements_optimized.txt
   ```

2. **Configure environment**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your API keys
   ```

3. **Set up API keys** (optional):
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   # or
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

## 🚀 Usage

### Command Line Interface

```bash
# Start the system
python bul_optimized.py

# Start with custom host/port
python bul_optimized.py --host 0.0.0.0 --port 8000

# Start in debug mode
python bul_optimized.py --debug
```

### API Usage

1. **Start the system**:
   ```bash
   python bul_optimized.py
   ```

2. **Submit a query**:
   ```bash
   curl -X POST "http://localhost:8000/documents/generate" \
        -H "Content-Type: application/json" \
        -d '{
          "query": "Create a marketing strategy for a new restaurant",
          "business_area": "marketing",
          "document_type": "strategy",
          "priority": 1
        }'
   ```

3. **Check task status**:
   ```bash
   curl "http://localhost:8000/tasks/{task_id}/status"
   ```

4. **List all tasks**:
   ```bash
   curl "http://localhost:8000/tasks"
   ```

### Python API

```python
from bul_optimized import BULSystem

# Initialize system
system = BULSystem()

# The system is now running with FastAPI
# Use the API endpoints to interact with it
```

## 📊 API Endpoints

### Core Endpoints

- `GET /` - System information
- `GET /health` - Health check
- `POST /documents/generate` - Generate document
- `GET /tasks/{task_id}/status` - Get task status
- `GET /tasks` - List all tasks
- `DELETE /tasks/{task_id}` - Delete task

### Agent Endpoints

- `GET /agents` - Get all available agents
- `GET /agents/{area}` - Get specific agent information

## 🔧 Configuration

### Environment Variables

- `BUL_API_HOST`: API server host (default: 0.0.0.0)
- `BUL_API_PORT`: API server port (default: 8000)
- `BUL_DEBUG`: Enable debug mode (default: false)
- `BUL_MAX_CONCURRENT_TASKS`: Max concurrent tasks (default: 5)
- `BUL_TASK_TIMEOUT`: Task timeout in seconds (default: 300)
- `BUL_OUTPUT_DIR`: Output directory (default: generated_documents)
- `BUL_LOG_LEVEL`: Log level (default: INFO)

### Business Area Configuration

Enable/disable specific business areas in the configuration:

```python
enabled_business_areas = [
    "marketing", "sales", "operations", "hr", "finance"
]
```

## 🔄 Processing Flow

1. **Query Submission**: Users submit business queries via API
2. **Query Analysis**: System analyzes query to determine business area and document type
3. **Agent Selection**: Appropriate business area agent is selected
4. **Document Generation**: Agent processes query and generates document
5. **Storage**: Document is saved to organized directory structure
6. **Response**: Task status and results are returned

## 🎨 Document Templates

The system includes specialized templates for:

- **Marketing**: Strategy frameworks, campaign templates, content guidelines
- **Sales**: Proposal templates, presentation structures, sales processes
- **Operations**: Procedure templates, workflow diagrams, quality standards
- **HR**: Policy templates, training materials, evaluation forms
- **Finance**: Budget templates, financial models, reporting formats

## 🔍 Query Analysis

The system automatically analyzes queries to:

- Determine primary business area
- Identify secondary relevant areas
- Select appropriate document types
- Assess complexity level
- Set processing priority
- Calculate confidence score

## 📈 Monitoring and Statistics

Track system performance with:

- Task processing statistics
- Success/failure rates
- Processing times
- Active/queued tasks
- Business area distribution

## 🛡️ Security and Rate Limiting

- Input validation and sanitization
- Rate limiting per IP address
- Secure document storage
- Error handling and logging
- CORS configuration

## 🚀 Deployment

### Development
```bash
python bul_optimized.py --debug
```

### Production
```bash
python bul_optimized.py --host 0.0.0.0 --port 8000
```

### Docker (Future)
```bash
docker build -t bul-optimized .
docker run -p 8000:8000 bul-optimized
```

## 📝 Example Queries

- "Create a marketing strategy for a new e-commerce store"
- "Develop a sales process for B2B software sales"
- "Write an operational manual for customer service"
- "Create HR policies for remote work"
- "Generate a financial plan for a startup"

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

**BUL - Business Universal Language (Optimized)**: Clean, modular, and efficient document generation for SMEs.

