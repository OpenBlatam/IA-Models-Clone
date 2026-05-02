# Robot Maintenance AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

AI system for teaching maintenance of robots and machines that uses OpenRouter, NLP, and ML with the best available libraries.

## 📚 Quick Documentation

- **[START.md](startup_docs/START.md)** - System quick start
- **[QUICK_REFERENCE.md](startup_docs/QUICK_REFERENCE.md)** - Quick reference for developers

## 🎯 Features

- **Intelligent Teaching**: Tutor system that teaches maintenance procedures for robots and machines
- **Natural Language Processing (NLP)**: Uses spaCy and transformers to understand maintenance queries
- **Machine Learning (ML)**: Predictive maintenance prediction using scikit-learn
- **OpenRouter Integration**: Access to advanced language models
- **Intelligent Diagnosis**: Symptom analysis and recommendations
- **Maintenance Prediction**: ML to predict when maintenance is needed
- **REST API**: Easy-to-use endpoints for integration
- **Cache System**: In-memory cache with TTL and LRU to improve performance
- **Exponential Backoff Retry**: Automatic retries with exponential backoff for greater robustness
- **Input Validation**: Complete validation of all inputs with clear error messages
- **Metrics and Monitoring**: Metric system to monitor performance and API usage
- **Enhanced Error Handling**: Robust error handling with appropriate HTTP codes
- **Rate Limiting**: Rate limiting system to protect the API (100 req/min per IP)
- **Enhanced Logging**: Configurable logging system with file support
- **Unit Tests**: Complete test suite for validation, cache, and rate limiting
- **API Documentation**: Complete API reference with examples
- **Request Logging Middleware**: Middleware for automatic logging of all requests
- **Detailed Health Check**: Health check with full system information
- **Conversation Export**: Export conversations in JSON and CSV
- **Report Generation**: Generate maintenance reports from conversations
- **Configured CORS**: CORS support for frontend integration
- **Docker Support**: Dockerfile and docker-compose for easy deployment
- **YAML Configuration**: Support for YAML configuration files
- **Startup Scripts**: Scripts to easily start the server
- **Data Persistence**: SQLite database for conversations and records
- **WebSockets**: Real-time updates via WebSockets
- **Performance Optimizations**: Timing decorators and batch processing
- **Security Utilities**: Input sanitization and enhanced validation
- **Authentication System**: API keys for access control
- **Notification System**: Notifications for alerts and updates
- **Analytics API**: Complete dashboard with metrics and advanced statistics
- **Advanced Search**: Search in conversations and maintenance logs
- **Batch Operations**: Efficient processing of multiple operations
- **Plugin Management**: Complete API for plugin management and execution
- **Alert System**: Intelligent alerts based on sensor analysis and ML
- **Intelligent Recommendations**: AI-based maintenance recommendation system
- **Incident Management**: Complete system for maintenance tickets and incidents
- **Comparison and Benchmarking**: Robot comparison and performance analysis
- **Advanced Reports**: Custom report generation (summary, detailed, predictive, costs)
- **Continuous Learning**: Feedback system and continuous improvement of ML models
- **Real-Time Dashboard**: Complete dashboard with widgets and real-time metrics
- **Webhook System**: External integrations via webhooks with system events
- **Advanced Export**: Export in multiple formats (JSON, CSV, Excel) with filters
- **Dynamic Configuration**: Runtime configuration management with validation
- **Advanced Monitoring**: Complete system monitoring with automatic alerts and resource metrics
- **Audit System**: Complete log of system activities with analysis and statistics
- **Maintenance Templates**: Reusable template system for maintenance procedures
- **Advanced Validation**: Data validation with customizable rules and batch validation

## 📋 Requirements

- Python 3.8+
- Open Router API Key (`OPENROUTER_API_KEY`)

## 🚀 Installation

### Option 1: Local Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install the Spanish spaCy model (optional but recommended):

```bash
python -m spacy download es_core_news_sm
```

3. Configure the environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
OPENROUTER_API_KEY=your-api-key-here
```

### Option 2: Docker (Recommended for Production)

See [docs/DOCKER.md](docs/DOCKER.md) for full instructions.

**Quick start with Docker Compose:**

```bash
# Copy example configuration
cp config/config.yaml.example config/config.yaml

# Configure API key
export OPENROUTER_API_KEY="your-api-key-here"

# Start service
docker-compose up -d
```

### Option 3: Startup Scripts

**Linux/Mac:**
```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

**Windows:**
```cmd
scripts\start.bat
```

## 💻 Basic Usage

### Usage as Python Module

```python
import asyncio
from robot_maintenance_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    # Ask a maintenance question
    response = await tutor.ask_maintenance_question(
        question="How do I change the oil of an industrial robot?",
        robot_type="industrial"
    )
    
    print(response["answer"])
    
    # Diagnosis with sensor data
    diagnosis = await tutor.diagnose_problem(
        symptoms="The robot makes strange noises",
        robot_type="industrial",
        sensor_data={
            "temperature": 85.0,
            "vibration": 6.5,
            "runtime_hours": 1500
        }
    )
    
    print(diagnosis["answer"])
    print(f"ML Prediction: {diagnosis['ml_prediction']}")
    
    await tutor.close()

asyncio.run(main())
```

### Usage with REST API

1. Start the server:

```bash
python main.py
```

2. Use the endpoints:

```bash
# Ask a question
curl -X POST http://localhost:8000/api/robot-maintenance/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I change the oil of an industrial robot?",
    "robot_type": "industrial"
  }'

# Diagnosis
curl -X POST http://localhost:8000/api/robot-maintenance/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "The robot makes strange noises and vibrates a lot",
    "robot_type": "industrial",
    "sensor_data": {
      "temperature": 85.0,
      "vibration": 6.5
    }
  }'

# Maintenance prediction
curl -X POST http://localhost:8000/api/robot-maintenance/predict \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial",
    "sensor_data": {
      "temperature": 75.0,
      "vibration": 4.0,
      "runtime_hours": 800
    }
  }'

# Generate checklist
curl -X POST http://localhost:8000/api/robot-maintenance/checklist \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial",
    "maintenance_type": "preventive"
  }'
```

## 📚 Project Structure

```
robot_maintenance_ai/
├── __init__.py
├── README.md
├── requirements.txt
├── main.py
├── config/
│   ├── __init__.py
│   └── maintenance_config.py      # System configuration
├── core/
│   ├── __init__.py
│   ├── maintenance_tutor.py       # Main tutor class
│   ├── nlp_processor.py          # NLP processor
│   ├── ml_predictor.py           # ML predictor
│   └── conversation_manager.py   # Conversation management
├── api/
│   ├── __init__.py
│   └── maintenance_api.py        # FastAPI endpoints
├── utils/
│   ├── __init__.py
│   └── helpers.py                # Helper functions
└── examples/
    └── basic_usage.py            # Usage examples
```

## ⚙️ Configuration

You can customize the configuration:

```python
from robot_maintenance_ai import MaintenanceConfig, OpenRouterConfig, NLPConfig, MLConfig

openrouter_config = OpenRouterConfig(
    api_key="your-api-key",
    default_model="openai/gpt-4-turbo",
    temperature=0.7,
    max_tokens=3000
)

nlp_config = NLPConfig(
    language="es",
    use_spacy=True,
    use_transformers=True
)

ml_config = MLConfig(
    enable_predictive_maintenance=True,
    model_path="ml_models"
)

config = MaintenanceConfig(
    openrouter=openrouter_config,
    nlp=nlp_config,
    ml=ml_config
)
```

## 🤖 Supported Robot Types

- Industrial
- Collaborative
- Medical
- Agricultural
- Logistics
- Domestic
- Military
- Space

## 🔧 Maintenance Types

- **Preventive**: Regular scheduled maintenance
- **Corrective**: Repair after failure
- **Predictive**: Based on ML prediction
- **Emergency**: Critical situations
- **Calibration**: Parameter adjustment
- **Cleaning**: Cleaning maintenance
- **Lubrication**: Lubrication maintenance
- **Inspection**: Review and evaluation

## 📊 ML Features

The system includes:

- **Maintenance Prediction**: Predicts when maintenance is needed based on sensor data
- **Anomaly Detection**: Identifies anomalous patterns in sensor data
- **Intelligent Recommendations**: Suggests actions based on ML analysis
- **Failure Time Estimation**: Predicts time until possible failure

## 🚀 Advanced Features

### Cache System
- In-memory cache with configurable TTL (Time-To-Live)
- Automatic LRU (Least Recently Used) eviction
- Hit/miss rate statistics
- `/cache/stats` endpoint to query statistics

### Exponential Backoff Retry
- Automatic retries for failed API calls
- Configurable exponential backoff
- Robust handling of timeouts and connection errors

### Input Validation
- Complete validation of all input parameters
- Clear and descriptive error messages
- Input sanitization to prevent injections
- Validation with Pydantic v2

### Metrics and Monitoring
- Tracking of all API requests
- Performance statistics (response time, error rate)
- Cache metrics (hit rate, miss rate)
- `/metrics` endpoint to query real-time statistics

### Rate Limiting
- Limit of 100 requests per minute per IP (configurable)
- Token bucket algorithm
- `Retry-After` headers when limit is exceeded
- `/rate-limit/stats` and `/rate-limit/reset` endpoints

### Enhanced Logging
- Configurable logging system
- Support for file logging
- Structured format with detailed information
- Configuration via environment variables

### Enhanced Error Handling
- Appropriate HTTP codes for different error types
- Detailed logging for debugging
- Specific handling of timeouts, connection errors, and validation
- Error tracking in metrics

### Unit Tests
- Complete test suite for validators
- Tests for cache system
- Tests for rate limiting
- Shared fixtures for testing

### Authentication System
- API key management
- Token validation
- User permissions
- API key revocation
- Optional protected endpoints

### Notification System
- Notifications for maintenance events
- Subscription to notification types
- Notification management per user
- Marking notifications as read
- Notification cleanup

## 🔬 Libraries Used

### NLP
- **spaCy**: Natural language processing
- **transformers**: Advanced language models
- **torch**: Deep learning framework

### ML
- **scikit-learn**: Classical machine learning
- **pandas**: Data manipulation
- **numpy**: Numerical calculations
- **joblib**: Model serialization

### API
- **FastAPI**: Modern web framework
- **httpx**: Asynchronous HTTP client
- **uvicorn**: ASGI server

## 🧪 Testing

Run tests with:

```bash
# All tests
pytest tests/

# Specific tests
pytest tests/test_validators.py
pytest tests/test_cache_manager.py
pytest tests/test_rate_limiter.py

# With coverage
pytest tests/ --cov=. --cov-report=html
```

See `tests/README.md` for more information.

## 📖 Examples

See `examples/basic_usage.py` for complete usage examples.

## 📚 Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete API reference
- **[WEBSOCKETS.md](docs/WEBSOCKETS.md)** - WebSockets guide
- **[AUTHENTICATION.md](docs/AUTHENTICATION.md)** - Authentication guide
- **[DOCKER.md](docs/DOCKER.md)** - Docker guide
- **[START.md](startup_docs/START.md)** - Quick start guide
- **[QUICK_REFERENCE.md](startup_docs/QUICK_REFERENCE.md)** - Quick reference

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of Blatam Academy.

## 🆘 Support

For support, open an issue in the repository or contact the Blatam Academy team.

## 🎓 Additional Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [spaCy Documentation](https://spacy.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
