# Robot Maintenance Teaching AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

Advanced AI system for teaching robot and machine maintenance using OpenRouter, NLP, and Machine Learning with the best available libraries.

## 🎯 Features

- **Personalized Teaching**: Maintenance procedures adapted to robot type and difficulty level
- **Intelligent Diagnosis**: Problem analysis based on symptoms using AI
- **Advanced NLP**: Natural language processing with spaCy and Transformers
- **ML Prediction**: Machine learning models to predict maintenance needs
- **Multiple Robot Types**: Support for industrial, service, collaborative, mobile, medical, and agricultural robots
- **Maintenance Schedules**: Automatic generation of maintenance calendars
- **REST API**: Complete endpoints for integration
- **Component Analysis**: Detailed explanations of components and their procedures
- **Cache System**: Intelligent cache for API responses, improving performance
- **Conversation History**: Storage and retrieval of interaction history
- **Robust Error Handling**: Retry logic with exponential backoff and improved error handling
- **Input Validation**: Complete validation of input parameters
- **Async Context Manager**: Support for automatic resource management
- **ML Training Endpoint**: API for training machine learning models
- **Enhanced Logging**: Complete logging system for debugging and monitoring

## 📋 Requirements

- Python 3.8+
- OpenRouter API Key (`OPENROUTER_API_KEY`)
- 8GB+ RAM recommended for NLP/ML models

## 🚀 Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install spaCy model

```bash
python -m spacy download es_core_news_md
```

### 3. Configure environment variables

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
OPENROUTER_API_KEY=your-api-key-here
```

## 💻 Basic Usage

### Usage as Python Module

#### Basic Usage

```python
import asyncio
from robot_maintenance_teaching_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    # Teach maintenance procedure
    result = await tutor.teach_maintenance_procedure(
        robot_type="industrial_robot",
        maintenance_type="preventive",
        difficulty="intermediate"
    )
    print(result["content"])
    
    # Diagnose problem
    diagnosis = await tutor.diagnose_problem(
        symptoms="The robot makes strange noises",
        robot_type="industrial_robot"
    )
    print(diagnosis["content"])
    
    # Explain component
    explanation = await tutor.explain_component(
        component_name="speed reducer",
        robot_type="industrial_robot"
    )
    print(explanation["content"])
    
    # Generate maintenance schedule
    schedule = await tutor.generate_maintenance_schedule(
        robot_type="industrial_robot",
        usage_hours=8
    )
    print(schedule["content"])
    
    # Get conversation history
    history = tutor.get_conversation_history(limit=10)
    print(f"History: {len(history)} conversations")
    
    await tutor.close()

asyncio.run(main())
```

#### Usage with Async Context Manager (Recommended)

```python
import asyncio
from robot_maintenance_teaching_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    config = MaintenanceConfig()
    
    # Use async context manager for automatic resource management
    async with RobotMaintenanceTutor(config) as tutor:
        result = await tutor.teach_maintenance_procedure(
            robot_type="industrial_robot",
            maintenance_type="preventive",
            difficulty="intermediate"
        )
        print(result["content"])
        # Resources are automatically closed when exiting the block

asyncio.run(main())
```

### Usage with REST API

1. Start the server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn api.maintenance_api:app --host 0.0.0.0 --port 8000
```

2. Use the endpoints:

```bash
# Teach maintenance procedure
curl -X POST http://localhost:8000/api/teach \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial_robot",
    "maintenance_type": "preventive",
    "difficulty": "intermediate"
  }'

# Diagnose problem
curl -X POST http://localhost:8000/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "The robot makes strange noises",
    "robot_type": "industrial_robot"
  }'

# Explain component
curl -X POST http://localhost:8000/api/explain-component \
  -H "Content-Type: application/json" \
  -d '{
    "component_name": "speed reducer",
    "robot_type": "industrial_robot"
  }'

# Generate maintenance schedule
curl -X POST http://localhost:8000/api/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial_robot",
    "usage_hours": 8,
    "environment": "industrial"
  }'

# Answer question
curl -X POST http://localhost:8000/api/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How often should I lubricate the joints?",
    "robot_type": "industrial_robot"
  }'

# NLP Analysis
curl -X POST http://localhost:8000/api/nlp/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The robot needs preventive maintenance"
  }'

# ML Prediction
curl -X POST http://localhost:8000/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "industrial_robot",
    "operating_hours": 5000.0,
    "error_count": 3,
    "temperature": 45.0,
    "vibration_level": 0.8,
    "last_maintenance_hours": 200.0
  }'
```

## 📚 Project Structure

```
robot_maintenance_teaching_ai/
├── __init__.py
├── README.md
├── main.py
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── maintenance_config.py      # System configuration
├── core/
│   ├── __init__.py
│   ├── maintenance_tutor.py       # Main tutor with OpenRouter
│   ├── nlp_processor.py          # NLP processor (spaCy + Transformers)
│   └── ml_predictor.py            # ML predictor (scikit-learn)
├── api/
│   ├── __init__.py
│   └── maintenance_api.py         # FastAPI endpoints
├── examples/
│   ├── basic_usage.py             # Basic examples
│   └── nlp_ml_example.py          # NLP/ML examples
├── ml_models/
│   └── saved_models/              # Saved ML models
├── nlp_utils/
│   └── (additional NLP utilities)
└── data/
    └── conversations/              # Conversation history
```

## ⚙️ Configuration

### Basic Configuration

```python
from robot_maintenance_teaching_ai import MaintenanceConfig, OpenRouterConfig, MLConfig, NLPConfig

# Configure OpenRouter
openrouter_config = OpenRouterConfig(
    api_key="your-api-key",
    default_model="openai/gpt-4-turbo",
    temperature=0.7,
    max_tokens=3000
)

# Configure ML
ml_config = MLConfig(
    model_type="ensemble",
    prediction_threshold=0.7,
    use_pretrained=True
)

# Configure NLP
nlp_config = NLPConfig(
    language="es",
    model_name="es_core_news_md",
    use_transformer=True,
    transformer_model="dccuchile/bert-base-spanish-wwm-uncased"
)

# Main configuration
config = MaintenanceConfig(
    openrouter=openrouter_config,
    ml=ml_config,
    nlp=nlp_config,
    robot_types=["industrial_robot", "service_robot"],
    adaptive_learning=True
)
```

## 🤖 Supported Robot Types

- **industrial_robot**: Industrial robots (robotic arms, welding robots, etc.)
- **service_robot**: Service robots (cleaning, customer service, etc.)
- **collaborative_robot**: Collaborative robots (cobots)
- **mobile_robot**: Mobile robots (AGV, exploration robots, etc.)
- **medical_robot**: Medical robots (surgical, rehabilitation, etc.)
- **agricultural_robot**: Agricultural robots (seeding, harvesting, etc.)

## 🔧 Maintenance Types

- **preventive**: Preventive maintenance
- **corrective**: Corrective maintenance
- **predictive**: Predictive maintenance
- **emergency**: Emergency maintenance
- **scheduled**: Scheduled maintenance
- **condition_based**: Condition-based maintenance

## 📊 Difficulty Levels

- **beginner**: For beginners, basic explanations
- **intermediate**: Intermediate level, standard procedures
- **advanced**: Advanced level, complex procedures
- **expert**: Expert level, specialized procedures

## 🧠 Technologies Used

### NLP (Natural Language Processing)
- **spaCy**: Text processing and entity extraction
- **Transformers (Hugging Face)**: BERT models for advanced analysis
- **NLTK**: Additional NLP tools
- **Gensim**: Semantic analysis and similarity

### Machine Learning
- **scikit-learn**: Classification and regression models
- **Random Forest**: For failure prediction
- **Gradient Boosting**: Advanced ensemble models
- **NumPy/Pandas**: Data processing

### AI and APIs
- **OpenRouter**: Access to advanced AI models (GPT-4, Claude, etc.)
- **FastAPI**: Modern and fast web framework
- **httpx**: Asynchronous HTTP client

## 📖 Examples

### Example 1: Teaching Maintenance

```python
from robot_maintenance_teaching_ai import RobotMaintenanceTutor

tutor = RobotMaintenanceTutor()

result = await tutor.teach_maintenance_procedure(
    robot_type="industrial_robot",
    maintenance_type="preventive",
    difficulty="intermediate"
)

print(result["content"])
```

### Example 2: Using NLP

```python
from robot_maintenance_teaching_ai.core.nlp_processor import MaintenanceNLPProcessor

nlp = MaintenanceNLPProcessor()

text = "The robot needs gear inspection and lubrication"
analysis = nlp.process_maintenance_query(text)

print("Entities:", analysis["entities"])
print("Keywords:", analysis["keywords"])
print("Sentiment:", analysis["sentiment"])
```

### Example 3: ML Prediction

```python
from robot_maintenance_teaching_ai.core.ml_predictor import MaintenancePredictor

predictor = MaintenancePredictor()

prediction = predictor.predict_maintenance_need(
    robot_type="industrial_robot",
    operating_hours=5000.0,
    error_count=3,
    temperature=45.0,
    vibration_level=0.8,
    last_maintenance_hours=200.0
)

print(f"Needs maintenance?: {prediction['needs_maintenance']}")
print(f"Confidence: {prediction['confidence']}")
print(f"Recommendation: {prediction['recommendation']}")
```

## 🔍 API Endpoints

### GET `/`
Basic API information

### POST `/api/teach`
Teach maintenance procedure

### POST `/api/diagnose`
Diagnose robot problem

### POST `/api/explain-component`
Explain robot component

### POST `/api/schedule`
Generate maintenance schedule

### POST `/api/answer`
Answer maintenance question

### POST `/api/nlp/analyze`
Analyze text with NLP

### POST `/api/ml/predict`
Predict maintenance need

### POST `/api/ml/train`
Train machine learning model with synthetic or real data

### GET `/api/conversation/history`
Get conversation history

### GET `/api/health`
System health check with detailed component information

## 🛠️ Development

### Run examples

```bash
# Basic example
python examples/basic_usage.py

# NLP/ML example
python examples/nlp_ml_example.py
```

### Run tests

```bash
# (Add tests in the future)
pytest tests/
```

## 📝 Notes

- spaCy model is downloaded automatically the first time
- Transformers models are downloaded automatically
- ML models can be trained with custom data using the `/api/ml/train` endpoint
- GPU is recommended for better performance with Transformers
- Cache system is enabled by default (configurable in `MaintenanceConfig`)
- Conversation history is automatically saved in `data/conversations/`
- System includes automatic retry logic with exponential backoff for transient errors
- It is recommended to use the async context manager (`async with`) for automatic resource management

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
- [spaCy Documentation](https://spacy.io/usage)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
