# AI Educational Tutor with Open Router

> Part of the [Blatam Academy Integrated Platform](../README.md)

Intelligent educational tutor system using Open Router to provide personalized educational assistance to students.

## 🎯 Features

### Key Functionalities
- **Personalized Tutoring** — Responses adapted to student's level and learning style
- **Multiple Subjects** — Support for Math, Science, History, Literature, Programming, and more
- **Learning Analysis** — Tracking student progress and adapting content
- **Exercise Generation** — Creates personalized practice exercises
- **Quiz Generation** — Creates complete quizzes with different question types
- **Conversation History** — Maintains context of previous interactions
- **REST API** — Easy-to-use endpoints for integration

### Advanced Features ⚡
- **Intelligent Cache System** — Reduces API calls and improves response times
- **Rate Limiting** — Speed control to avoid exceeding API limits
- **Metrics & Analytics** — Detailed tracking of usage, performance, and costs
- **Multiple Models** — Support for different Open Router models
- **Retry Logic** — Automatic retries in case of errors
- **Progress Analysis** — Identification of student strengths and weaknesses
- **Reporting System** — Full reporting in JSON, Markdown, and HTML
- **Gamification** — Badges, points, levels, and leaderboards system
- **Learning Streaks** — Tracking of consecutive study days
- **Data Export** — Export reports and statistics in multiple formats
- **Automatic Evaluation** — Intelligent evaluation of answers and quizzes
- **Recommendation Engine** — Personalized learning recommendations
- **Notification System** — Intelligent notifications for engagement
- **Learning Paths** — Structured paths by subject and level
- **Automatic Feedback** — Instant and personalized feedback
- **Analytics Dashboard** — Real-time visualizations and statistics
- **Database System** — Full data persistence
- **Auth & Authorization** — Users, roles, and permissions system
- **Automatic Backups** — Data backup system
- **Webhooks System** — Event notifications for integrations
- **LMS Integration** — Support for Moodle, Canvas, Blackboard, and more
- **Full Testing** — Test suite with pytest
- **Docker Ready** — Docker configuration ready for production
- **Python SDK** — Full SDK for easy integration
- **API Versioning** — Support for multiple API versions
- **Advanced Validation** — Robust input validation system
- **Utility Scripts** — Automated setup and backup

## 📋 Requirements

- Python 3.8+
- Open Router API Key (`OPENROUTER_API_KEY`)

## 🚀 Installation

### Quick Setup

```bash
# Run setup script
python scripts/setup.py

# Update .env with your API key
# Then run
python main.py
```

### Option 1: Local Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: Docker

1. Build and run with Docker Compose:

```bash
docker-compose up -d
```

### Option 3: Manual Docker

1. Build image:

```bash
docker build -t ai-tutor-educational .
```

2. Run container:

```bash
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your-api-key ai-tutor-educational
```

2. Configure environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
OPENROUTER_API_KEY=your-api-key-here
```

## 💻 Basic Usage

### Usage as Python Module

```python
import asyncio
from ai_tutor_educacional_openrouter import AITutor, TutorConfig

async def main():
    config = TutorConfig()
    tutor = AITutor(config)
    
    # Ask a question
    response = await tutor.ask_question(
        question="What is photosynthesis?",
        subject="science",
        difficulty="intermediate"
    )
    
    print(response["answer"])
    
    # Explain a concept
    explanation = await tutor.explain_concept(
        concept="derivatives",
        subject="math",
        difficulty="advanced"
    )
    
    print(explanation["answer"])
    
    # Generate exercises
    exercises = await tutor.generate_exercise(
        topic="quadratic equations",
        subject="math",
        difficulty="intermediate",
        num_exercises=5
    )
    
    print(exercises["answer"])
    
    await tutor.close()

asyncio.run(main())
```

### Usage with REST API

1. Start server:

```python
from ai_tutor_educacional_openrouter.api.tutor_api import create_tutor_app
import uvicorn

app = create_tutor_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

2. Use endpoints:

```bash
# Ask a question
curl -X POST http://localhost:8000/api/tutor/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is photosynthesis?",
    "subject": "science",
    "difficulty": "intermediate"
  }'

# Explain a concept
curl -X POST http://localhost:8000/api/tutor/explain \
  -H "Content-Type: application/json" \
  -d '{
    "concept": "derivatives",
    "subject": "math",
    "difficulty": "advanced"
  }'

# Generate exercises
curl -X POST http://localhost:8000/api/tutor/exercises \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "quadratic equations",
    "subject": "math",
    "difficulty": "intermediate",
    "num_exercises": 5
  }'

# Generate quiz
curl -X POST http://localhost:8000/api/tutor/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "algebra",
    "subject": "math",
    "difficulty": "intermediate",
    "num_questions": 10,
    "question_types": ["multiple_choice", "short_answer"]
  }'

# Get metrics
curl http://localhost:8000/api/tutor/metrics

# Clear cache
curl -X DELETE http://localhost:8000/api/tutor/cache
```

## 📚 Project Structure

```
ai_tutor_educacional_openrouter/
├── __init__.py
├── main.py                       # Server entry point
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── tutor_config.py          # System configuration
├── core/
│   ├── __init__.py
│   ├── tutor.py                 # Main tutor class (improved)
│   ├── conversation_manager.py  # Conversation management
│   ├── learning_analyzer.py     # Learning analysis
│   ├── cache_manager.py         # Cache system (NEW)
│   ├── rate_limiter.py          # Rate limiting (NEW)
│   ├── metrics_collector.py     # Metrics & analytics (NEW)
│   └── quiz_generator.py        # Quiz generator (NEW)
├── api/
│   ├── __init__.py
│   └── tutor_api.py             # FastAPI Endpoints (improved)
├── utils/
│   ├── __init__.py
│   └── helpers.py               # Helper functions
└── examples/
    ├── __init__.py
    ├── basic_usage.py           # Basic usage examples
    └── api_usage.py             # REST API examples
```

## ⚙️ Configuration

You can customize the configuration by creating a `TutorConfig` instance:

```python
from ai_tutor_educacional_openrouter import TutorConfig, OpenRouterConfig

openrouter_config = OpenRouterConfig(
    api_key="your-api-key",
    default_model="openai/gpt-4",
    temperature=0.7,
    max_tokens=2000
)

config = TutorConfig(
    openrouter=openrouter_config,
    subjects=["math", "programming"],
    adaptive_learning=True,
    provide_exercises=True
)
```

## 🔧 Supported Subjects

- Math
- Science
- History
- Literature
- Physics
- Chemistry
- Biology
- Programming

## 📊 Difficulty Levels

- **Basic**: Fundamental concepts and simple explanations
- **Intermediate**: More complex concepts with examples
- **Advanced**: Complex topics with deep analysis

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of Blatam Academy.

---

[← Back to Main README](../README.md)
