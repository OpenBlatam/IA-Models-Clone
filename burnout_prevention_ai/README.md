# Burnout Prevention AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

AI system for burnout prevention and management in the workplace, helping identify early signs and provide customized coping strategies.

## 🎯 Features

- **Burnout Assessment**: Comprehensive analysis of burnout risk based on multiple factors
- **Wellness Check**: Evaluation of general wellness status and recommendations
- **Coping Strategies**: Personalized recommendations for managing stress
- **Conversational Chat**: Empathetic AI assistant for burnout and wellness conversations
- **Progress Tracking**: Progress analysis over time with personalized insights
- **Trend Analysis**: Pattern identification and predictions based on history
- **Educational Resources**: Personalized resource library (articles, videos, podcasts, books)
- **Personalized Plans**: Generation of structured plans adapted to each user
- **OpenRouter Integration**: Uses advanced AI models via OpenRouter

## 🚀 Installation

```bash
# Basic installation (production)
pip install -r requirements.txt

# Installation with development tools
pip install -r requirements-dev.txt

# Minimal installation (core only)
pip install -r requirements-minimal.txt

# Configure environment variables
export OPENROUTER_API_KEY="your-api-key"
export BURNOUT_AI_PORT=8025
```

## ⚙️ Configuration

Create `.env` file:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
BURNOUT_AI_HOST=0.0.0.0
BURNOUT_AI_PORT=8025
DEBUG=False
```

## 🏃 Usage

### Start Server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8025 --reload
```

### Endpoints

#### Assess Burnout
```bash
POST /api/v1/assess
```

Body:
```json
{
  "work_hours_per_week": 50,
  "stress_level": 8,
  "sleep_hours_per_night": 5.5,
  "work_satisfaction": 4,
  "physical_symptoms": ["fatigue", "headaches"],
  "emotional_symptoms": ["anxiety", "irritability"],
  "work_environment": "High pressure, tight deadlines",
  "additional_context": "Remote work, multiple projects"
}
```

#### Wellness Check
```bash
POST /api/v1/wellness-check
```

Body:
```json
{
  "current_mood": "anxious and exhausted",
  "energy_level": 3,
  "recent_challenges": "Important project with tight deadline",
  "support_system": "Family and some colleagues"
}
```

#### Coping Strategies
```bash
POST /api/v1/coping-strategies
```

Body:
```json
{
  "stressor_type": "Work overload",
  "current_coping_methods": ["working more hours"],
  "available_time": "30 minutes daily",
  "preferences": ["exercise", "meditation"]
}
```

#### Conversational Chat
```bash
POST /api/v1/chat
```

Body:
```json
{
  "message": "I feel very exhausted lately, what can I do?",
  "conversation_history": []
}
```

#### Progress Tracking
```bash
POST /api/v1/progress
```

Body:
```json
{
  "user_id": "user123",
  "assessment_history": [
    {"date": "2024-01-01", "burnout_score": 75},
    {"date": "2024-01-15", "burnout_score": 65}
  ],
  "goals": ["Reduce work hours", "Improve sleep"],
  "current_status": {"stress_level": 6, "energy_level": 5}
}
```

#### Trend Analysis
```bash
POST /api/v1/trends
```

Body:
```json
{
  "assessments": [
    {"date": "2024-01-01", "burnout_score": 80, "stress_level": 9},
    {"date": "2024-01-08", "burnout_score": 75, "stress_level": 8},
    {"date": "2024-01-15", "burnout_score": 70, "stress_level": 7}
  ],
  "time_period_days": 30
}
```

#### Educational Resources
```bash
POST /api/v1/resources
```

Body:
```json
{
  "topic": "Work stress management",
  "level": "intermediate",
  "format_preference": "article"
}
```

#### Personalized Plan
```bash
POST /api/v1/personalized-plan
```

Body:
```json
{
  "current_situation": {
    "burnout_score": 70,
    "main_stressors": ["Work overload", "Lack of boundaries"]
  },
  "goals": [
    "Reduce burnout score to 50",
    "Establish healthy boundaries",
    "Improve work-life balance"
  ],
  "constraints": {
    "available_time": "1 hour daily",
    "budget": "limited"
  },
  "preferences": {
    "activities": ["exercise", "meditation", "reading"]
  }
}
```

#### Health Check
```bash
GET /api/v1/health
```

## 📊 Example Responses

### Burnout Assessment
```json
{
  "burnout_risk_level": "high",
  "burnout_score": 75.5,
  "risk_factors": [
    "Excessive work hours",
    "Lack of sleep",
    "High stress level"
  ],
  "recommendations": [
    "Establish clear work schedule boundaries",
    "Prioritize 7-8 hours of sleep",
    "Implement stress management techniques"
  ],
  "immediate_actions": [
    "Take a 15-minute break now",
    "Schedule recovery time this week",
    "Talk to your supervisor about workload"
  ],
  "long_term_strategies": [
    "Review and adjust work expectations",
    "Develop self-care routine",
    "Establish healthy boundaries"
  ],
  "assessment_date": "2024-01-15T10:30:00"
}
```

### Progress Tracking
```json
{
  "progress_score": 65.0,
  "trend": "improving",
  "milestones_achieved": [
    "10 point reduction in burnout score",
    "Establishment of schedule boundaries",
    "Improvement in sleep hours"
  ],
  "next_steps": [
    "Maintain established boundaries",
    "Continue with self-care routine",
    "Evaluate weekly workload"
  ],
  "insights": "You have shown consistent improvement over the last 15 days. The reduction in your burnout score indicates that implemented strategies are working. Continue with the current approach and consider adding more time for recovery activities.",
  "progress_date": "2024-01-15T10:30:00"
}
```

### Trend Analysis
```json
{
  "overall_trend": "improving",
  "key_metrics": {
    "burnout_score_avg": 72.5,
    "stress_level_trend": "decreasing",
    "improvement_rate": 0.67
  },
  "patterns": [
    "Consistent improvement on weekends",
    "Increased stress on Mondays",
    "Positive correlation between sleep and wellness"
  ],
  "predictions": {
    "next_week": {
      "expected_score": 68,
      "confidence": "medium"
    },
    "next_month": {
      "expected_score": 55,
      "confidence": "high"
    }
  },
  "recommendations": [
    "Maintain current self-care routine",
    "Focus on improving sleep quality",
    "Better plan Mondays to reduce stress",
    "Continue weekly monitoring"
  ],
  "analysis_date": "2024-01-15T10:30:00"
}
```

### Educational Resources
```json
{
  "resources": [
    {
      "title": "Understanding and Preventing Burnout",
      "type": "article",
      "description": "Complete guide on burnout and prevention strategies",
      "url": "https://example.com/burnout-guide",
      "duration": "15 min read"
    },
    {
      "title": "Stress Management Techniques",
      "type": "video",
      "description": "Practical stress management techniques",
      "url": "https://example.com/stress-video",
      "duration": "20 min"
    }
  ],
  "learning_path": [
    "1. Understand what burnout is",
    "2. Identify your risk factors",
    "3. Learn stress management techniques",
    "4. Implement prevention strategies",
    "5. Monitor and adjust continuously"
  ],
  "key_concepts": [
    "Burnout vs stress",
    "Early symptoms",
    "Healthy boundaries",
    "Self-care",
    "Work-life balance"
  ],
  "action_items": [
    "Read article on prevention",
    "Practice a stress management technique",
    "Set a schedule limit this week"
  ]
}
```

### Personalized Plan
```json
{
  "plan_name": "Recovery and Prevention Plan - 8 Weeks",
  "duration_weeks": 8,
  "weekly_goals": [
    {
      "week": 1,
      "goal": "Establish basic schedule boundaries",
      "actions": [
        "Define fixed work schedule",
        "Communicate boundaries to team",
        "Implement notification blocking after hours"
      ],
      "focus_area": "Work boundaries"
    },
    {
      "week": 2,
      "goal": "Improve sleep quality",
      "actions": [
        "Establish sleep routine",
        "Create optimal sleep environment",
        "Limit screens before sleep"
      ],
      "focus_area": "Recovery"
    }
  ],
  "daily_actions": [
    "10-minute meditation",
    "15-minute break every 2 hours",
    "Light exercise (20 min walk)",
    "Daily reflection on wellness",
    "Complete disconnection after work hours"
  ],
  "milestones": [
    {
      "week": 2,
      "milestone": "Boundaries established and communicated"
    },
    {
      "week": 4,
      "milestone": "Sleep routine improved"
    },
    {
      "week": 6,
      "milestone": "Burnout score reduced by 15 points"
    },
    {
      "week": 8,
      "milestone": "Self-care system established"
    }
  ],
  "resources": [
    "Meditation App",
    "Book on healthy boundaries",
    "Sleep hygiene guide"
  ],
  "created_date": "2024-01-15T10:30:00"
}
```

## 🏗️ Architecture

```
burnout_prevention_ai/
├── main.py                 # Main application
├── config/                 # Configuration
│   └── app_config.py
├── infrastructure/         # Infrastructure
│   └── openrouter/         # OpenRouter Client
│       ├── api_client.py
│       └── openrouter_client.py
├── services/               # Business services
│   └── burnout_service.py
├── api/                    # API endpoints
│   └── routes/
│       └── burnout_routes.py
├── schemas.py              # Pydantic models
└── requirements.txt
```

## 🔧 Development

### Code Structure

- **main.py**: Entry point of the FastAPI application
- **config/**: Application configuration
- **infrastructure/**: External clients (OpenRouter)
- **services/**: Business logic
- **api/**: API Endpoints
- **schemas.py**: Pydantic data models

### Adding New Features

1. Add schema in `schemas.py`
2. Implement logic in `services/burnout_service.py`
3. Create endpoint in `api/routes/burnout_routes.py`

## 📝 Notes

- The service uses OpenRouter to access advanced AI models
- Responses are generated dynamically using Claude 3.5 Sonnet by default
- The system is designed to be empathetic, conversational, and non-judgmental
- All conversations are confidential
- The chat adapts its communication style to the user
- Personalized plans are generated considering real constraints and preferences
- Trend analysis identifies patterns for more accurate predictions
- Educational resources are personalized based on level and user preferences

## 🚀 Performance Improvements

The project uses optimized libraries for better performance:

- **orjson**: JSON 2-3x faster than standard library
- **uvloop**: Ultra-fast event loop (Linux/macOS)
- **structlog**: Structured logging with JSON
- **httpx**: Modern asynchronous HTTP client
- **tenacity**: Intelligent retries with exponential backoff

## 📦 Main Dependencies

- **FastAPI 0.115+**: Modern and fast web framework
- **Pydantic 2.9+**: Data validation with better performance
- **httpx 0.27+**: Asynchronous HTTP client
- **structlog**: Structured logging for production
- **prometheus-client**: Metrics for monitoring
- **slowapi**: Optional rate limiting

## 🔒 Security

- API keys must be stored securely
- Consider implementing authentication for production
- Validate all user inputs
- Implement rate limiting to prevent abuse

## 📚 Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Burnout Prevention Resources](https://www.who.int/news/item/28-05-2019-burn-out-an-occupational-phenomenon-international-classification-of-diseases)

## 🤝 Contribution

Contributions are welcome. Please:

1. Fork the project
2. Create a branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

Proprietary - Blatam Academy

---

[← Back to Main README](../README.md)
