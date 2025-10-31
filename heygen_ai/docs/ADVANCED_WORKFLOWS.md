# Advanced AI Workflows Documentation

## Overview

The HeyGen AI equivalent system now includes advanced AI workflows powered by LangChain and OpenRouter. These workflows provide sophisticated, multi-step content creation processes that go beyond simple video generation.

## 🚀 Available Workflows

### 1. Educational Series Workflow
Creates complete educational video series with research, planning, and content generation.

**Features:**
- Topic research and analysis
- Series structure planning
- Episode script generation
- Content optimization
- Multi-language support

**Use Cases:**
- Online courses
- Training programs
- Educational content
- Skill development series

### 2. Marketing Campaign Workflow
Generates comprehensive marketing campaigns with multiple video variants and audience analysis.

**Features:**
- Brand and product analysis
- Target audience research
- Message development
- Campaign script variants
- A/B testing preparation

**Use Cases:**
- Product launches
- Brand awareness campaigns
- Lead generation
- Customer acquisition

### 3. Product Demo Workflow
Creates product demonstration videos with feature analysis and benefit mapping.

**Features:**
- Product feature analysis
- Feature prioritization
- Benefit mapping
- Demo script generation
- Call-to-action optimization

**Use Cases:**
- Product launches
- Feature showcases
- Sales presentations
- Customer onboarding

### 4. News Summary Workflow
Generates news summary videos with fact-checking and multi-language support.

**Features:**
- News research and fact-checking
- Neutral summary generation
- Multi-language translation
- Source verification
- Journalistic standards

**Use Cases:**
- News organizations
- Content aggregators
- Educational news
- International news

## 🔧 API Endpoints

### Workflow Status
```http
GET /api/v1/workflows/available
```

**Response:**
```json
{
  "available_workflows": [
    "educational_series",
    "marketing_campaign", 
    "product_demo",
    "news_summary"
  ],
  "total_count": 4,
  "langchain_required": true
}
```

### Educational Series
```http
POST /api/v1/workflows/educational-series
```

**Request Body:**
```json
{
  "topic": "Introduction to Machine Learning",
  "series_length": 5,
  "target_audience": "university students",
  "difficulty_level": "intermediate",
  "language": "en"
}
```

**Response:**
```json
{
  "workflow_type": "educational_series",
  "status": "completed",
  "topic": "Introduction to Machine Learning",
  "series_length": 5,
  "episodes": [
    {
      "episode_number": 1,
      "title": "Episode 1: Introduction to Machine Learning",
      "script": "Welcome to our series on machine learning...",
      "duration": 5.0,
      "key_points": ["Key point 1", "Key point 2", "Key point 3"]
    }
  ],
  "series_metadata": {
    "series_title": "Complete Guide to Machine Learning",
    "total_episodes": 5,
    "total_duration": 25.0,
    "target_audience": "students and educators",
    "difficulty_level": "intermediate"
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Marketing Campaign
```http
POST /api/v1/workflows/marketing-campaign
```

**Request Body:**
```json
{
  "product_info": {
    "name": "AI Video Creator Pro",
    "description": "Professional AI-powered video creation platform",
    "features": [
      "Advanced AI avatars",
      "Multi-language support",
      "Real-time voice cloning"
    ],
    "target_users": "Content creators, marketers, educators"
  },
  "target_audience": "Professional content creators and marketing teams",
  "campaign_type": "product_launch",
  "budget_range": "high",
  "goals": ["brand_awareness", "lead_generation"]
}
```

**Response:**
```json
{
  "workflow_type": "marketing_campaign",
  "status": "completed",
  "product_info": {...},
  "target_audience": "Professional content creators...",
  "campaign_scripts": [
    {
      "variant": 1,
      "message_focus": "Problem-solving",
      "target_audience": "Problem-focused users",
      "call_to_action": "Solve your problems today",
      "script": "Are you struggling with video creation?..."
    }
  ],
  "brand_analysis": {
    "brand_positioning": "Premium quality solution",
    "unique_selling_propositions": ["Feature 1", "Feature 2"],
    "target_market": "Professional users"
  },
  "audience_analysis": {
    "demographics": {
      "age_range": "25-45",
      "education_level": "Bachelor's degree or higher"
    },
    "interests": ["Technology", "Innovation"],
    "pain_points": ["Time management", "Complexity"]
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Product Demo
```http
POST /api/v1/workflows/product-demo
```

**Request Body:**
```json
{
  "product_info": {
    "name": "SmartHome Hub",
    "description": "Central control system for smart home devices",
    "features": [
      "Voice control integration",
      "Mobile app control",
      "Automation scheduling"
    ],
    "target_users": "Homeowners and tech enthusiasts"
  },
  "demo_type": "feature_showcase",
  "target_users": "general",
  "focus_areas": ["ease_of_use", "automation"]
}
```

**Response:**
```json
{
  "workflow_type": "product_demo",
  "status": "completed",
  "product_info": {...},
  "demo_script": "Welcome to the SmartHome Hub demo...",
  "product_analysis": {
    "target_users": "Homeowners and tech enthusiasts",
    "use_cases": ["Use case 1", "Use case 2"],
    "complexity_level": "Intermediate"
  },
  "feature_priority": [
    {
      "priority": 1,
      "feature": "Voice control integration",
      "importance": "High"
    }
  ],
  "benefit_mapping": {
    "Voice control integration": [
      "Hands-free operation",
      "Accessibility improvement",
      "Convenience enhancement"
    ]
  },
  "cta_variations": [
    "Try it today and see the difference!",
    "Start your free trial now",
    "Get started in minutes"
  ],
  "created_at": "2024-01-01T00:00:00Z"
}
```

### News Summary
```http
POST /api/v1/workflows/news-summary
```

**Request Body:**
```json
{
  "news_topic": "Recent developments in renewable energy technology",
  "target_languages": ["en", "es", "fr"],
  "summary_length": "medium",
  "include_fact_checking": true,
  "neutral_tone": true
}
```

**Response:**
```json
{
  "workflow_type": "news_summary",
  "status": "completed",
  "news_topic": "Recent developments in renewable energy technology",
  "video_script": "Today we bring you the latest developments...",
  "summary": "Recent advances in renewable energy technology...",
  "translations": {
    "es": "Hoy les traemos los últimos desarrollos...",
    "fr": "Aujourd'hui, nous vous apportons les derniers développements..."
  },
  "news_research": {
    "facts": ["Fact 1", "Fact 2"],
    "context": "Background context...",
    "sources": ["Reliable source 1", "Reliable source 2"]
  },
  "fact_check_results": {
    "status": "verified",
    "confidence": "high"
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

## 🐍 Python Usage Examples

### Basic Workflow Usage
```python
import asyncio
from heygen_ai.core import HeyGenAI

async def create_educational_series():
    # Initialize with OpenRouter API key
    heygen = HeyGenAI(openrouter_api_key="your_openrouter_api_key")
    
    # Create educational series
    result = await heygen.create_educational_series(
        topic="Introduction to Machine Learning",
        series_length=5
    )
    
    print(f"Created {result['series_metadata']['total_episodes']} episodes")
    return result

# Run the workflow
asyncio.run(create_educational_series())
```

### Marketing Campaign Example
```python
async def create_marketing_campaign():
    heygen = HeyGenAI(openrouter_api_key="your_openrouter_api_key")
    
    product_info = {
        "name": "AI Video Creator Pro",
        "description": "Professional AI-powered video creation platform",
        "features": ["Advanced AI avatars", "Multi-language support"],
        "target_users": "Content creators, marketers, educators"
    }
    
    result = await heygen.create_marketing_campaign(
        product_info=product_info,
        target_audience="Professional content creators"
    )
    
    print(f"Created {len(result['campaign_scripts'])} campaign variants")
    return result
```

### Complete Pipeline Example
```python
async def complete_pipeline():
    heygen = HeyGenAI(openrouter_api_key="your_openrouter_api_key")
    
    # Step 1: Educational series
    series = await heygen.create_educational_series(
        topic="Data Science Fundamentals",
        series_length=3
    )
    
    # Step 2: Marketing campaign
    campaign = await heygen.create_marketing_campaign(
        product_info={"name": "Data Science Platform", "features": ["AI", "Analytics"]},
        target_audience="Data scientists and analysts"
    )
    
    # Step 3: Product demo
    demo = await heygen.create_product_demo(
        product_info={"name": "Data Science Platform", "features": ["AI", "Analytics"]}
    )
    
    # Step 4: News summary
    news = await heygen.create_news_summary(
        news_topic="Latest developments in data science and AI",
        target_languages=["en", "es"]
    )
    
    return {
        "educational_series": series,
        "marketing_campaign": campaign,
        "product_demo": demo,
        "news_summary": news
    }
```

## 🔍 Workflow Status and Health

### Check Available Workflows
```python
heygen = HeyGenAI(openrouter_api_key="your_openrouter_api_key")
workflows = heygen.get_available_workflows()
print(f"Available workflows: {workflows}")
```

### Health Check
```python
health_status = heygen.health_check()
langchain_status = heygen.get_langchain_status()

print("System Health:", health_status)
print("LangChain Status:", langchain_status)
```

## ⚙️ Configuration

### Environment Variables
```bash
# Required for advanced workflows
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional settings
LANGCHAIN_ENABLED=true
DEFAULT_LLM_MODEL=openai/gpt-4
TEMPERATURE=0.7
MAX_TOKENS=4000
```

### LangChain Models
The system supports multiple AI models via OpenRouter:
- `openai/gpt-4` - OpenAI's GPT-4
- `anthropic/claude-3-sonnet` - Anthropic's Claude-3
- `meta-llama/llama-2-70b-chat` - Meta's Llama-2
- `google/gemini-pro` - Google's Gemini Pro

## 🚀 Performance Optimization

### Workflow Optimization Tips
1. **Model Selection**: Choose appropriate models for each workflow type
2. **Batch Processing**: Use batch operations for multiple workflows
3. **Caching**: Enable LangChain caching for repeated operations
4. **Parallel Execution**: Run independent workflows in parallel

### Resource Management
- Monitor API usage and costs
- Implement rate limiting
- Use appropriate model sizes for tasks
- Cache frequently used results

## 🔒 Security and Best Practices

### API Key Management
- Store OpenRouter API keys securely
- Use environment variables
- Implement proper access controls
- Monitor API usage

### Content Safety
- Validate all input data
- Implement content filtering
- Use fact-checking for news workflows
- Maintain journalistic standards

## 📊 Monitoring and Analytics

### Workflow Metrics
- Success/failure rates
- Processing times
- Resource usage
- Cost tracking

### Health Monitoring
- Component health checks
- LangChain integration status
- Model availability
- API response times

## 🆘 Troubleshooting

### Common Issues

**1. LangChain Not Available**
```
Error: Advanced workflows not available. Please provide OpenRouter API key.
```
**Solution**: Ensure OpenRouter API key is set and valid.

**2. Model Unavailable**
```
Error: Model not available via OpenRouter
```
**Solution**: Check model availability and API key permissions.

**3. Workflow Timeout**
```
Error: Workflow execution timed out
```
**Solution**: Increase timeout limits or optimize workflow complexity.

### Debug Mode
Enable debug mode for detailed logging:
```bash
export HEYGEN_DEBUG=true
export HEYGEN_LOG_LEVEL=DEBUG
```

## 📚 Additional Resources

- [LangChain Documentation](https://langchain.com/)
- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [HeyGen AI API Documentation](./API_DOCUMENTATION.md)
- [Examples Directory](../examples/)

## 🔄 Updates and Changelog

### Recent Updates
- ✅ Added Advanced AI Workflows
- ✅ Integrated LangChain and OpenRouter
- ✅ Educational Series Workflow
- ✅ Marketing Campaign Workflow
- ✅ Product Demo Workflow
- ✅ News Summary Workflow
- ✅ Comprehensive API endpoints
- ✅ Health monitoring and status checks

---

**Powered by LangChain & OpenRouter** 🚀 