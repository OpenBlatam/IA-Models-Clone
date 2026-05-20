# LinkedIn Posts System - Quick Start Guide

## 🚀 Get Started in 5 Minutes

This quick start guide will help you get the LinkedIn Posts system up and running with LangChain integration.

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- FastAPI (for API endpoints)
- Basic understanding of async/await

## ⚡ Quick Installation

### 1. Install Dependencies

```bash
pip install fastapi langchain openai pydantic loguru uvicorn
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LINKEDIN_POSTS_LOG_LEVEL="INFO"
```

### 3. Basic Usage Example

```python
import asyncio
from linkedin_posts.infrastructure.langchain_integration import LinkedInPostGenerator

async def generate_simple_post():
    # Initialize the generator
    generator = LinkedInPostGenerator("your-api-key", "gpt-4")
    
    # Generate a post
    post_data = await generator.generate_post(
        topic="Digital Transformation",
        key_points=[
            "Companies must adapt to digital technologies",
            "Employee training is crucial for success",
            "Data-driven decision making is the future"
        ],
        target_audience="Business leaders and executives",
        industry="technology",
        tone="professional"
    )
    
    print(f"Generated Post: {post_data['title']}")
    print(f"Content: {post_data['content']}")
    print(f"Hashtags: {post_data['hashtags']}")
    print(f"Estimated Engagement: {post_data['estimated_engagement']}%")

# Run the example
asyncio.run(generate_simple_post())
```

## 🎯 Common Use Cases

### 1. Generate Thought Leadership Post

```python
async def generate_thought_leadership():
    generator = LinkedInPostGenerator("your-api-key", "gpt-4")
    
    post = await generator.generate_thought_leadership_post(
        industry_trend="AI in Healthcare",
        analysis="AI is revolutionizing patient care and diagnosis",
        prediction="By 2025, 80% of healthcare organizations will use AI"
    )
    
    return post
```

### 2. Create Storytelling Post

```python
async def generate_storytelling():
    generator = LinkedInPostGenerator("your-api-key", "gpt-4")
    
    post = await generator.generate_storytelling_post(
        personal_experience="Failed startup taught me resilience",
        lesson_learned="Failure is the best teacher",
        industry_application="Apply lessons to current business challenges"
    )
    
    return post
```

### 3. Generate Industry-Specific Post

```python
async def generate_industry_post():
    generator = LinkedInPostGenerator("your-api-key", "gpt-4")
    
    post = await generator.generate_industry_specific_post(
        topic="Sustainable Business Practices",
        industry="finance",
        company_size="enterprise",
        target_role="professionals"
    )
    
    return post
```

## 🔧 API Quick Start

### 1. Start the API Server

```python
from fastapi import FastAPI
from linkedin_posts.presentation.api import linkedin_post_router

app = FastAPI(title="LinkedIn Posts API")
app.include_router(linkedin_post_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Generate Post via API

```bash
curl -X POST "http://localhost:8000/linkedin-posts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Future of Remote Work",
    "key_points": [
      "Hybrid work models are here to stay",
      "Technology enables seamless collaboration",
      "Work-life balance is more important than ever"
    ],
    "target_audience": "HR professionals and managers",
    "industry": "human resources",
    "tone": "professional"
  }'
```

### 3. Optimize Existing Post

```bash
curl -X POST "http://localhost:8000/linkedin-posts/{post_id}/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "optimization_type": "comprehensive"
  }'
```

## 📊 Content Templates

### Professional Post Template

```python
async def professional_post():
    generator = LinkedInPostGenerator("your-api-key", "gpt-4")
    
    return await generator.generate_post(
        topic="Your Topic",
        key_points=["Point 1", "Point 2", "Point 3"],
        target_audience="Your target audience",
        industry="Your industry",
        tone="professional",
        post_type="text",
        keywords=["keyword1", "keyword2"],
        additional_context="Any additional context"
    )
```

### Inspirational Post Template

```python
async def inspirational_post():
    generator = LinkedInPostGenerator("your-api-key", "gpt-4")
    
    return await generator.generate_post(
        topic="Motivational Topic",
        key_points=["Inspirational point 1", "Inspirational point 2"],
        target_audience="Professionals seeking motivation",
        industry="personal development",
        tone="inspirational",
        post_type="text"
    )
```

## 🧪 A/B Testing Quick Start

### 1. Create A/B Test Variants

```python
async def create_ab_test():
    generator = LinkedInPostGenerator("your-api-key", "gpt-4")
    
    variants = await generator.generate_multiple_variants(
        topic="Your Topic",
        key_points=["Point 1", "Point 2"],
        target_audience="Your audience",
        industry="Your industry",
        num_variants=3,
        tone="professional"
    )
    
    return variants
```

### 2. Test Different Tones

```python
async def test_tone_variations():
    generator = LinkedInPostGenerator("your-api-key", "gpt-4")
    
    tones = ["professional", "casual", "inspirational", "educational"]
    variants = []
    
    for tone in tones:
        variant = await generator.generate_post(
            topic="Same Topic",
            key_points=["Same points"],
            target_audience="Same audience",
            industry="Same industry",
            tone=tone
        )
        variants.append(variant)
    
    return variants
```

## 📈 Analytics Quick Start

### 1. Analyze Post Engagement

```python
async def analyze_engagement():
    from linkedin_posts.infrastructure.langchain_integration import EngagementAnalyzer
    
    analyzer = EngagementAnalyzer(llm=None)  # Initialize with your LLM
    
    analysis = await analyzer.comprehensive_analysis(
        content="Your post content here",
        target_audience="Your target audience",
        industry="Your industry"
    )
    
    print(f"Overall Score: {analysis['composite_score']}")
    print(f"Recommendations: {analysis['recommendations']}")
    
    return analysis
```

### 2. Optimize Content

```python
async def optimize_content():
    from linkedin_posts.infrastructure.langchain_integration import ContentOptimizer
    
    optimizer = ContentOptimizer(llm=None)  # Initialize with your LLM
    
    optimized = await optimizer.optimize_comprehensive(
        content="Your content to optimize",
        target_audience="Your audience",
        keywords=["keyword1", "keyword2"]
    )
    
    print(f"Original: {optimized['original']}")
    print(f"Optimized: {optimized['final_optimized']}")
    
    return optimized
```

## 🔧 Configuration Quick Start

### 1. Basic Configuration

```python
from linkedin_posts.shared.config.settings import LinkedInPostSettings

settings = LinkedInPostSettings(
    openai_api_key="your-api-key",
    langchain_model="gpt-4",
    enable_auto_optimization=True,
    enable_engagement_prediction=True
)
```

### 2. Custom Configuration

```python
settings = LinkedInPostSettings(
    openai_api_key="your-api-key",
    langchain_model="gpt-4",
    langchain_temperature=0.8,
    max_content_length=2500,
    max_hashtags=20,
    enable_auto_optimization=True,
    optimization_timeout=45,
    enable_ab_testing=True,
    max_ab_test_variants=4
)
```

## 🚀 Production Deployment

### 1. Environment Setup

```bash
# Production environment variables
export OPENAI_API_KEY="your-production-api-key"
export LINKEDIN_POSTS_LOG_LEVEL="WARNING"
export LINKEDIN_POSTS_DATABASE_URL="postgresql://user:pass@localhost/db"
export LINKEDIN_POSTS_ENABLE_CACHING=true
export LINKEDIN_POSTS_RATE_LIMIT_REQUESTS=1000
```

### 2. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Health Check

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }
```

## 📚 Next Steps

### 1. Explore Advanced Features
- [Content Templates](./docs/templates.md)
- [A/B Testing Guide](./docs/ab-testing.md)
- [Analytics Dashboard](./docs/analytics.md)

### 2. Integration Examples
- [Slack Integration](./docs/integrations/slack.md)
- [Zapier Integration](./docs/integrations/zapier.md)
- [Custom Webhooks](./docs/integrations/webhooks.md)

### 3. Best Practices
- [Content Creation Best Practices](./docs/best-practices.md)
- [Performance Optimization](./docs/performance.md)
- [Security Guidelines](./docs/security.md)

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**
   ```bash
   # Check your API key
   echo $OPENAI_API_KEY
   ```

2. **Rate Limiting**
   ```python
   # Increase timeout
   settings.optimization_timeout = 60
   ```

3. **Content Generation Fails**
   ```python
   # Check content length
   if len(content) > settings.max_content_length:
       content = content[:settings.max_content_length]
   ```

### Getting Help

- 📖 [Full Documentation](./docs/)
- 🐛 [Issue Tracker](https://github.com/your-repo/issues)
- 💬 [Community Forum](https://community.example.com)
- 📧 [Support Email](mailto:support@example.com)

## 🎉 Congratulations!

You've successfully set up the LinkedIn Posts system! Start generating amazing content and watch your engagement grow. 🚀

---

**Need help?** Check out our [full documentation](./docs/) or [contact support](mailto:support@example.com). 