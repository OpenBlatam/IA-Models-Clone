# AI & Deep Learning Features

This document describes the AI and deep learning capabilities integrated into the Lovable Community platform.

## Overview

The platform now includes comprehensive AI features powered by:
- **PyTorch** for deep learning operations
- **Transformers** for pre-trained language models
- **Sentence Transformers** for semantic embeddings
- **Diffusers** (ready for future image generation)

## Features

### 1. Semantic Search with Embeddings

**Service**: `EmbeddingService`

- Generates embeddings for chat content using sentence transformers
- Enables semantic similarity search (find chats by meaning, not just keywords)
- Uses model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Automatically creates embeddings when chats are published

**Endpoints**:
- `POST /lovable/ai/embeddings/{chat_id}` - Create/update embedding
- `GET /lovable/ai/search/semantic?query=...` - Semantic search

**Example**:
```python
# Find chats semantically similar to "machine learning tutorials"
GET /lovable/ai/search/semantic?query=machine%20learning%20tutorials&limit=10
```

### 2. Sentiment Analysis

**Service**: `SentimentService`

- Analyzes sentiment of chat content (positive/negative/neutral)
- Uses model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Stores sentiment labels and confidence scores
- Automatically analyzes sentiment when chats are published

**Endpoints**:
- `POST /lovable/ai/sentiment/{chat_id}` - Analyze chat sentiment
- `POST /lovable/ai/sentiment/analyze?text=...` - Analyze arbitrary text

**Example**:
```python
# Analyze sentiment of a chat
POST /lovable/ai/sentiment/chat-123

# Response:
{
  "chat_id": "chat-123",
  "sentiment": {
    "label": "positive",
    "score": 0.95
  }
}
```

### 3. Content Moderation

**Service**: `ModerationService`

- Detects toxic, harmful, or inappropriate content
- Uses model: `unitary/toxic-bert`
- Configurable toxicity threshold (default: 0.7)
- Flags content for review
- Automatically moderates content when chats are published

**Endpoints**:
- `POST /lovable/ai/moderate/{chat_id}` - Moderate a chat
- `POST /lovable/ai/moderate/check?text=...` - Check text before publishing

**Example**:
```python
# Check if content should be blocked
POST /lovable/ai/moderate/check?text=Your text here

# Response:
{
  "is_toxic": false,
  "toxicity_score": 0.15,
  "should_block": false,
  "flags": []
}
```

### 4. Text Generation

**Service**: `TextGenerationService`

- Generates text using language models (GPT-2 by default)
- Supports text completion, enhancement, and tag generation
- Configurable temperature and sampling parameters
- Mixed precision support for GPU acceleration

**Endpoints**:
- `POST /lovable/ai/generate?prompt=...` - Generate text
- `POST /lovable/ai/enhance/description?description=...` - Enhance description
- `POST /lovable/ai/generate/tags?title=...` - Generate tags

**Example**:
```python
# Generate tags for a chat
POST /lovable/ai/generate/tags?title=Python Tutorial&description=Learn Python basics

# Response:
{
  "title": "Python Tutorial",
  "description": "Learn Python basics",
  "suggested_tags": ["python", "tutorial", "programming", "coding", "beginner"]
}
```

### 5. AI-Powered Recommendations

**Service**: `RecommendationService`

- Recommends similar chats based on embeddings
- Personalized recommendations based on user preferences
- Considers sentiment preferences
- Tag-based recommendations

**Endpoints**:
- `GET /lovable/ai/recommend/{chat_id}` - Similar chats
- `GET /lovable/ai/recommend/user/{user_id}` - Personalized recommendations
- `GET /lovable/ai/recommend/tags?tags=...` - Tag-based recommendations

**Example**:
```python
# Get personalized recommendations
GET /lovable/ai/recommend/user/user-123?limit=10&use_sentiment=true

# Response:
{
  "user_id": "user-123",
  "recommendations": [
    {
      "chat_id": "chat-456",
      "similarity": 0.87,
      "title": "Advanced Python",
      "score": 8.5,
      "sentiment_match": true
    }
  ]
}
```

## Database Models

### ChatEmbedding
Stores embeddings for semantic search:
- `chat_id`: Reference to chat
- `embedding`: Vector representation (JSON)
- `embedding_model`: Model used to generate embedding

### ChatAIMetadata
Stores AI analysis results:
- `sentiment_label`: positive/negative/neutral
- `sentiment_score`: Confidence score
- `toxicity_score`: Toxicity detection score
- `is_toxic`: Boolean flag
- `moderation_flags`: List of moderation issues

## Configuration

All AI features can be configured via environment variables:

```env
# Enable/disable AI features
AI_ENABLED=True
USE_GPU=True
DEVICE=cuda

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
BATCH_SIZE_EMBEDDINGS=32

# Sentiment Analysis
SENTIMENT_ENABLED=True
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# Content Moderation
MODERATION_ENABLED=True
MODERATION_MODEL=unitary/toxic-bert
MODERATION_THRESHOLD=0.7

# Text Generation
TEXT_GENERATION_ENABLED=True
TEXT_GENERATION_MODEL=gpt2
MAX_GENERATION_LENGTH=200

# Mixed Precision (for GPU)
USE_MIXED_PRECISION=True

# Model Cache
MODEL_CACHE_DIR=./models_cache
```

## Automatic Processing

When a chat is published, the system automatically:

1. **Moderates** the content (if moderation enabled)
2. **Analyzes sentiment** (if sentiment analysis enabled)
3. **Generates embedding** (for semantic search)

This processing happens asynchronously and doesn't block the publish operation. If AI processing fails, the chat is still published, but a warning is logged.

## Performance Considerations

- **GPU Support**: Automatically uses GPU if available
- **Mixed Precision**: Uses FP16 on GPU for faster inference
- **Batch Processing**: Embeddings can be generated in batches
- **Lazy Loading**: Models are loaded only when needed
- **Caching**: Models are cached to avoid reloading

## Best Practices

1. **Enable GPU** for production deployments with CUDA support
2. **Adjust batch sizes** based on available memory
3. **Monitor toxicity scores** and adjust threshold as needed
4. **Use semantic search** instead of keyword search for better results
5. **Leverage recommendations** to improve user engagement

## Future Enhancements

- [ ] Diffusion models for image generation
- [ ] Fine-tuning on community data
- [ ] Multi-modal embeddings (text + images)
- [ ] Real-time recommendation updates
- [ ] Advanced content quality scoring
- [ ] Custom model training pipeline

## Dependencies

All AI features require:
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `sentence-transformers>=2.2.0`
- `diffusers>=0.24.0` (for future image generation)
- `gradio>=4.0.0` (for interactive demos)

See `requirements.txt` for complete list.















