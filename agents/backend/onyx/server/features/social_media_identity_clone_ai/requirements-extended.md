# Extended Requirements - Additional Libraries Guide

## New Libraries Added

### 1. Social Media Scraping & APIs

**instaloader>=4.12.0**
- Instagram scraping (unofficial but reliable)
- Download posts, stories, profiles
- Usage:
```python
import instaloader
loader = instaloader.Instaloader()
profile = instaloader.Profile.from_username(loader.context, "username")
```

**yt-dlp>=2024.8.0**
- YouTube/TikTok downloader (better than youtube-dl)
- Extract metadata, transcripts, thumbnails
- Usage:
```python
import yt_dlp
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
```

**playwright>=1.48.0**
- Modern browser automation
- Better than Selenium for modern web apps
- Usage:
```python
from playwright.async_api import async_playwright
async with async_playwright() as p:
    browser = await p.chromium.launch()
    page = await browser.new_page()
    await page.goto(url)
```

### 2. Image Processing & Computer Vision

**opencv-python>=4.10.0**
- Computer vision and image processing
- Face detection, object recognition
- Usage:
```python
import cv2
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

**face-recognition>=1.3.0**
- Face recognition library
- Compare faces, detect faces in images
- Usage:
```python
import face_recognition
image = face_recognition.load_image_file("person.jpg")
encodings = face_recognition.face_encodings(image)
```

### 3. Advanced NLP & Sentiment Analysis

**vaderSentiment>=3.1.6**
- Sentiment analysis optimized for social media
- Works well with emojis and slang
- Usage:
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores("This is amazing! 😍")
```

**textstat>=0.7.3**
- Text statistics and readability
- Flesch reading ease, syllable count, etc.
- Usage:
```python
import textstat
readability = textstat.flesch_reading_ease(text)
syllables = textstat.syllable_count(text)
```

**emoji>=2.10.0**
- Emoji handling and analysis
- Usage:
```python
import emoji
text = "I love Python! 🐍"
emojis = emoji.emoji_list(text)
emoji_count = emoji.emoji_count(text)
```

**keybert>=0.8.0**
- Keyword extraction using BERT
- Usage:
```python
from keybert import KeyBERT
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
```

### 4. Task Queues & Background Jobs

**celery>=5.4.0**
- Distributed task queue
- Perfect for async profile extraction
- Usage:
```python
from celery import Celery
app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def extract_profile(username):
    # Long-running task
    return result
```

**rq>=1.16.0**
- Simple task queue (lighter than Celery)
- Usage:
```python
from rq import Queue
from redis import Redis
redis_conn = Redis()
q = Queue(connection=redis_conn)
job = q.enqueue(extract_profile, 'username')
```

### 5. Rate Limiting & Throttling

**slowapi>=0.1.9**
- Rate limiting for FastAPI
- Usage:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/endpoint")
@limiter.limit("5/minute")
async def endpoint():
    return {"message": "Hello"}
```

**tenacity>=9.0.0**
- Retry library with multiple strategies
- Usage:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def api_call():
    # Will retry with exponential backoff
    pass
```

### 6. Vector Databases & Embeddings

**chromadb>=0.5.0**
- Vector database for embeddings
- Store and search identity embeddings
- Usage:
```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("identities")
collection.add(embeddings=embeddings, ids=ids)
results = collection.query(query_embeddings=query, n_results=10)
```

**faiss-cpu>=1.7.4**
- Facebook AI Similarity Search
- Fast similarity search
- Usage:
```python
import faiss
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
distances, indices = index.search(query, k=10)
```

### 7. Monitoring & Observability

**sentry-sdk>=2.17.0**
- Error tracking and monitoring
- Usage:
```python
import sentry_sdk
sentry_sdk.init(dsn="your-dsn")
# Automatically captures exceptions
```

**prometheus-client>=0.21.0**
- Prometheus metrics
- Usage:
```python
from prometheus_client import Counter, Histogram
requests = Counter('requests_total', 'Total requests')
duration = Histogram('request_duration_seconds', 'Request duration')
```

### 8. Data Validation & Schema

**marshmallow>=3.21.0**
- Object serialization/deserialization
- Usage:
```python
from marshmallow import Schema, fields

class ProfileSchema(Schema):
    username = fields.Str(required=True)
    followers = fields.Int()

schema = ProfileSchema()
result = schema.load(data)
```

**jsonschema>=4.22.0**
- JSON schema validation
- Usage:
```python
import jsonschema
schema = {"type": "object", "properties": {"username": {"type": "string"}}}
jsonschema.validate(instance={"username": "test"}, schema=schema)
```

### 9. Content Analysis & Generation

**gradio>=4.19.0**
- UI for ML models
- Quick demos and interfaces
- Usage:
```python
import gradio as gr

def generate_content(prompt):
    return model.generate(prompt)

gr.Interface(fn=generate_content, inputs="text", outputs="text").launch()
```

**diffusers>=0.30.0**
- Diffusion models for image generation
- Usage:
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe(prompt).images[0]
```

### 10. Utilities

**humanize>=4.9.0**
- Human-readable values
- Usage:
```python
from humanize import naturaltime, naturalsize
naturaltime(datetime.now() - timedelta(days=1))  # "a day ago"
naturalsize(1024)  # "1.0 KB"
```

**python-dateutil>=2.9.0**
- Date parsing and manipulation
- Usage:
```python
from dateutil.parser import parse
date = parse("2025-01-15T10:30:00")
```

## Installation Groups

Install specific groups based on needs:

```bash
# Core (always needed)
pip install fastapi uvicorn pydantic sqlalchemy

# Social Media Scraping
pip install instaloader yt-dlp playwright

# Image Processing
pip install opencv-python pillow face-recognition

# NLP Advanced
pip install vaderSentiment textstat emoji keybert

# Task Queues
pip install celery redis rq

# Rate Limiting
pip install slowapi tenacity

# Vector Databases
pip install chromadb faiss-cpu

# Monitoring
pip install sentry-sdk prometheus-client

# All at once
pip install -r requirements.txt
```

## Performance Considerations

- **faiss-gpu**: Use GPU version if available (10-100x faster)
- **orjson**: Use for JSON-heavy workloads
- **asyncpg**: Use for PostgreSQL (faster than psycopg2)
- **httpx**: Better async performance than requests
- **ruff**: 10-100x faster linting than flake8

## Security Notes

- Always use official APIs when available
- Respect rate limits and terms of service
- Use proxies and rotation for scraping
- Encrypt sensitive data (cryptography library)
- Validate all inputs (pydantic, jsonschema)

## Migration Tips

1. **Replace aiohttp with httpx** for better async support
2. **Use spacy instead of NLTK** for better performance
3. **Use celery for long-running tasks** instead of threading
4. **Use chromadb for embeddings** instead of storing in database
5. **Use sentry for error tracking** instead of custom logging

