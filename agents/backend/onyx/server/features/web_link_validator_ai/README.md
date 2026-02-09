# Web Link Validator AI

AI-powered web link validator that checks if links are real, accessible, and relevant to user queries. Solves the problem of AI-generated fake links that don't exist or aren't relevant.

## Features

- ✅ **URL Existence Validation**: Checks if URLs are accessible via HTTP/HTTPS
- 🤖 **AI-Powered Relevance Analysis**: Uses OpenRouter AI to verify if links are relevant to queries
- 🔍 **Legitimacy Detection**: Identifies fake or generated links
- ⚡ **Batch Processing**: Validate multiple links in parallel
- 🚀 **Fast & Async**: Built with FastAPI for high performance

## Problem Solved

When asking AI assistants (like ChatGPT) for links, they often provide fake URLs that:
- Don't exist
- Don't relate to the query
- Are hallucinated/generated

This service validates links using:
1. HTTP checks to verify existence
2. AI analysis to check relevance and legitimacy
3. Content analysis when available

## Installation

```bash
cd agents/backend/onyx/server/features/web_link_validator_ai
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o
PORT=8025
```

## Usage

### Start the server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8025 --reload
```

### API Endpoints

#### 1. Validate Single Link

```bash
curl -X POST "http://localhost:8025/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "query": "Python documentation"
  }'
```

Response:
```json
{
  "url": "https://example.com",
  "valid": true,
  "exists": true,
  "relevant": false,
  "relevance_score": 0.2,
  "is_legitimate": true,
  "reason": "URL exists but not relevant to query",
  "suggestions": ["https://docs.python.org"],
  "timestamp": "2025-11-25T12:00:00"
}
```

#### 2. Validate Multiple Links

```bash
curl -X POST "http://localhost:8025/validate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com",
      "https://docs.python.org"
    ],
    "query": "Python documentation"
  }'
```

#### 3. Quick Check (Existence Only)

```bash
curl "http://localhost:8025/check/https://example.com"
```

#### 4. Health Check

```bash
curl "http://localhost:8025/health"
```

## How It Works

1. **URL Parsing**: Validates URL format
2. **HTTP Check**: Attempts to access the URL
3. **Content Fetching**: Retrieves page content (if accessible)
4. **AI Analysis**: Uses OpenRouter to:
   - Verify URL structure legitimacy
   - Check content relevance to query
   - Identify fake/generated links
   - Suggest real alternatives if needed

## Integration with OpenRouter

The service uses OpenRouter to access multiple AI models. Configure your preferred model:

- `openai/gpt-4o` (default)
- `anthropic/claude-3-opus`
- `google/gemini-pro`
- Or any other model supported by OpenRouter

## Response Fields

- `valid`: Overall validation result (exists AND relevant AND legitimate)
- `exists`: URL is accessible
- `relevant`: Content is relevant to query (AI analysis)
- `relevance_score`: 0.0-1.0 relevance score
- `is_legitimate`: URL appears to be a real website (not fake)
- `reason`: Explanation of validation result
- `suggestions`: Alternative real links if original is fake

## Use Cases

- Validate links from AI responses
- Check if generated URLs are real
- Verify link relevance before sharing
- Batch validate multiple links
- API integration for link validation services

## Port

Default port: **8025**

## License

Part of Blatam Academy platform

