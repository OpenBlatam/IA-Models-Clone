# Real AI Document Processor

A practical, functional AI document processing system using real, existing technologies.

## Features

### Core Capabilities
- **Text Analysis**: Basic text statistics and metrics
- **Sentiment Analysis**: Emotion and sentiment detection using NLTK
- **Text Classification**: Automatic text categorization using Transformers
- **Text Summarization**: Automatic summarization using BART model
- **Question Answering**: Answer questions about document content
- **Keyword Extraction**: Extract important keywords and phrases
- **Language Detection**: Detect the language of the text
- **Named Entity Recognition**: Extract entities using spaCy
- **Part-of-Speech Tagging**: Analyze grammatical structure

### Technologies Used
- **FastAPI**: Modern, fast web framework
- **spaCy**: Advanced NLP library
- **NLTK**: Natural Language Toolkit
- **Transformers**: Hugging Face transformers library
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning library

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Install Dependencies
```bash
pip install -r real_requirements.txt
```

### Install spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### Install NLTK Data
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### Start the Server
```bash
python real_app.py
```

The server will start on `http://localhost:8000`

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Core Endpoints

#### Process Text
```bash
POST /api/v1/real-documents/process-text
```
Process text with AI capabilities.

**Parameters:**
- `text` (string): Text to process
- `task` (string): Task type (analyze, sentiment, classify, summarize)

#### Analyze Sentiment
```bash
POST /api/v1/real-documents/analyze-sentiment
```
Analyze sentiment of text.

**Parameters:**
- `text` (string): Text to analyze

#### Classify Text
```bash
POST /api/v1/real-documents/classify-text
```
Classify text into categories.

**Parameters:**
- `text` (string): Text to classify

#### Summarize Text
```bash
POST /api/v1/real-documents/summarize-text
```
Generate summary of text.

**Parameters:**
- `text` (string): Text to summarize

#### Extract Keywords
```bash
POST /api/v1/real-documents/extract-keywords
```
Extract keywords from text.

**Parameters:**
- `text` (string): Text to analyze
- `top_n` (int): Number of keywords to extract (default: 10)

#### Detect Language
```bash
POST /api/v1/real-documents/detect-language
```
Detect language of text.

**Parameters:**
- `text` (string): Text to analyze

#### Answer Question
```bash
POST /api/v1/real-documents/answer-question
```
Answer questions about document content.

**Parameters:**
- `context` (string): Document context
- `question` (string): Question to answer

### Utility Endpoints

#### Health Check
```bash
GET /api/v1/real-documents/health
```
Check system health and model status.

#### Get Capabilities
```bash
GET /api/v1/real-documents/capabilities
```
Get available AI capabilities.

## Example Usage

### Using curl

#### Analyze Text
```bash
curl -X POST "http://localhost:8000/api/v1/real-documents/process-text" \
  -F "text=This is a great product! I love it." \
  -F "task=analyze"
```

#### Analyze Sentiment
```bash
curl -X POST "http://localhost:8000/api/v1/real-documents/analyze-sentiment" \
  -F "text=This is a great product! I love it."
```

#### Summarize Text
```bash
curl -X POST "http://localhost:8000/api/v1/real-documents/summarize-text" \
  -F "text=Your long document text here..."
```

#### Extract Keywords
```bash
curl -X POST "http://localhost:8000/api/v1/real-documents/extract-keywords" \
  -F "text=Your document text here..." \
  -F "top_n=5"
```

### Using Python

```python
import requests

# Analyze text
response = requests.post(
    "http://localhost:8000/api/v1/real-documents/process-text",
    data={
        "text": "This is a great product! I love it.",
        "task": "analyze"
    }
)
result = response.json()
print(result)

# Analyze sentiment
response = requests.post(
    "http://localhost:8000/api/v1/real-documents/analyze-sentiment",
    data={"text": "This is a great product! I love it."}
)
sentiment = response.json()
print(sentiment)
```

## Configuration

Create a `.env` file to configure the system:

```env
# Application settings
DEBUG=false
LOG_LEVEL=INFO

# AI Model settings
SPACY_MODEL=en_core_web_sm
MAX_TEXT_LENGTH=5120

# Processing settings
ENABLE_SPACY=true
ENABLE_NLTK=true
ENABLE_TRANSFORMERS=true

# API settings
RATE_LIMIT_PER_MINUTE=100
MAX_FILE_SIZE_MB=10

# Optional: OpenAI integration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

## Model Information

### spaCy Model
- **Model**: `en_core_web_sm`
- **Purpose**: Named entity recognition, part-of-speech tagging
- **Size**: ~12 MB
- **Languages**: English

### NLTK
- **Purpose**: Sentiment analysis, tokenization
- **Models**: VADER sentiment analyzer
- **Languages**: English

### Transformers Models
- **Classification**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Summarization**: `facebook/bart-large-cnn`
- **Question Answering**: `distilbert-base-cased-distilled-squad`

## Performance

### System Requirements
- **RAM**: 4GB+ recommended
- **CPU**: 2+ cores recommended
- **Storage**: 2GB+ for models

### Processing Times
- **Basic Analysis**: < 1 second
- **Sentiment Analysis**: < 1 second
- **Text Classification**: 1-3 seconds
- **Summarization**: 2-5 seconds
- **Question Answering**: 1-3 seconds

## Troubleshooting

### Common Issues

#### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

#### NLTK Data Missing
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

#### Transformers Models Not Loading
- Check internet connection
- Ensure sufficient disk space
- Check PyTorch installation

#### Memory Issues
- Reduce `MAX_TEXT_LENGTH` in configuration
- Use smaller models
- Increase system RAM

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
```

### Type Checking
```bash
mypy .
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Create an issue on GitHub













