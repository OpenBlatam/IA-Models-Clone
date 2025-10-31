# AI Document Classifier v2.0 - Enhanced Edition

An advanced AI-powered system that can identify document types from a single text query and export appropriate template designs with comprehensive analytics, batch processing, and external service integration.

## 🚀 Enhanced Features

### Core Classification
- **Multi-Method Classification**: AI, ML, and pattern-based classification
- **Advanced NLP**: SpaCy and NLTK integration for sophisticated text analysis
- **Machine Learning Models**: Random Forest, Gradient Boosting, Naive Bayes, and ensemble methods
- **External AI Services**: OpenAI GPT, Hugging Face, and other AI providers
- **Feature Extraction**: Comprehensive linguistic and structural analysis

### Template System
- **Dynamic Template Generation**: AI-generated templates based on requirements
- **Multiple Complexity Levels**: Basic, Intermediate, Advanced, Professional
- **Customizable Styles**: Academic, Business, Creative, Technical presets
- **Multi-Format Export**: JSON, YAML, Markdown, HTML, PDF
- **Industry-Specific Templates**: Technology, Healthcare, Finance, Legal, etc.

### Batch Processing & Analytics
- **High-Performance Batch Processing**: Multi-threading and multiprocessing support
- **Comprehensive Analytics**: Performance metrics, confidence distributions, error analysis
- **Caching System**: Intelligent caching with TTL and performance optimization
- **Progress Tracking**: Real-time batch processing status and progress callbacks
- **Export Capabilities**: Analytics export in JSON and CSV formats

### External Integrations
- **Translation Services**: Google Translate integration for multi-language support
- **Grammar Checking**: Grammarly and other grammar services
- **Plagiarism Detection**: Copyscape and content originality checking
- **Content Analysis**: Readability, sentiment, and content quality analysis
- **Document Generation**: External document generation services

## Supported Document Types

- **Novel** (Fiction, Science Fiction, Romance)
- **Contract** (Service Agreement, Employment Contract)
- **Design** (Technical Design, Architectural Design, Product Design)
- **Business Plan**
- **Academic Paper**
- **Technical Manual**
- **Marketing Material**
- **User Manual**
- **Report**
- **Proposal**

## 🚀 Quick Start

### Automated Setup (Recommended)

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd ai_document_classifier
   python setup.py
   ```

2. **Configure API keys** (optional):
   ```bash
   # Edit .env file with your API keys
   OPENAI_API_KEY=your_openai_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key_here
   ```

3. **Start the enhanced server**:
   ```bash
   python main.py
   ```

4. **Access the APIs**:
   - Standard API: `http://localhost:8000/docs`
   - Enhanced API: `http://localhost:8000/ai-document-classifier/v2/docs`

### Using Docker (Production)

1. **Build and run**:
   ```bash
   docker-compose up -d
   ```

2. **Access at**: `http://localhost:8001`

### Manual Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLP models**:
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('all')"
   ```

3. **Run setup**:
   ```bash
   python setup.py --skip-deps
   ```

4. **Start server**:
   ```bash
   python main.py
   ```

## API Usage

### Classify a Document

```bash
curl -X POST "http://localhost:8001/ai-document-classifier/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "I want to write a science fiction novel about space exploration",
       "use_ai": true
     }'
```

### Get Templates for a Document Type

```bash
curl "http://localhost:8001/ai-document-classifier/templates/novel"
```

### Export a Template

```bash
curl -X POST "http://localhost:8001/ai-document-classifier/export-template" \
     -H "Content-Type: application/json" \
     -d '{
       "document_type": "novel",
       "template_name": "Standard Novel",
       "format": "markdown"
     }'
```

### Classify and Export in One Call

```bash
curl "http://localhost:8001/ai-document-classifier/classify-and-export?query=I%20want%20to%20write%20a%20contract&format=json"
```

## Demo Script

Run the interactive demo:

```bash
python examples/demo.py
```

This will:
- Test the classifier with various document types
- Show classification accuracy
- Allow interactive testing
- Demonstrate template export functionality

## API Documentation

Once the service is running, visit:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for AI-powered classification (optional)
- `DEBUG`: Enable debug mode (default: false)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

### Template Customization

Templates are stored in YAML format in the `templates/` directory. You can:

1. Modify existing templates
2. Add new document types
3. Create custom template designs

Example template structure:
```yaml
- name: "Custom Template"
  document_type: "novel"
  sections:
    - name: "Title Page"
      required: true
      description: "Book title and author"
  formatting:
    font: "Times New Roman"
    size: 12
    line_spacing: 1.5
  metadata:
    pages_per_chapter: 10
    total_chapters: 20
```

## Architecture

```
ai_document_classifier/
├── document_classifier_engine.py  # Core classification logic
├── api_endpoints.py               # FastAPI endpoints
├── main.py                        # Application entry point
├── templates/                     # Template definitions
│   ├── novel_templates.yaml
│   ├── contract_templates.yaml
│   └── design_templates.yaml
├── examples/                      # Demo and examples
│   └── demo.py
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml            # Docker Compose setup
└── README.md                     # This file
```

## Classification Methods

### 1. AI-Powered Classification (Recommended)
- Uses OpenAI GPT models
- Higher accuracy for complex queries
- Requires OpenAI API key
- Better handling of context and nuance

### 2. Pattern-Based Classification (Fallback)
- Uses keyword matching and regex patterns
- No external dependencies
- Fast and reliable
- Good for clear document type indicators

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Run the demo script for examples
3. Review the template files for customization
4. Check the logs for debugging information
