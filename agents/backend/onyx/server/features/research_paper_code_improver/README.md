# Research Paper Code Improver

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

AI system that allows you to:
1. **Upload research papers** via PDFs or links
2. **Extract complete information** from papers
3. **Train a model** based on the extracted knowledge
4. **Improve GitHub code** using the trained model

## 🚀 Key Features

- 📄 **PDF Processing**: Extraction of text, figures, tables, and references
- 🔗 **Link Processing**: Download and analysis of papers from URLs
- 🧠 **Model Training**: Fine-tuning of language models based on papers
- 💻 **Code Improvement**: Analysis and improvement of GitHub code using paper knowledge
- 🔍 **GitHub Analysis**: Integration with GitHub API to examine repositories
- 📊 **Vector Database**: Storage of embeddings for semantic search (ChromaDB)
- 🧬 **RAG (Retrieval Augmented Generation)**: Code improvements using relevant papers + LLMs
- 💾 **Persistent Storage**: Paper database with search and management
- 🤖 **LLM Integration**: Support for OpenAI GPT-4 and Anthropic Claude
- 🔎 **Semantic Search**: Find relevant papers for code improvements

## 📁 Structure

```
research_paper_code_improver/
├── core/                    # Main business logic
│   ├── paper_extractor.py   # Paper information extraction
│   ├── model_trainer.py     # Model training
│   └── code_improver.py     # Code improvement
├── api/                     # API Endpoints
│   ├── routes.py            # Main routes
│   └── schemas.py           # Pydantic schemas
├── config/                  # Configurations
│   └── settings.py          # System configuration
├── utils/                   # Utilities
│   ├── pdf_processor.py     # PDF processing
│   ├── link_downloader.py   # Link downloading
│   └── github_integration.py # GitHub integration
├── models/                  # Data models
│   └── paper_model.py       # Paper models
├── training/                # Training scripts
│   └── train.py             # Main training script
├── data/                    # Stored data
│   ├── papers/              # Downloaded PDFs
│   ├── embeddings/          # Generated embeddings
│   └── models/              # Trained models
├── main.py                  # Main FastAPI application
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Installation

```bash
cd research_paper_code_improver
pip install -r requirements.txt
```

## 💻 Basic Usage

### 1. Load Papers

```python
from core.paper_extractor import PaperExtractor

extractor = PaperExtractor()

# From PDF
paper = extractor.extract_from_pdf("path/to/paper.pdf")

# From link
paper = extractor.extract_from_link("https://arxiv.org/pdf/...")
```

### 2. Train Model

```python
from core.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train_from_papers(papers=[paper1, paper2, ...])
```

### 3. Improve Code

```python
from core.code_improver import CodeImprover

improver = CodeImprover()
improved_code = improver.improve_code(
    github_repo="user/repo",
    file_path="src/main.py"
)
```

## 🔗 API Endpoints

- `POST /api/papers/upload` - Upload PDF
- `POST /api/papers/link` - Process link
- `POST /api/training/train` - Train model
- `POST /api/code/improve` - Improve GitHub code
- `GET /api/models/status` - Model status

## 📊 Models Used

- **Base Model**: GPT-4, Claude, or open-source models (Llama, Mistral)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (with memory fallback)
- **RAG**: Retrieval Augmented Generation with semantic search

## 🔒 Security

- PDF file validation
- Endpoint rate limiting
- JWT authentication for GitHub API
- Input sanitization

## 📈 Roadmap

- [ ] Support for multiple formats (LaTeX, Markdown)
- [ ] Integration with more sources (arXiv, PubMed, etc.)
- [ ] Visualization dashboard
- [ ] Before/after improvement comparison
- [ ] Export of trained models
