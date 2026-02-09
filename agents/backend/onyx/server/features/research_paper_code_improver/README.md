# Research Paper Code Improver

## 📋 Descripción

Sistema de IA que permite:
1. **Cargar papers de investigación** mediante PDFs o links
2. **Extraer información completa** de los papers
3. **Entrenar un modelo** basado en el conocimiento extraído
4. **Mejorar código de GitHub** usando el modelo entrenado

## 🚀 Características Principales

- 📄 **Procesamiento de PDFs**: Extracción de texto, figuras, tablas y referencias
- 🔗 **Procesamiento de Links**: Descarga y análisis de papers desde URLs
- 🧠 **Entrenamiento de Modelo**: Fine-tuning de modelos de lenguaje basado en papers
- 💻 **Mejora de Código**: Análisis y mejora de código de GitHub usando conocimiento de papers
- 🔍 **Análisis de GitHub**: Integración con GitHub API para examinar repositorios
- 📊 **Vector Database**: Almacenamiento de embeddings para búsqueda semántica (ChromaDB)
- 🧬 **RAG (Retrieval Augmented Generation)**: Mejoras de código usando papers relevantes + LLMs
- 💾 **Almacenamiento Persistente**: Base de datos de papers con búsqueda y gestión
- 🤖 **Integración LLM**: Soporte para OpenAI GPT-4 y Anthropic Claude
- 🔎 **Búsqueda Semántica**: Encuentra papers relevantes para mejoras de código

## 📁 Estructura

```
research_paper_code_improver/
├── core/                    # Lógica de negocio principal
│   ├── paper_extractor.py   # Extracción de información de papers
│   ├── model_trainer.py     # Entrenamiento de modelos
│   └── code_improver.py     # Mejora de código
├── api/                     # Endpoints de API
│   ├── routes.py            # Rutas principales
│   └── schemas.py           # Esquemas Pydantic
├── config/                  # Configuraciones
│   └── settings.py          # Configuración del sistema
├── utils/                   # Utilidades
│   ├── pdf_processor.py     # Procesamiento de PDFs
│   ├── link_downloader.py   # Descarga de links
│   └── github_integration.py # Integración con GitHub
├── models/                  # Modelos de datos
│   └── paper_model.py       # Modelos de papers
├── training/                # Scripts de entrenamiento
│   └── train.py             # Script principal de entrenamiento
├── data/                    # Datos almacenados
│   ├── papers/              # PDFs descargados
│   ├── embeddings/          # Embeddings generados
│   └── models/              # Modelos entrenados
├── main.py                  # Aplicación FastAPI principal
├── requirements.txt         # Dependencias
└── README.md               # Este archivo
```

## 🔧 Instalación

```bash
cd research_paper_code_improver
pip install -r requirements.txt
```

## 💻 Uso Básico

### 1. Cargar Papers

```python
from core.paper_extractor import PaperExtractor

extractor = PaperExtractor()

# Desde PDF
paper = extractor.extract_from_pdf("path/to/paper.pdf")

# Desde link
paper = extractor.extract_from_link("https://arxiv.org/pdf/...")
```

### 2. Entrenar Modelo

```python
from core.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train_from_papers(papers=[paper1, paper2, ...])
```

### 3. Mejorar Código

```python
from core.code_improver import CodeImprover

improver = CodeImprover()
improved_code = improver.improve_code(
    github_repo="user/repo",
    file_path="src/main.py"
)
```

## 🔗 API Endpoints

- `POST /api/papers/upload` - Subir PDF
- `POST /api/papers/link` - Procesar link
- `POST /api/training/train` - Entrenar modelo
- `POST /api/code/improve` - Mejorar código de GitHub
- `GET /api/models/status` - Estado del modelo

## 📊 Modelos Utilizados

- **Base Model**: GPT-4, Claude, o modelos open-source (Llama, Mistral)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (con fallback a memoria)
- **RAG**: Retrieval Augmented Generation con búsqueda semántica

## 🔒 Seguridad

- Validación de archivos PDF
- Rate limiting en endpoints
- Autenticación JWT para GitHub API
- Sanitización de inputs

## 📈 Roadmap

- [ ] Soporte para múltiples formatos (LaTeX, Markdown)
- [ ] Integración con más fuentes (arXiv, PubMed, etc.)
- [ ] Dashboard de visualización
- [ ] Comparación de mejoras antes/después
- [ ] Exportación de modelos entrenados

