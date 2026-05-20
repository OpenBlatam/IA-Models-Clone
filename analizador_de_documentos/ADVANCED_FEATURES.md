# Características Avanzadas

## 🎯 Nuevas Funcionalidades Implementadas

### 1. Comparador de Documentos (`DocumentComparator`)

Sistema para comparar documentos y detectar similitud semántica.

**Características:**
- Comparación semántica usando embeddings
- Detección de keywords y entidades comunes
- Análisis de diferencias detallado
- Búsqueda de documentos similares en corpus
- Detección de plagio

**Uso:**
```python
from core.document_comparator import DocumentComparator

comparator = DocumentComparator(analyzer)

# Comparar dos documentos
similarity = await comparator.compare_documents(
    doc1_content="Texto del documento 1",
    doc2_content="Texto del documento 2",
    doc1_id="doc1",
    doc2_id="doc2"
)

print(f"Similitud: {similarity.similarity_score:.2%}")
print(f"Keywords comunes: {similarity.common_keywords}")

# Encontrar documentos similares
similar_docs = await comparator.find_similar_documents(
    target_doc="Documento objetivo",
    document_corpus=[("doc1", "Contenido..."), ("doc2", "Contenido...")],
    threshold=0.7,
    top_k=5
)
```

**API:**
```bash
POST /api/analizador-documentos/advanced/compare
POST /api/analizador-documentos/advanced/find-similar
POST /api/analizador-documentos/advanced/detect-plagiarism
```

### 2. Extractor de Información Estructurada (`StructuredExtractor`)

Extrae información estructurada de documentos según schemas personalizados.

**Métodos de extracción:**
- **Entity**: Reconocimiento de entidades nombradas
- **Keyword**: Extracción de palabras clave
- **Classification**: Clasificación de documentos
- **QA**: Pregunta-respuesta
- **Regex**: Expresiones regulares
- **Auto**: Extracción automática inteligente

**Uso:**
```python
from core.structured_extractor import StructuredExtractor, create_extraction_schema

extractor = StructuredExtractor(analyzer)

# Definir schema
schema = create_extraction_schema([
    {"name": "author", "type": "person", "method": "entity"},
    {"name": "date", "type": "date", "method": "entity"},
    {"name": "keywords", "type": "list", "method": "keyword", "limit": 10},
    {"name": "category", "type": "string", "method": "classification"},
    {"name": "summary", "type": "string", "method": "qa", "question": "What is this about?"}
])

# Extraer información
data = await extractor.extract_structured_data(content, schema)
```

**API:**
```bash
POST /api/analizador-documentos/advanced/extract-structured
```

### 3. Analizador de Estilo y Legibilidad (`StyleAnalyzer`)

Analiza el estilo de escritura, legibilidad y calidad de documentos.

**Métricas proporcionadas:**
- Estadísticas de escritura (palabras, oraciones, párrafos)
- Score de legibilidad (0-100)
- Complejidad del texto
- Tono y sentimiento
- Riqueza de vocabulario
- Densidad de puntuación
- Evaluación de calidad general

**Uso:**
```python
from core.style_analyzer import StyleAnalyzer

style_analyzer = StyleAnalyzer(analyzer)

# Analizar estilo
style = await style_analyzer.analyze_writing_style(content)

print(f"Legibilidad: {style['readability_score']}/100")
print(f"Complejidad: {style['complexity']}")
print(f"Tono: {style['tone']}")

# Evaluar calidad
quality = await style_analyzer.assess_quality(content)
print(f"Score de calidad: {quality['quality_score']}/100")
print(f"Calificación: {quality['grade']}")
```

**API:**
```bash
POST /api/analizador-documentos/advanced/analyze-style
POST /api/analizador-documentos/advanced/assess-quality
```

### 4. Sistema de Exportación (`ResultExporter`)

Exporta resultados en múltiples formatos.

**Formatos soportados:**
- JSON
- CSV
- Markdown
- HTML

**Uso:**
```python
from utils.exporters import ResultExporter

# Exportar a JSON
ResultExporter.export_json(data, "results.json")

# Exportar a CSV
ResultExporter.export_csv(data_list, "results.csv")

# Exportar a Markdown
ResultExporter.export_markdown(data, "results.md")

# Exportar a HTML
ResultExporter.export_html(data, "results.html")

# Exportar en múltiples formatos
ResultExporter.export_multiple_formats(
    data,
    "results",
    formats=["json", "csv", "markdown", "html"]
)
```

**API:**
```bash
POST /api/analizador-documentos/advanced/export
```

## 📊 Casos de Uso

### 1. Detección de Plagio

```python
comparator = DocumentComparator(analyzer)

# Comparar documento sospechoso con corpus de referencia
plagiarism_results = await comparator.detect_plagiarism(
    suspicious_document="Documento a verificar...",
    reference_corpus=[
        ("ref1", "Documento de referencia 1"),
        ("ref2", "Documento de referencia 2")
    ],
    threshold=0.85
)

for result in plagiarism_results:
    if result["risk_level"] == "high":
        print(f"⚠️ Posible plagio detectado: {result['reference_document_id']}")
```

### 2. Extracción de Información de Facturas

```python
extractor = StructuredExtractor(analyzer)

invoice_schema = create_extraction_schema([
    {"name": "invoice_number", "method": "qa", "question": "What is the invoice number?"},
    {"name": "date", "type": "date", "method": "entity"},
    {"name": "total_amount", "method": "qa", "question": "What is the total amount?"},
    {"name": "vendor", "type": "organization", "method": "entity"},
])

invoice_data = await extractor.extract_structured_data(invoice_content, invoice_schema)
```

### 3. Análisis de Calidad de Contenido

```python
style_analyzer = StyleAnalyzer(analyzer)

# Analizar calidad
quality = await style_analyzer.assess_quality(content)

if quality["grade"] >= "B":
    print("✅ Contenido de buena calidad")
    for feedback in quality["feedback"]:
        print(feedback)
else:
    print("⚠️ El contenido necesita mejoras")
```

### 4. Búsqueda Semántica de Documentos

```python
comparator = DocumentComparator(analyzer)

# Encontrar documentos similares
similar_docs = await comparator.find_similar_documents(
    target_doc="Documento de búsqueda",
    document_corpus=all_documents,
    threshold=0.7,
    top_k=10
)

for doc in similar_docs:
    print(f"📄 {doc.document2_id}: {doc.similarity_score:.2%} similitud")
```

## 🔗 Integración Completa

```python
from core.document_analyzer import DocumentAnalyzer
from core.document_comparator import DocumentComparator
from core.structured_extractor import StructuredExtractor
from core.style_analyzer import StyleAnalyzer

# Inicializar analizador
analyzer = DocumentAnalyzer()

# Inicializar componentes avanzados
comparator = DocumentComparator(analyzer)
extractor = StructuredExtractor(analyzer)
style_analyzer = StyleAnalyzer(analyzer)

# 1. Analizar documento
result = await analyzer.analyze_document(content)

# 2. Comparar con otros documentos
similarity = await comparator.compare_documents(doc1, doc2)

# 3. Extraer información estructurada
structured_data = await extractor.extract_structured_data(content, schema)

# 4. Analizar estilo y calidad
style = await style_analyzer.analyze_writing_style(content)
quality = await style_analyzer.assess_quality(content)

# 5. Exportar resultados
from utils.exporters import ResultExporter
ResultExporter.export_multiple_formats({
    "analysis": result,
    "similarity": similarity,
    "structured": structured_data,
    "style": style,
    "quality": quality
}, "complete_analysis")
```

## 📈 Beneficios

- **Comparación de documentos**: Detecta similitud y posibles plagios
- **Extracción estructurada**: Obtén datos específicos según tus necesidades
- **Análisis de calidad**: Evalúa y mejora la calidad de tus documentos
- **Exportación flexible**: Resultados en el formato que necesites
- **Búsqueda semántica**: Encuentra documentos relacionados inteligentemente

## 🚀 Próximas Mejoras

- Integración con bases de datos vectoriales
- Análisis de tendencias temporales
- Dashboard web interactivo
- OCR mejorado para imágenes
- Análisis multi-idioma avanzado
- Validación de documentos con reglas personalizadas

---

**Versión**: 1.2.0  
**Última actualización**: 2024
















