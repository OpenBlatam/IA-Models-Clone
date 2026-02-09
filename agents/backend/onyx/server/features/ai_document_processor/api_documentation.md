# API Documentation - AI Document Processor

## 📚 Documentación Completa de la API

### Base URL
```
http://localhost:8001
```

### Autenticación
Actualmente no requiere autenticación, pero se puede configurar JWT en el futuro.

---

## 🔄 Endpoints Principales

### 1. **POST** `/ai-document-processor/process`

Procesa un documento completo: extrae texto, clasifica y transforma a formato profesional.

#### Parámetros
- **file** (multipart/form-data): Archivo a procesar
- **target_format** (form-data): Formato objetivo (`consultancy`, `technical`, `academic`, `commercial`, `legal`)
- **language** (form-data): Idioma del documento (`es`, `en`, `fr`, etc.)

#### Ejemplo de Request
```bash
curl -X POST "http://localhost:8001/ai-document-processor/process" \
  -F "file=@documento.pdf" \
  -F "target_format=consultancy" \
  -F "language=es"
```

#### Ejemplo de Response
```json
{
  "success": true,
  "original_filename": "documento.pdf",
  "classification": {
    "filename": "documento.pdf",
    "document_type": "pdf",
    "area": "business",
    "category": "consultancy_report",
    "confidence": 0.85,
    "language": "es",
    "word_count": 1250,
    "key_topics": ["negocio", "estrategia", "mercado", "análisis", "recomendaciones"],
    "summary": "Documento de análisis empresarial con recomendaciones estratégicas...",
    "metadata": {
      "classification_methods": {
        "pattern": 0.8,
        "ai": 0.9,
        "ml": 0.0
      }
    }
  },
  "professional_document": {
    "title": "Informe de Consultoría",
    "format": "consultancy",
    "language": "es",
    "content": "# Informe de Consultoría\n\n## Resumen Ejecutivo\n\n...",
    "structure": {
      "header": "INFORME DE CONSULTORÍA",
      "subtitle": "Análisis y Recomendaciones Estratégicas",
      "footer": "Documento confidencial - 15/10/2025"
    },
    "sections": [
      {
        "name": "Resumen Ejecutivo",
        "content": "El análisis revela...",
        "word_count": 150,
        "order": 1
      }
    ],
    "metadata": {
      "transformation_method": "ai",
      "original_length": 5000,
      "transformed_length": 3500,
      "template_used": "consultancy"
    }
  },
  "message": "Documento procesado exitosamente"
}
```

---

### 2. **POST** `/ai-document-processor/classify`

Solo clasifica un documento sin transformarlo.

#### Parámetros
- **file** (multipart/form-data): Archivo a clasificar

#### Ejemplo de Request
```bash
curl -X POST "http://localhost:8001/ai-document-processor/classify" \
  -F "file=@documento.docx"
```

#### Ejemplo de Response
```json
{
  "success": true,
  "filename": "documento.docx",
  "classification": {
    "filename": "documento.docx",
    "document_type": "word",
    "area": "technology",
    "category": "technical_documentation",
    "confidence": 0.92,
    "language": "es",
    "word_count": 800,
    "key_topics": ["software", "sistema", "desarrollo", "arquitectura", "implementación"],
    "summary": "Documentación técnica de sistema de software...",
    "metadata": {
      "classification_methods": {
        "pattern": 0.9,
        "ai": 0.95,
        "ml": 0.0
      }
    }
  }
}
```

---

### 3. **POST** `/ai-document-processor/transform`

Transforma texto en documento profesional.

#### Parámetros
- **text** (form-data): Texto a transformar
- **target_format** (form-data): Formato objetivo
- **language** (form-data): Idioma
- **document_type** (form-data, opcional): Tipo de documento

#### Ejemplo de Request
```bash
curl -X POST "http://localhost:8001/ai-document-processor/transform" \
  -d "text=Este es un análisis de mercado digital..." \
  -d "target_format=consultancy" \
  -d "language=es"
```

#### Ejemplo de Response
```json
{
  "success": true,
  "professional_document": {
    "title": "Informe de Consultoría",
    "format": "consultancy",
    "language": "es",
    "content": "# Informe de Consultoría\n\n## Resumen Ejecutivo\n\n...",
    "structure": {
      "header": "INFORME DE CONSULTORÍA",
      "subtitle": "Análisis y Recomendaciones Estratégicas",
      "footer": "Documento confidencial - 15/10/2025"
    },
    "sections": [
      {
        "name": "Resumen Ejecutivo",
        "content": "El análisis de mercado digital...",
        "word_count": 120,
        "order": 1
      }
    ],
    "metadata": {
      "transformation_method": "ai",
      "original_length": 500,
      "transformed_length": 800,
      "template_used": "consultancy"
    }
  },
  "message": "Documento transformado exitosamente"
}
```

---

### 4. **GET** `/ai-document-processor/health`

Verifica el estado del servicio.

#### Ejemplo de Request
```bash
curl -X GET "http://localhost:8001/ai-document-processor/health"
```

#### Ejemplo de Response
```json
{
  "status": "healthy",
  "service": "AI Document Processor",
  "version": "1.0.0",
  "features": {
    "document_processing": "active",
    "ai_classification": "active",
    "professional_transformation": "active"
  }
}
```

---

### 5. **GET** `/ai-document-processor/supported-formats`

Obtiene los formatos soportados.

#### Ejemplo de Request
```bash
curl -X GET "http://localhost:8001/ai-document-processor/supported-formats"
```

#### Ejemplo de Response
```json
{
  "input_formats": [
    {
      "extension": ".md",
      "type": "Markdown",
      "description": "Documentos Markdown con formato"
    },
    {
      "extension": ".pdf",
      "type": "PDF",
      "description": "Documentos PDF con extracción de texto"
    },
    {
      "extension": ".docx",
      "type": "Word",
      "description": "Documentos Word (nuevo formato)"
    },
    {
      "extension": ".doc",
      "type": "Word",
      "description": "Documentos Word (formato antiguo)"
    },
    {
      "extension": ".txt",
      "type": "Texto",
      "description": "Archivos de texto plano"
    }
  ],
  "output_formats": [
    {
      "format": "consultancy",
      "description": "Documentos de consultoría profesional",
      "sections": ["Resumen Ejecutivo", "Análisis", "Recomendaciones"]
    },
    {
      "format": "technical",
      "description": "Documentación técnica",
      "sections": ["Introducción", "Especificaciones", "Implementación"]
    },
    {
      "format": "academic",
      "description": "Documentos académicos",
      "sections": ["Resumen", "Metodología", "Resultados", "Conclusiones"]
    },
    {
      "format": "commercial",
      "description": "Documentos comerciales",
      "sections": ["Propuesta de Valor", "Análisis de Mercado", "Estrategia"]
    },
    {
      "format": "legal",
      "description": "Documentos legales",
      "sections": ["Definiciones", "Términos", "Condiciones", "Cláusulas"]
    }
  ]
}
```

---

## 📊 Códigos de Estado HTTP

| Código | Descripción |
|--------|-------------|
| 200 | OK - Solicitud exitosa |
| 400 | Bad Request - Parámetros inválidos |
| 413 | Payload Too Large - Archivo demasiado grande |
| 415 | Unsupported Media Type - Formato no soportado |
| 422 | Unprocessable Entity - Error de validación |
| 500 | Internal Server Error - Error interno del servidor |

---

## ⚠️ Manejo de Errores

### Error de Archivo No Válido
```json
{
  "detail": "Archivo no válido: formato no soportado"
}
```

### Error de Procesamiento
```json
{
  "detail": "Error procesando documento: No se pudo extraer texto del archivo"
}
```

### Error de Clasificación
```json
{
  "detail": "Error clasificando documento: OpenAI API no disponible"
}
```

---

## 🔧 Ejemplos de Uso en Diferentes Lenguajes

### Python
```python
import requests

# Procesar documento
with open('documento.pdf', 'rb') as f:
    files = {'file': f}
    data = {
        'target_format': 'consultancy',
        'language': 'es'
    }
    response = requests.post(
        'http://localhost:8001/ai-document-processor/process',
        files=files,
        data=data
    )
    result = response.json()
    print(result)
```

### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('documento.pdf'));
form.append('target_format', 'consultancy');
form.append('language', 'es');

axios.post('http://localhost:8001/ai-document-processor/process', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log(response.data);
})
.catch(error => {
  console.error(error);
});
```

### cURL
```bash
# Procesar documento
curl -X POST "http://localhost:8001/ai-document-processor/process" \
  -F "file=@documento.pdf" \
  -F "target_format=consultancy" \
  -F "language=es"

# Clasificar documento
curl -X POST "http://localhost:8001/ai-document-processor/classify" \
  -F "file=@documento.docx"

# Transformar texto
curl -X POST "http://localhost:8001/ai-document-processor/transform" \
  -d "text=Este es un análisis de mercado..." \
  -d "target_format=technical" \
  -d "language=es"
```

---

## 📈 Límites y Restricciones

### Archivos
- **Tamaño máximo**: 50MB
- **Formatos soportados**: .md, .pdf, .docx, .doc, .txt
- **Timeout de procesamiento**: 5 minutos

### Texto
- **Longitud máxima**: 100,000 caracteres
- **Idiomas soportados**: Español, Inglés (otros en desarrollo)

### API
- **Rate limiting**: 60 requests/minuto
- **Procesamiento concurrente**: 10 requests simultáneos

---

## 🔍 Documentación Interactiva

Para documentación interactiva completa, visita:
```
http://localhost:8001/docs
```

Esta página incluye:
- Interfaz Swagger UI
- Ejemplos interactivos
- Esquemas de datos
- Pruebas en vivo


