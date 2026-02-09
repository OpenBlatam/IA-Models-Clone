# Document Workflow Chain

## 🚀 Descripción

El **Document Workflow Chain** es un sistema de IA profesional y optimizado que permite la generación continua de documentos mediante encadenamiento inteligente de prompts. Cada documento generado se convierte automáticamente en la entrada para generar el siguiente documento, creando flujos de trabajo eficientes y escalables.

### ✨ Características Principales

- **Generación Continua**: Crea documentos de forma ininterrumpida usando una sola consulta inicial
- **Encadenamiento Inteligente**: Cada documento alimenta automáticamente la generación del siguiente
- **Gestión de Flujos**: Control completo sobre el estado y progreso de los workflows
- **Generación de Títulos**: Crea títulos atractivos para blogs automáticamente
- **API REST Completa**: Endpoints para integración con otros sistemas
- **Persistencia de Estado**: Mantiene el historial completo de cada cadena de documentos
- **Múltiples Clientes de IA**: Soporte para OpenAI, Anthropic, Cohere y clientes personalizados
- **Base de Datos Avanzada**: Persistencia completa con PostgreSQL y SQLAlchemy
- **Dashboard Web**: Interfaz visual para monitoreo y gestión de workflows
- **Integraciones Inteligentes**: Conexión automática con otros servicios de Blatam Academy
- **Análisis de Calidad**: Sistema de puntuación automática de calidad de contenido
- **Optimización SEO**: Integración automática con servicios de SEO
- **Detección de Redundancia**: Verificación automática de contenido único
- **Consistencia de Marca**: Análisis y ajuste automático del tono de marca
- **Plantillas de Contenido**: Sistema de plantillas predefinidas para diferentes tipos de contenido
- **Soporte Multi-idioma**: Generación de contenido en 6 idiomas principales
- **Analytics Avanzado**: Sistema de análisis predictivo y tendencias
- **Análisis de Sentimientos**: Evaluación automática del tono y sentimiento del contenido
- **Optimización de Prompts**: Sistema inteligente de optimización de prompts
- **Métricas de Rendimiento**: Seguimiento detallado de métricas de calidad y rendimiento

## 🏗️ Arquitectura

```
Document Workflow Chain
├── main.py                     # Aplicación principal FastAPI
├── start.py                    # Script de inicio optimizado
├── system_config.py            # Configuración centralizada del sistema
├── workflow_chain_engine.py    # Motor principal del sistema
├── api_endpoints.py            # Endpoints REST API
├── ai_clients.py              # Integración con clientes de IA
├── database.py                # Modelos y operaciones de base de datos
├── dashboard.py               # Dashboard web de monitoreo
├── integrations.py            # Integraciones con servicios Blatam
├── external_integrations.py   # Integraciones con servicios externos
├── content_analyzer.py        # Análisis avanzado de contenido
├── content_templates.py       # Sistema de plantillas de contenido
├── multilang_support.py       # Soporte multi-idioma
├── advanced_analytics.py      # Analytics y análisis predictivo
├── advanced_analysis.py       # Análisis avanzado de documentos
├── content_quality_control.py # Control de calidad de contenido
├── content_versioning.py      # Sistema de versionado
├── workflow_scheduler.py      # Programador de workflows
├── workflow_automation.py     # Automatización de workflows
├── intelligent_generation.py  # Generación inteligente
├── trend_analysis.py          # Análisis de tendencias
├── ai_optimization.py         # Optimización de IA
├── intelligent_cache.py       # Sistema de caché inteligente
├── test_workflow.py           # Suite de pruebas
├── examples/                   # Ejemplos y demos
├── templates/                  # Plantillas HTML para dashboard
├── Dockerfile                  # Configuración Docker
├── docker-compose.yml          # Orquestación de servicios
├── nginx.conf                  # Configuración de load balancer
├── requirements.txt            # Dependencias Python
├── env.example                 # Variables de entorno
└── README.md                   # Este archivo
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.11+
- Docker y Docker Compose
- Cliente de IA (OpenAI, Anthropic, Cohere, etc.)

### Instalación Local

```bash
# Clonar o navegar al directorio
cd document_workflow_chain

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
export AI_API_KEY="tu_api_key_aqui"
export AI_CLIENT_TYPE="openai"  # o anthropic, cohere
export AI_MODEL="gpt-4"         # o claude-3, etc.
```

### Instalación con Docker

```bash
# Construir y ejecutar con Docker Compose
docker-compose up --build

# O ejecutar solo el servicio principal
docker build -t document-workflow-chain .
docker run -p 8001:8000 -e AI_API_KEY=tu_key document-workflow-chain
```

## 📖 Uso Básico

### 1. Crear una Cadena de Workflow

```python
from workflow_chain_engine import WorkflowChainEngine

# Inicializar el motor
engine = WorkflowChainEngine(ai_client=tu_cliente_ai)

# Crear una nueva cadena
chain = await engine.create_workflow_chain(
    name="Guía de Marketing Digital",
    description="Serie de artículos sobre marketing digital",
    initial_prompt="Escribe una introducción al marketing digital moderno"
)
```

### 2. Continuar la Cadena

```python
# Generar el siguiente documento automáticamente
next_doc = await engine.continue_workflow_chain(chain.id)

# O con un prompt personalizado
next_doc = await engine.continue_workflow_chain(
    chain.id, 
    continuation_prompt="Ahora escribe sobre estrategias de SEO"
)
```

### 3. Generar Títulos de Blog

```python
# Generar un título atractivo
title = await engine.generate_blog_title(
    "Contenido del blog sobre inteligencia artificial..."
)
print(title)  # "Inteligencia Artificial: El Futuro del Marketing Digital"
```

## 🎛️ Dashboard Web

### Acceso al Dashboard
- **URL**: `http://localhost:8020/dashboard/`
- **Características**:
  - Visualización en tiempo real de workflows activos
  - Estadísticas de rendimiento y calidad
  - Gestión de cadenas (pausar, reanudar, completar)
  - Gráficos de tendencias de calidad
  - Métricas de uso de tokens y tiempo de generación

### Funcionalidades del Dashboard
- **Monitoreo en Tiempo Real**: Ve el estado de todos tus workflows
- **Gestión Visual**: Controla tus cadenas con clicks simples
- **Analytics Avanzados**: Gráficos de rendimiento y calidad
- **Exportación de Datos**: Descarga estadísticas y reportes

## 🔗 Integraciones Inteligentes

### Servicios Integrados
- **Content Redundancy Detector**: Verifica unicidad del contenido
- **SEO Optimizer**: Optimiza automáticamente para motores de búsqueda
- **Brand Voice Analyzer**: Mantiene consistencia de marca
- **Blog Publisher**: Publica automáticamente en tu blog

### Flujo de Integración
1. **Generación**: Crea contenido con IA
2. **Verificación**: Detecta redundancia automáticamente
3. **Optimización**: Mejora SEO y consistencia de marca
4. **Publicación**: Publica o programa automáticamente

## 🎨 Plantillas de Contenido

### Tipos de Plantillas Disponibles
- **Blog Post**: Artículos estructurados con introducción, cuerpo y conclusión
- **Tutorial**: Guías paso a paso con instrucciones detalladas
- **Product Description**: Descripciones de productos optimizadas para ventas
- **News Article**: Artículos de noticias con estructura periodística
- **Social Media Post**: Publicaciones optimizadas para redes sociales

### Uso de Plantillas
```python
# Crear workflow con plantilla
chain = await engine.create_workflow_with_template(
    template_id="blog_post",
    topic="Inteligencia Artificial en Marketing",
    name="Serie de IA Marketing",
    description="Artículos sobre IA en marketing",
    language_code="es",
    word_count=1500,
    tone="profesional",
    audience="marketing professionals"
)
```

## 🌍 Soporte Multi-idioma

### Idiomas Soportados
- **Inglés (en)**: Contenido profesional y conversacional
- **Español (es)**: Contenido cálido y amigable
- **Francés (fr)**: Contenido elegante y formal
- **Portugués (pt)**: Contenido cálido y acogedor
- **Alemán (de)**: Contenido preciso y profesional
- **Italiano (it)**: Contenido apasionado y expresivo

### Adaptación Cultural
- Formato de fechas localizado
- Formato de números regional
- Estilo de escritura culturalmente apropiado
- Nivel de formalidad adaptado
- Palabras clave localizadas

## 📊 Analytics Avanzado

### Métricas de Rendimiento
- **Puntuación de Calidad**: Evaluación automática de calidad del contenido
- **Tiempo de Generación**: Métricas de velocidad de generación
- **Uso de Tokens**: Seguimiento de costos y eficiencia
- **Puntuación de Engagement**: Medición de potencial de engagement
- **Puntuación SEO**: Evaluación de optimización para motores de búsqueda
- **Legibilidad**: Análisis de facilidad de lectura

### Análisis Predictivo
- **Tendencias de Calidad**: Predicción de tendencias de calidad
- **Optimización de Costos**: Recomendaciones para reducir costos
- **Mejora de Rendimiento**: Sugerencias de optimización
- **Insights de Engagement**: Predicciones de engagement

### Dashboard de Analytics
- Visualización en tiempo real de métricas
- Gráficos de tendencias interactivos
- Alertas automáticas de rendimiento
- Reportes de optimización personalizados

## 🌐 API REST

### Endpoints Principales

#### Crear Workflow Chain
```http
POST /api/v1/document-workflow-chain/create
Content-Type: application/json

{
    "name": "Mi Serie de Blog",
    "description": "Serie sobre tecnología",
    "initial_prompt": "Escribe sobre el futuro de la IA"
}
```

#### Continuar Workflow Chain
```http
POST /api/v1/document-workflow-chain/continue
Content-Type: application/json

{
    "chain_id": "uuid-del-workflow",
    "continuation_prompt": "Continúa con el siguiente tema"
}
```

#### Obtener Historial de la Cadena
```http
GET /api/v1/document-workflow-chain/chain/{chain_id}/history
```

#### Generar Título de Blog
```http
POST /api/v1/document-workflow-chain/generate-title
Content-Type: application/json

{
    "content": "Contenido del blog para generar título..."
}
```

#### Crear Workflow con Plantilla
```http
POST /api/v1/document-workflow-chain/create-with-template
Content-Type: application/json

{
    "template_id": "blog_post",
    "topic": "Inteligencia Artificial en Marketing",
    "name": "Serie de IA Marketing",
    "description": "Artículos sobre IA en marketing",
    "language_code": "es",
    "word_count": 1500,
    "tone": "profesional",
    "audience": "marketing professionals"
}
```

#### Analizar Contenido
```http
POST /api/v1/document-workflow-chain/analyze-content
Content-Type: application/json

{
    "content": "Contenido a analizar...",
    "title": "Título del documento",
    "language_code": "es"
}
```

#### Obtener Análisis de Rendimiento
```http
GET /api/v1/document-workflow-chain/chain/{chain_id}/performance
```

#### Obtener Insights del Workflow
```http
GET /api/v1/document-workflow-chain/chain/{chain_id}/insights
```

#### Obtener Plantillas Disponibles
```http
GET /api/v1/document-workflow-chain/templates?category=blogging
```

#### Obtener Idiomas Soportados
```http
GET /api/v1/document-workflow-chain/languages
```

#### Obtener Resumen de Analytics
```http
GET /api/v1/document-workflow-chain/analytics/summary
```

### Gestión de Workflows

```http
# Pausar workflow
POST /api/v1/document-workflow-chain/chain/{chain_id}/pause

# Reanudar workflow
POST /api/v1/document-workflow-chain/chain/{chain_id}/resume

# Completar workflow
POST /api/v1/document-workflow-chain/chain/{chain_id}/complete

# Exportar workflow
GET /api/v1/document-workflow-chain/chain/{chain_id}/export
```

## 🎯 Casos de Uso

### 1. Serie de Blog Automática
```python
# Crear una serie completa de blog posts
chain = await engine.create_workflow_chain(
    name="Guía Completa de SEO",
    description="Serie de 10 artículos sobre SEO",
    initial_prompt="Escribe una introducción completa al SEO"
)

# Generar 9 artículos más automáticamente
for i in range(9):
    await engine.continue_workflow_chain(chain.id)
```

### 2. Contenido Educativo Escalonado
```python
# Crear contenido educativo progresivo
chain = await engine.create_workflow_chain(
    name="Curso de Python",
    description="Lecciones progresivas de Python",
    initial_prompt="Escribe la lección 1: Introducción a Python"
)

# Cada lección se basa en la anterior
for lesson in range(2, 11):
    await engine.continue_workflow_chain(
        chain.id,
        f"Escribe la lección {lesson} basándote en la anterior"
    )
```

### 3. Documentación Técnica
```python
# Generar documentación técnica completa
chain = await engine.create_workflow_chain(
    name="Documentación API",
    description="Documentación completa de la API",
    initial_prompt="Escribe la introducción a nuestra API REST"
)
```

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
# Configuración de IA
AI_CLIENT_TYPE=openai          # openai, anthropic, cohere
AI_API_KEY=tu_api_key
AI_MODEL=gpt-4                 # gpt-4, claude-3, etc.

# Configuración de Base de Datos
POSTGRES_PASSWORD=tu_password
REDIS_URL=redis://localhost:6379

# Configuración de Logging
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
```

### Personalización del Motor

```python
# Configuración personalizada
settings = {
    "max_chain_length": 50,
    "auto_title_generation": True,
    "content_quality_threshold": 0.8,
    "continuation_strategy": "semantic_similarity"
}

chain = await engine.create_workflow_chain(
    name="Mi Workflow",
    description="Workflow personalizado",
    initial_prompt="Prompt inicial",
    settings=settings
)
```

## 📊 Monitoreo y Logs

### Health Check
```http
GET /api/v1/document-workflow-chain/health
```

Respuesta:
```json
{
    "status": "healthy",
    "active_chains": 5,
    "completed_chains": 12,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Logs Estructurados
El sistema genera logs detallados para:
- Creación de workflows
- Generación de documentos
- Errores y excepciones
- Métricas de rendimiento

## 🧪 Testing

### Ejecutar Demo
```bash
python examples/demo.py
```

### Tests Unitarios
```bash
pytest tests/
```

### Tests de Integración
```bash
pytest tests/integration/
```

## 🚀 Despliegue en Producción

### Docker Compose
```bash
# Despliegue completo con base de datos y Redis
docker-compose -f docker-compose.yml up -d
```

### Kubernetes
```yaml
# Ejemplo de deployment para Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-workflow-chain
spec:
  replicas: 3
  selector:
    matchLabels:
      app: document-workflow-chain
  template:
    metadata:
      labels:
        app: document-workflow-chain
    spec:
      containers:
      - name: document-workflow-chain
        image: document-workflow-chain:latest
        ports:
        - containerPort: 8000
        env:
        - name: AI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-secrets
              key: api-key
```

## 🔒 Seguridad

- Autenticación JWT para endpoints
- Rate limiting para prevenir abuso
- Validación de entrada en todos los endpoints
- Logs de auditoría para seguimiento
- Encriptación de datos sensibles

## 📈 Escalabilidad

- Arquitectura stateless para horizontal scaling
- Cache con Redis para optimización
- Base de datos PostgreSQL para persistencia
- Load balancing con Nginx
- Monitoreo con Prometheus

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

- **Documentación**: [Wiki del proyecto]
- **Issues**: [GitHub Issues]
- **Discusiones**: [GitHub Discussions]
- **Email**: soporte@blatam-academy.com

## 🎉 Agradecimientos

- Equipo de Blatam Academy
- Comunidad de desarrolladores
- Contribuidores de código abierto

---

**Document Workflow Chain** - Transformando la creación de contenido con IA 🚀
