# BUL - Business Unlimited

**Advanced AI-powered document generation system for SMEs using OpenRouter and LangChain**

BUL (Business Unlimited) is a comprehensive system that continuously processes business queries and generates professional documents across all business areas for Small and Medium Enterprises (SMEs). The system keeps working until manually stopped, providing real-time document generation capabilities.

## 🚀 Features

- **Continuous Processing**: Keeps working until manually stopped
- **OpenRouter Integration**: Uses multiple AI models via OpenRouter
- **LangChain Integration**: Advanced prompt engineering and document generation
- **SME-Focused**: Specialized agents for different business areas
- **Multi-Format Output**: Supports Markdown, HTML, PDF, DOCX, and more
- **Real-time API**: RESTful API for query submission and document retrieval
- **Intelligent Analysis**: Automatic query analysis and business area detection
- **Comprehensive Templates**: Pre-built templates for various document types

## 🏗️ Architecture

```
bul/
├── core/                    # Core system components
│   ├── bul_engine.py       # Main BUL engine
│   ├── continuous_processor.py  # Continuous processing system
│   ├── ultra_adaptive_kv_cache_engine.py  # Ultra Adaptive KV Cache Engine ⚡
│   ├── ultra_adaptive_kv_cache_optimizer.py  # Cache optimizations
│   ├── ultra_adaptive_kv_cache_advanced_features.py  # Advanced cache features
│   ├── ultra_adaptive_kv_cache_monitor.py  # Real-time monitoring
│   ├── ultra_adaptive_kv_cache_security.py  # Security features
│   ├── ultra_adaptive_kv_cache_analytics.py  # Analytics
│   ├── ultra_adaptive_kv_cache_prometheus.py  # Prometheus metrics
│   ├── ultra_adaptive_kv_cache_cli.py  # CLI tool
│   ├── transformer_optimizer.py  # Transformer optimizations
│   └── diffusion_optimizer.py  # Diffusion model optimizations
├── agents/                  # Business area agents
│   └── sme_agent_manager.py # SME agent management
├── api/                     # API endpoints
│   └── bul_api.py          # FastAPI REST API
├── config/                  # Configuration
│   ├── bul_config.py       # Main configuration
│   └── openrouter_config.py # OpenRouter configuration
├── utils/                   # Utility classes
│   ├── document_processor.py # Document processing
│   └── query_analyzer.py    # Query analysis
├── templates/               # Document templates
├── main.py                  # Main entry point
└── requirements.txt         # Dependencies
```

## ⚡ Ultra Adaptive KV Cache Engine

El sistema BUL incluye un **Ultra Adaptive KV Cache Engine** de nivel empresarial que proporciona:

### Características Principales

- ✅ **Multi-GPU Support**: Detección automática y balanceo inteligente de carga entre GPUs
- ✅ **Adaptive Caching**: Políticas LRU, LFU, FIFO y Adaptive con ajuste automático
- ✅ **Persistence**: Persistencia de caché en disco y checkpointing automático
- ✅ **Performance Monitoring**: Métricas P50, P95, P99, throughput tracking
- ✅ **Session Management**: Gestión eficiente de sesiones y limpieza automática
- ✅ **Security**: Sanitización de requests, rate limiting, control de acceso
- ✅ **Real-time Monitoring**: Dashboard en tiempo real con métricas y alertas
- ✅ **Self-Healing**: Recuperación automática de errores y problemas
- ✅ **Advanced Features**: 
  - Request prefetching inteligente
  - Deduplicación automática de requests
  - Streaming de respuestas token por token
  - Priority queue (CRITICAL, HIGH, NORMAL, LOW)
  - Batch optimization automático
  - Adaptive throttling basado en carga del sistema

### Rendimiento

- **Throughput (Cached)**: 50-200 req/s concurrentes
- **Latency P50 (Cached)**: <100ms
- **Latency P95 (Cached)**: <500ms
- **Latency P99 (Cached)**: <1s
- **Batch Processing**: 100-500 req/s

### Documentación Completa

- 📖 [README Completo del KV Cache](core/README_ULTRA_ADAPTIVE_KV_CACHE.md)
- 📚 [Características Completas](core/ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md)
- 📝 [Documentación API](core/ULTRA_ADAPTIVE_KV_CACHE_DOCS.md)

### Uso Rápido del KV Cache

```python
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration

# Crear engine optimizado
engine = TruthGPTIntegration.create_engine_for_truthgpt()

# Procesar request con caché optimizado
result = await engine.process_request({
    'text': 'Tu consulta de negocio',
    'max_length': 100,
    'temperature': 0.7,
    'session_id': 'user_123'
})
```

## 🎯 Business Areas Supported

- **Marketing**: Strategy, campaigns, social media, brand guidelines
- **Sales**: Strategy, proposals, playbooks, customer management
- **Operations**: Manuals, workflows, quality management
- **HR**: Policies, training, recruitment, performance
- **Finance**: Planning, budgets, analysis, reporting
- **Legal**: Compliance, contracts, policies, risk management
- **Technical**: Documentation, systems, security, automation
- **Content**: Writing, blogs, manuals, training materials
- **Strategy**: Business plans, roadmaps, initiatives
- **Customer Service**: Support, satisfaction, retention

## 📋 Document Types Generated

- **Strategy Documents**: Business plans, marketing strategies, operational plans
- **Manuals**: Procedures, workflows, training materials
- **Templates**: Proposals, contracts, forms, checklists
- **Analysis Reports**: Market analysis, financial reports, performance reviews
- **Policies**: HR policies, operational procedures, compliance documents

## 🛠️ Installation

1. **Clone or navigate to the BUL directory**:
   ```bash
   cd C:\blatam-academy\agents\backend\onyx\server\features\bul
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your OpenRouter API key
   ```

4. **Set up OpenRouter API key**:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

## 🚀 Usage

### Command Line Interface

```bash
# Start full system (processor + API)
python main.py --mode full

# Start only continuous processor
python main.py --mode processor

# Start only API server
python main.py --mode api --port 8000

# Start with debug mode
python main.py --mode full --debug
```

### API Usage

1. **Start the system**:
   ```bash
   python main.py --mode full
   ```

2. **Submit a query**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{
          "query": "Create a marketing strategy for a new restaurant",
          "priority": 1
        }'
   ```

3. **Check task status**:
   ```bash
   curl "http://localhost:8000/task/{task_id}/status"
   ```

4. **Get generated documents**:
   ```bash
   curl "http://localhost:8000/task/{task_id}/documents"
   ```

### Python API

```python
from bul import BULEngine, ContinuousProcessor

# Initialize system
engine = BULEngine()
processor = ContinuousProcessor()

# Start continuous processing
await processor.start()

# Submit query
task_id = await engine.submit_query(
    query="Create a sales strategy for B2B software",
    priority=1
)

# Get results
documents = engine.get_completed_documents(task_id)
```

## 🔧 Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `BUL_DEBUG`: Enable debug mode (default: false)
- `BUL_API_HOST`: API server host (default: 0.0.0.0)
- `BUL_API_PORT`: API server port (default: 8000)
- `BUL_MAX_CONCURRENT_TASKS`: Max concurrent processing tasks (default: 5)

### Business Area Configuration

Enable/disable specific business areas in `config/bul_config.py`:

```python
enabled_areas = [
    "marketing", "sales", "operations", "hr", "finance",
    "legal", "technical", "content", "strategy", "customer_service"
]
```

## 📊 API Endpoints

### Core Endpoints

- `POST /query` - Submit a business query
- `GET /task/{task_id}/status` - Get task status
- `GET /task/{task_id}/documents` - Get generated documents
- `GET /documents` - List all documents
- `GET /search` - Search documents
- `GET /stats` - Get processing statistics

### Management Endpoints

- `GET /agents` - Get available business area agents
- `POST /processor/start` - Start continuous processor
- `POST /processor/stop` - Stop continuous processor
- `GET /processor/status` - Get processor status
- `GET /health` - Health check

## 🔄 Continuous Processing

The system operates in continuous mode by default:

1. **Query Submission**: Users submit business queries via API
2. **Query Analysis**: System analyzes query to determine business area and document types
3. **Task Creation**: Creates processing tasks with appropriate priority
4. **Document Generation**: Uses AI models to generate comprehensive documents
5. **Storage**: Saves documents in organized directory structure
6. **Continuous Loop**: Keeps processing until manually stopped

## 🎨 Document Templates

The system includes pre-built templates for:

- **Business Plans**: Executive summary, market analysis, financial projections
- **Marketing Strategies**: Target audience, marketing mix, budget allocation
- **Sales Proposals**: Client solutions, pricing, implementation plans
- **Operational Manuals**: Procedures, workflows, quality standards
- **HR Policies**: Employee guidelines, procedures, compliance

## 🔍 Query Analysis

The system automatically analyzes queries to:

- Determine primary business area
- Identify secondary relevant areas
- Select appropriate document types
- Assess complexity level
- Set processing priority
- Estimate processing time

## 📈 Monitoring and Statistics

Track system performance with:

- Total tasks processed
- Success/failure rates
- Average processing times
- Active/queued tasks
- Document generation statistics
- Business area distribution

## 🛡️ Security and Rate Limiting

- API key authentication (optional)
- Rate limiting per IP address
- Input validation and sanitization
- Secure document storage
- Error handling and logging

## 🚀 Deployment

### Development
```bash
python main.py --mode full --debug
```

### Production
```bash
python main.py --mode full --host 0.0.0.0 --port 8000
```

### Con KV Cache Optimizado
```bash
# Usar configuración de producción con KV cache
python main.py --mode full --host 0.0.0.0 --port 8000 --enable-kv-cache
```

### Docker (Future)
```bash
docker build -t bul-system .
docker run -p 8000:8000 bul-system
```

### Monitoreo del KV Cache

```bash
# Usar CLI para monitorear el cache
python core/ultra_adaptive_kv_cache_cli.py monitor --dashboard

# Ver estadísticas
python core/ultra_adaptive_kv_cache_cli.py stats

# Health check
python core/ultra_adaptive_kv_cache_cli.py health
```

## 📝 Example Queries

- "Create a marketing strategy for a new e-commerce store"
- "Develop a sales process for B2B software sales"
- "Write an operational manual for customer service"
- "Create HR policies for remote work"
- "Generate a financial plan for a startup"
- "Develop a content strategy for social media"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the Blatam Academy system.

## 📊 Monitoring y Métricas

### Métricas del KV Cache

El sistema incluye integración completa con Prometheus para métricas:

```python
from bulk.core.ultra_adaptive_kv_cache_prometheus import PrometheusMetrics

# Iniciar servidor de métricas
metrics = PrometheusMetrics()
metrics.start_server(port=9090)
```

### Health Checks

```python
from bulk.core.ultra_adaptive_kv_cache_health_checker import HealthChecker

health_checker = HealthChecker(engine)
status = await health_checker.check_health()
```

### Analytics

```python
from bulk.core.ultra_adaptive_kv_cache_analytics import Analytics

analytics = Analytics(engine)
report = analytics.generate_report()
```

## 🔧 Configuración Avanzada del KV Cache

### Configuración de Producción

```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigPreset

# Aplicar preset de producción
ConfigPreset.apply_preset(engine, 'production')

# Presets disponibles:
# - development: Configuración para desarrollo
# - production: Configuración optimizada para producción
# - high_performance: Máximo rendimiento
# - memory_efficient: Optimizado para memoria
# - bulk_processing: Para procesamiento masivo
```

### Configuración Dinámica

```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigManager

config_manager = ConfigManager(engine, config_file='config.json')

# Actualizar configuración en tiempo de ejecución
await config_manager.update_config('cache_size', 32768)

# Recargar desde archivo
await config_manager.reload_from_file()
```

## 🛡️ Seguridad

El KV Cache incluye características de seguridad empresarial:

- ✅ **Sanitización de Requests**: Protección contra XSS, SQL injection, path traversal
- ✅ **Rate Limiting**: Múltiples estrategias (sliding window, token bucket)
- ✅ **Access Control**: IP whitelist/blacklist, validación de API keys
- ✅ **HMAC Validation**: Validación de firmas de requests
- ✅ **Security Monitoring**: Seguimiento y alertas de eventos de seguridad

```python
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper

secure_engine = SecureEngineWrapper(
    engine,
    enable_sanitization=True,
    enable_rate_limiting=True,
    enable_access_control=True
)
```

## 🔄 Backup y Restauración

```python
from bulk.core.ultra_adaptive_kv_cache_backup import BackupManager, ScheduledBackup

# Crear backup
backup_mgr = BackupManager(engine)
backup_path = backup_mgr.create_backup(compress=True)

# Restaurar
backup_mgr.restore_backup(backup_path)

# Backup programado
scheduler = ScheduledBackup(backup_mgr, interval_hours=24)
await scheduler.start()
```

## 📚 Guías Adicionales

### Guía de Uso Avanzado

Para uso avanzado del sistema BUL, consulta la [Guía de Uso Avanzado](ADVANCED_USAGE_GUIDE.md) que incluye:

- ⚡ Optimización avanzada del KV Cache
- 🔗 Integración con otros sistemas (FastAPI, Celery)
- 🎯 Patrones de uso avanzados
- 🔧 Tuning de rendimiento
- 📈 Escalabilidad y producción
- 🛡️ Seguridad avanzada
- 📊 Monitoring y alertas personalizadas

### Ejemplos Avanzados

```python
# Ejemplo completo de producción
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper
from bulk.core.ultra_adaptive_kv_cache_monitor import PerformanceMonitor

# Setup completo
engine = TruthGPTIntegration.create_engine_for_truthgpt()
secure_engine = SecureEngineWrapper(engine, enable_sanitization=True)
monitor = PerformanceMonitor(secure_engine)
await monitor.start_monitoring()

# Procesar con todas las optimizaciones
result = await secure_engine.process_request_secure(
    request,
    client_ip="192.168.1.100",
    api_key="your-key"
)
```

## 🆘 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the logs in `bul.log`
- Check system status at `/health`
- Check KV Cache documentation: `core/README_ULTRA_ADAPTIVE_KV_CACHE.md`
- Use CLI tool: `python core/ultra_adaptive_kv_cache_cli.py --help`
- Consult [Advanced Usage Guide](ADVANCED_USAGE_GUIDE.md) for advanced use cases

## 🔗 Recursos Relacionados

- [Guía de Arquitectura](../ARCHITECTURE_GUIDE.md)
- [Guía de Inicio Rápido](../QUICK_START_GUIDE.md)
- [README Principal del Sistema](../README.md)

---

**BUL - Business Unlimited**: Empowering SMEs with AI-driven document generation powered by Ultra Adaptive KV Cache Engine. ⚡

**Características Destacadas:**
- ✅ Sistema de caché ultra optimizado
- ✅ Soporte multi-GPU
- ✅ Monitoreo en tiempo real
- ✅ Seguridad empresarial
- ✅ Auto-tuning inteligente
- ✅ Escalabilidad horizontal

