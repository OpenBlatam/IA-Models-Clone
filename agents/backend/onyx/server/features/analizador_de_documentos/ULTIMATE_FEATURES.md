# 🏆 Funcionalidades Ultimate - Document Analyzer

## ✨ Nuevas Funcionalidades Ultimate Implementadas

### 1. 🔌 Sistema de Plugins
- **Sistema extensible**: Agregar funcionalidades personalizadas
- **Plugins dinámicos**: Cargar/descargar plugins en tiempo de ejecución
- **Hooks**: Sistema de hooks para extensibilidad
- **Gestión**: Registrar, inicializar y gestionar plugins

```python
from analizador_de_documentos.core.document_plugins import DocumentPlugin, PluginInfo

# Crear plugin personalizado
class CustomAnalysisPlugin(DocumentPlugin):
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="custom_analysis",
            version="1.0.0",
            description="Plugin personalizado de análisis"
        )
    
    async def initialize(self):
        logger.info("Plugin personalizado inicializado")
    
    async def custom_analysis(self, content: str):
        return {"custom_result": "Análisis personalizado"}

# Registrar plugin
analyzer = DocumentAnalyzer()
plugin = CustomAnalysisPlugin(analyzer)
analyzer.register_plugin(plugin)

# Inicializar plugins
await analyzer.initialize_plugins()

# Usar plugin
custom_plugin = analyzer.get_plugin("custom_analysis")
result = await custom_plugin.custom_analysis(document_content)
```

### 2. ⚡ Análisis en Tiempo Real
- **Streaming de eventos**: Eventos en tiempo real durante el análisis
- **Progreso en vivo**: Actualizaciones de progreso
- **Historial de eventos**: Registro de todos los eventos
- **Análisis activos**: Monitoreo de análisis en curso

```python
analyzer = DocumentAnalyzer()

# Callback de progreso
async def on_progress(progress: float, message: str):
    print(f"Progreso: {progress:.1f}% - {message}")

# Análisis en tiempo real
async for event in analyzer.analyze_realtime(
    document_id="doc_123",
    content=document_content,
    tasks=["classification", "summarization"],
    on_progress=on_progress
):
    print(f"[{event.event_type}] {event.message}")
    
    if event.event_type == "complete":
        print(f"Resultado: {event.data['result']}")
        print(f"Tiempo: {event.data['processing_time']:.2f}s")
    
    elif event.event_type == "error":
        print(f"Error: {event.message}")

# Obtener historial de eventos
history = analyzer.realtime_analyzer.get_event_history("doc_123")
for event in history:
    print(f"{event.timestamp}: {event.event_type} - {event.message}")

# Ver análisis activos
active = analyzer.realtime_analyzer.get_active_analyses()
print(f"Análisis activos: {len(active)}")
```

### 3. 💾 Integración con Bases de Datos
- **Persistencia**: Guardar resultados de análisis
- **Historial**: Recuperar historial de análisis
- **Múltiples adaptadores**: SQLite, PostgreSQL, MongoDB (placeholders)
- **Búsqueda**: Buscar análisis por criterios

```python
analyzer = DocumentAnalyzer()

# Guardar análisis en base de datos
import time
start = time.time()
result = await analyzer.analyze_document(document_content)
quality = await analyzer.analyze_quality(result.content)
grammar = await analyzer.analyze_grammar(result.content)
processing_time = time.time() - start

await analyzer.save_analysis_to_database(
    document_id="doc_123",
    analysis_result=result,
    analysis_type="full",
    quality_score=quality.overall_score,
    grammar_score=grammar.overall_score,
    processing_time=processing_time
)

# Obtener historial de análisis
history = await analyzer.get_analysis_from_database("doc_123")
print(f"Total análisis guardados: {len(history)}")

for record in history:
    print(f"  {record.timestamp}: {record.analysis_type}")
    print(f"    Calidad: {record.quality_score}")
    print(f"    Gramática: {record.grammar_score}")
    print(f"    Tiempo: {record.processing_time:.2f}s")

# Obtener último análisis
latest = await analyzer.database.get_latest_analysis("doc_123")
if latest:
    print(f"Último análisis: {latest.timestamp}")
```

## 🎯 Ejemplos Completos

### Ejemplo 1: Workflow Ultimate Completo
```python
from analizador_de_documentos.core.document_analyzer import DocumentAnalyzer
from analizador_de_documentos.core.document_plugins import DocumentPlugin, PluginInfo

# 1. Crear analizador
analyzer = DocumentAnalyzer()

# 2. Registrar plugin personalizado
class MyCustomPlugin(DocumentPlugin):
    def get_info(self):
        return PluginInfo(name="my_plugin", version="1.0.0")
    
    async def initialize(self):
        print("Plugin inicializado")
    
    async def enhance_analysis(self, result):
        return {"enhanced": True, "original": result}

plugin = MyCustomPlugin(analyzer)
analyzer.register_plugin(plugin)
await analyzer.initialize_plugins()

# 3. Análisis en tiempo real con progreso
async def show_progress(progress, message):
    print(f"📊 {progress:.0f}% - {message}")

async for event in analyzer.analyze_realtime(
    document_id="doc_123",
    content=document_content,
    on_progress=show_progress
):
    if event.event_type == "complete":
        result = event.data["result"]
        
        # 4. Guardar en base de datos
        await analyzer.save_analysis_to_database(
            document_id="doc_123",
            analysis_result=result,
            processing_time=event.data["processing_time"]
        )
        
        # 5. Usar plugin para mejorar
        if plugin:
            enhanced = await plugin.enhance_analysis(result)
            print(f"Resultado mejorado: {enhanced}")

# 6. Recuperar historial
history = await analyzer.get_analysis_from_database("doc_123")
print(f"Total análisis en BD: {len(history)}")
```

### Ejemplo 2: Sistema de Plugins con Hooks
```python
analyzer = DocumentAnalyzer()

# Registrar hook personalizado
async def on_analysis_complete(document_id, result):
    print(f"Análisis completado para {document_id}")
    # Enviar notificación, actualizar dashboard, etc.

analyzer.plugin_manager.register_hook("analysis_complete", on_analysis_complete)

# Análisis que dispara hook
result = await analyzer.analyze_document(document_content="...")

# Disparar hook manualmente
await analyzer.plugin_manager.trigger_hook(
    "analysis_complete",
    "doc_123",
    result
)
```

### Ejemplo 3: Monitoreo en Tiempo Real
```python
analyzer = DocumentAnalyzer()

# Múltiples análisis en paralelo
documents = [
    ("doc_1", "Contenido 1..."),
    ("doc_2", "Contenido 2..."),
    ("doc_3", "Contenido 3..."),
]

async def analyze_with_monitoring(doc_id, content):
    async for event in analyzer.analyze_realtime(doc_id, content):
        print(f"[{doc_id}] {event.event_type}: {event.message}")
        
        if event.event_type == "complete":
            # Guardar automáticamente
            await analyzer.save_analysis_to_database(
                doc_id, event.data["result"],
                processing_time=event.data["processing_time"]
            )

# Ejecutar en paralelo
tasks = [
    analyze_with_monitoring(doc_id, content)
    for doc_id, content in documents
]

await asyncio.gather(*tasks)

# Ver análisis activos
active = analyzer.realtime_analyzer.get_active_analyses()
print(f"Análisis activos: {active}")
```

## 🚀 Ventajas Ultimate

- **Extensibilidad**: Sistema de plugins para funcionalidades personalizadas
- **Tiempo Real**: Análisis con streaming de eventos
- **Persistencia**: Base de datos para historial completo
- **Hooks**: Sistema de eventos para integraciones
- **Monitoreo**: Seguimiento de análisis activos
- **Escalabilidad**: Listo para producción enterprise

---

**Estado**: ✅ **Todas las Funcionalidades Ultimate Implementadas**
