# Refactorización: Sistemas Unificados de Procesamiento, Workflow y Export

## Resumen

Esta refactorización consolida servicios de procesamiento, workflow y exportación en sistemas unificados para mejorar la arquitectura y simplificar el uso.

## Cambios Realizados

### 1. Unified Processing System (`unified_processing_system.py`)

**Consolida:**
- `VideoProcessor` (procesamiento de video)
- `ImageProcessor` (procesamiento de imagen)
- `ColorAnalyzer` (análisis de color)
- `ColorMatcher` (matching de color)
- `VideoQualityAnalyzer` (análisis de calidad)

**Características:**
- Interfaz unificada para todo el procesamiento de media
- Soporte multi-formato
- Auto-detección de tipo de media
- Análisis de color y calidad integrado
- Matching de colores

**Tipos de Media:**
- `VIDEO`: Procesamiento de video
- `IMAGE`: Procesamiento de imagen
- `AUTO`: Auto-detección

**Uso:**
```python
from services.unified_processing_system import UnifiedProcessingSystem, MediaType

processor = UnifiedProcessingSystem(ffmpeg_path="/usr/bin/ffmpeg")

# Procesar media con auto-detección
result = await processor.process_media(
    input_path="input.mp4",
    output_path="output.mp4",
    color_params={"brightness": 0.1, "contrast": 1.2},
    analyze=True,
    analyze_quality=True
)

# Match colores
color_match = await processor.match_colors(
    input_path="image.jpg",
    reference="reference.jpg"
)
```

**Ubicación:** `services/unified_processing_system.py`

### 2. Unified Workflow System (`unified_workflow_system.py`)

**Consolida:**
- `WorkflowManager` (gestión de workflows)
- `ServiceOrchestrator` (orquestación de servicios)
- `DataPipeline` (pipelines de datos)

**Características:**
- Interfaz unificada para workflows
- Auto-selección de modo de ejecución
- Orquestación de servicios
- Pipelines de datos
- Conversión automática entre formatos

**Modos de Ejecución:**
- `WORKFLOW`: Usa WorkflowManager (workflows simples)
- `ORCHESTRATION`: Usa ServiceOrchestrator (workflows complejos)
- `PIPELINE`: Usa DataPipeline (procesamiento de datos)
- `AUTO`: Auto-selección basada en complejidad

**Uso:**
```python
from services.unified_workflow_system import UnifiedWorkflowSystem, WorkflowMode

workflow_system = UnifiedWorkflowSystem(
    workflows_dir="workflows",
    services=all_services
)

# Ejecutar workflow (auto-selecciona modo)
result = await workflow_system.execute_workflow(
    workflow_id="color_grading_pipeline",
    input_data={"image_path": "input.jpg"},
    mode=WorkflowMode.AUTO
)
```

**Ubicación:** `services/unified_workflow_system.py`

### 3. Unified Export System (`unified_export_system.py`)

**Consolida:**
- `ParameterExporter` (exportación de parámetros)
- `ComparisonGenerator` (generación de comparaciones)

**Características:**
- Interfaz unificada para exportación
- Múltiples formatos de exportación
- Generación de comparaciones
- Templates personalizados

**Formatos de Exportación:**
- `JSON`: Formato JSON
- `XML`: Formato XML
- `CSV`: Formato CSV
- `YAML`: Formato YAML
- `LUT`: Look-Up Table
- `PRESET`: Preset de color

**Uso:**
```python
from services.unified_export_system import UnifiedExportSystem, ExportFormat

export_system = UnifiedExportSystem(output_dir="exports")

# Exportar parámetros
result = await export_system.export_parameters(
    color_params={"brightness": 0.1, "contrast": 1.2},
    format=ExportFormat.JSON
)

# Generar comparación
comparison = await export_system.generate_comparison(
    original_path="original.jpg",
    graded_path="graded.jpg",
    layout="side_by_side"
)
```

**Ubicación:** `services/unified_export_system.py`

## Actualizaciones en Service Factory

El `RefactoredServiceFactory` ha sido actualizado para incluir los nuevos sistemas unificados:

```python
# En _init_processing()
"unified_processing_system": UnifiedProcessingSystem(
    ffmpeg_path=self.config.video_processing.ffmpeg_path,
    histogram_bins=self.config.color_analysis.histogram_bins,
    color_space=self.config.color_analysis.color_space
)

# En _init_support()
"unified_export_system": UnifiedExportSystem(
    output_dir=self._get_storage_path("exports")
)

# En _init_advanced()
"unified_workflow_system": UnifiedWorkflowSystem(
    workflows_dir=self._get_storage_path("workflows"),
    services=self._services
)
```

## Compatibilidad hacia Atrás

Los servicios originales (`VideoProcessor`, `ImageProcessor`, `ColorAnalyzer`, `WorkflowManager`, `ParameterExporter`, etc.) siguen disponibles en los exports para mantener compatibilidad, pero se recomienda migrar a los nuevos sistemas unificados.

## Migración

### Procesamiento

```python
# Antes
from services.video_processor import VideoProcessor
from services.image_processor import ImageProcessor
from services.color_analyzer import ColorAnalyzer

video_proc = VideoProcessor()
image_proc = ImageProcessor()
color_analyzer = ColorAnalyzer()

# Después
from services.unified_processing_system import UnifiedProcessingSystem

processor = UnifiedProcessingSystem()
# Una sola interfaz para todo el procesamiento
```

### Workflow

```python
# Antes
from services.workflow_manager import WorkflowManager
from services.service_orchestrator import ServiceOrchestrator

workflow_mgr = WorkflowManager()
orchestrator = ServiceOrchestrator()

# Después
from services.unified_workflow_system import UnifiedWorkflowSystem

workflow_system = UnifiedWorkflowSystem()
# Auto-selecciona el mejor modo de ejecución
```

### Export

```python
# Antes
from services.parameter_exporter import ParameterExporter
from services.comparison_generator import ComparisonGenerator

exporter = ParameterExporter()
comparison = ComparisonGenerator()

# Después
from services.unified_export_system import UnifiedExportSystem

export_system = UnifiedExportSystem()
# Una sola interfaz para todas las exportaciones
```

## Beneficios

1. **Reducción de Duplicación**: Eliminación de código duplicado entre servicios relacionados
2. **Mejor Organización**: Servicios consolidados con responsabilidades claras
3. **Funcionalidad Mejorada**: Combinación de las mejores características de cada servicio
4. **Mantenibilidad**: Un solo lugar para mantener y actualizar funcionalidad relacionada
5. **Consistencia**: API unificada para operaciones similares
6. **Flexibilidad**: Modos configurables y auto-selección inteligente

## Estadísticas

- **Servicios consolidados**: 8 servicios → 3 sistemas unificados
- **Reducción de complejidad**: ~37% menos servicios para gestionar
- **Mejora de mantenibilidad**: Un solo punto de entrada para funcionalidad relacionada
- **Total de sistemas unificados**: 11 (incluyendo anteriores)

## Próximos Pasos

1. Migrar código existente que use los servicios antiguos
2. Actualizar documentación y ejemplos
3. Considerar deprecar los servicios antiguos en futuras versiones
4. Expandir funcionalidad de los sistemas unificados según necesidades


