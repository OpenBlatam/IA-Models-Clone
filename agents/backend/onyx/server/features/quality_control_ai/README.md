# Quality Control AI

Sistema de Control de Calidad con Detección de Defectos por Cámara

## Descripción

Sistema completo de control de calidad que utiliza una cámara para detectar objetos, identificar anomalías y clasificar defectos en productos. El sistema integra múltiples técnicas de visión por computadora y machine learning para inspección automática de calidad.

## Características

- **Control de Cámara**: Gestión completa de cámaras con configuración flexible
- **Detección de Objetos**: Detección de objetos usando modelos de deep learning (YOLOv8, Faster R-CNN, SSD)
- **Detección de Anomalías**: Múltiples métodos de detección de anomalías (estadístico, autoencoder, bordes, color)
- **Clasificación de Defectos**: Clasificación automática de defectos en múltiples categorías
- **Análisis de Calidad**: Análisis detallado y scoring de calidad
- **Inspección en Tiempo Real**: Inspección continua desde cámara en vivo
- **Análisis de Video**: Análisis completo de archivos de video y streaming
- **Visualización Avanzada**: Visualización de resultados con gráficos y estadísticas
- **Sistema de Alertas**: Alertas automáticas basadas en umbrales de calidad
- **Generación de Reportes**: Reportes en JSON, CSV y HTML
- **Optimización de Rendimiento**: Caching y optimizaciones para mejor rendimiento

## Estructura

```
quality_control_ai/
├── __init__.py
├── core/
│   ├── camera_controller.py      # Controlador de cámara
│   ├── object_detector.py        # Detector de objetos
│   ├── anomaly_detector.py       # Detector de anomalías
│   └── defect_classifier.py     # Clasificador de defectos
├── config/
│   ├── camera_config.py          # Configuración de cámara
│   └── detection_config.py       # Configuración de detección
├── services/
│   ├── quality_inspector.py     # Inspector principal de calidad
│   └── defect_analyzer.py        # Analizador de defectos
└── utils/
    ├── image_utils.py           # Utilidades de imagen
    └── detection_utils.py        # Utilidades de detección
```

## Uso Básico

### Inicializar Inspector

```python
from quality_control_ai import QualityInspector, CameraConfig, DetectionConfig

# Configurar cámara
camera_config = CameraConfig()
camera_config.update_settings(
    camera_index=0,
    resolution_width=1920,
    resolution_height=1080,
    fps=30
)

# Configurar detección
detection_config = DetectionConfig()
detection_config.update_settings(
    confidence_threshold=0.5,
    anomaly_threshold=0.7
)

# Crear inspector
inspector = QualityInspector(camera_config, detection_config)
```

### Inspección desde Cámara

```python
# Inicializar cámara
inspector.start_inspection()

# Inspeccionar frame
result = inspector.inspect_frame()

print(f"Calidad: {result['quality_score']}/100")
print(f"Defectos detectados: {result['defects_detected']}")
print(f"Estado: {result['summary']['status']}")

# Detener inspección
inspector.stop_inspection()
inspector.release()
```

### Inspección de Imagen

```python
# Inspeccionar imagen desde archivo
result = inspector.inspect_frame("path/to/image.jpg")

# Ver resultados
for defect in result['defects']:
    print(f"Tipo: {defect['type']}, Severidad: {defect['severity']}")
```

### Análisis Detallado de Defectos

```python
from quality_control_ai import DefectAnalyzer

analyzer = DefectAnalyzer()

# Analizar defectos
analysis = analyzer.analyze_defects(
    defects=result['defects'],
    image_size=(1920, 1080)
)

print(f"Análisis por tipo: {analysis['type_analysis']}")
print(f"Recomendaciones: {analysis['recommendations']}")
```

## Tipos de Defectos Detectados

El sistema puede detectar y clasificar los siguientes tipos de defectos:

- **Scratch** (Rayón): Líneas delgadas y largas
- **Crack** (Grieta): Líneas más anchas y profundas
- **Dent** (Abolladura): Deformaciones circulares/elípticas
- **Discoloration** (Decoloración): Cambios anómalos de color
- **Deformation** (Deformación): Cambios en la forma
- **Missing Part** (Parte Faltante): Componentes ausentes
- **Surface Imperfection** (Imperfección de Superficie): Irregularidades superficiales
- **Contamination** (Contaminación): Material extraño
- **Size Variation** (Variación de Tamaño): Desviaciones dimensionales
- **Other** (Otro): Defectos no categorizados

## Configuración

### Configuración de Cámara

```python
camera_config = CameraConfig()
camera_config.update_settings(
    camera_index=0,              # Índice de cámara
    resolution_width=1920,       # Ancho de resolución
    resolution_height=1080,       # Alto de resolución
    fps=30,                       # Frames por segundo
    brightness=0.5,               # Brillo (0-1)
    contrast=0.5,                 # Contraste (0-1)
    saturation=0.5,               # Saturación (0-1)
    exposure=0.5,                 # Exposición (0-1)
    auto_focus=True,              # Autoenfoque
    white_balance="auto"          # Balance de blancos
)
```

### Configuración de Detección

```python
detection_config = DetectionConfig()
detection_config.update_settings(
    confidence_threshold=0.5,     # Umbral de confianza
    nms_threshold=0.4,            # Umbral de NMS
    anomaly_threshold=0.7,         # Umbral de anomalía
    use_autoencoder=True,         # Usar autoencoder
    use_statistical=True,          # Usar detección estadística
    object_detection_model="yolov8",  # Modelo de detección
    min_defect_size=10,           # Tamaño mínimo de defecto
    max_defect_size=10000         # Tamaño máximo de defecto
)
```

## Métodos de Detección de Anomalías

El sistema utiliza múltiples métodos para detectar anomalías:

1. **Detección Estadística**: Usa análisis estadístico (Z-score) para identificar outliers
2. **Autoencoder**: Modelo de autoencoder para reconstrucción y detección de anomalías
3. **Detección por Bordes**: Análisis de contornos irregulares
4. **Detección por Color**: Identificación de cambios anómalos de color

## Scoring de Calidad

El sistema calcula un score de calidad (0-100) basado en:

- Número y severidad de anomalías detectadas
- Número y severidad de defectos clasificados
- Cobertura de área afectada

**Interpretación del Score:**
- 90-100: Excelente
- 75-89: Bueno
- 60-74: Aceptable
- 40-59: Pobre (requiere revisión)
- 0-39: Rechazado

## Dependencias

- OpenCV (`cv2`)
- NumPy
- PIL/Pillow
- scikit-learn (para análisis estadístico)
- ultralytics (opcional, para YOLOv8)

## Ejemplo Completo

```python
from quality_control_ai import (
    QualityInspector, 
    CameraConfig, 
    DetectionConfig,
    VideoAnalyzer,
    AlertSystem,
    QualityVisualizer,
    ReportGenerator
)

# Configurar
camera_config = CameraConfig()
detection_config = DetectionConfig()

# Crear inspector
inspector = QualityInspector(camera_config, detection_config)

# Iniciar inspección
if inspector.start_inspection():
    # Inspeccionar
    result = inspector.inspect_frame()
    
    # Mostrar resultados
    print(f"Score de Calidad: {result['quality_score']}")
    print(f"Estado: {result['summary']['status']}")
    print(f"Recomendación: {result['summary']['recommendation']}")
    
    # Visualizar resultados
    visualizer = QualityVisualizer()
    vis_image = visualizer.visualize_inspection(
        image=result.get('image'),
        objects=result.get('objects'),
        defects=result.get('defects'),
        quality_score=result.get('quality_score')
    )
    cv2.imwrite('inspection_result.jpg', vis_image)
    
    # Generar reporte
    report_gen = ReportGenerator()
    report_gen.generate_html_report(result, 'report.html')
    
    # Sistema de alertas
    alert_system = AlertSystem()
    alerts = alert_system.check_inspection_result(result)
    for alert in alerts:
        print(f"ALERTA [{alert.level.value}]: {alert.message}")
    
    # Detener
    inspector.stop_inspection()
    inspector.release()
```

## Análisis de Video

```python
from quality_control_ai import VideoAnalyzer, QualityInspector

# Crear analizador de video
inspector = QualityInspector()
video_analyzer = VideoAnalyzer(inspector)

# Analizar archivo de video
result = video_analyzer.analyze_video_file(
    video_path="production_line.mp4",
    frame_skip=5,  # Analizar cada 5 frames
    max_frames=1000
)

print(f"Calidad promedio: {result['average_quality_score']}")
print(f"Estado: {result['status']}")

# Análisis de streaming en tiempo real
for result in video_analyzer.analyze_stream(duration=60):
    if result.get('quality_score', 100) < 60:
        print(f"Alerta: Calidad baja en frame {result['frame_number']}")
```

## Sistema de Alertas

```python
from quality_control_ai import AlertSystem, AlertLevel

alert_system = AlertSystem()

# Configurar umbrales
alert_system.set_threshold("quality_score_low", 40.0)
alert_system.set_threshold("defect_count_critical", 5)

# Callback personalizado
def on_alert(alert):
    if alert.level == AlertLevel.CRITICAL:
        send_email_notification(alert.message)
    print(f"[{alert.level.value}] {alert.message}")

alert_system.register_callback(on_alert)

# Verificar resultado
alerts = alert_system.check_inspection_result(inspection_result)

# Obtener estadísticas
stats = alert_system.get_alert_statistics()
print(f"Alertas críticas recientes: {stats['recent_critical']}")
```

## Visualización Avanzada

```python
from quality_control_ai import QualityVisualizer

visualizer = QualityVisualizer()

# Visualizar inspección
vis_image = visualizer.visualize_inspection(
    image=image,
    objects=objects,
    anomalies=anomalies,
    defects=defects,
    quality_score=85.5,
    show_labels=True
)

# Crear imagen resumen
summary_image = visualizer.create_summary_image(
    image=image,
    inspection_result=result,
    save_path="summary.jpg"
)

# Generar gráficos de estadísticas
plot_bytes = visualizer.create_statistics_plot(
    inspection_result=result,
    save_path="statistics.png"
)
```

## Generación de Reportes

```python
from quality_control_ai import ReportGenerator

report_gen = ReportGenerator()

# Reporte JSON
report_gen.generate_json_report(
    inspection_result=result,
    output_path="report.json",
    include_images=False
)

# Reporte CSV (múltiples inspecciones)
report_gen.generate_csv_report(
    inspection_results=[result1, result2, result3],
    output_path="batch_report.csv"
)

# Reporte HTML
report_gen.generate_html_report(
    inspection_result=result,
    output_path="report.html",
    include_charts=True
)
```

## Optimización de Rendimiento

```python
from quality_control_ai import PerformanceOptimizer, measure_time

optimizer = PerformanceOptimizer(cache_size=200)

# Usar decorador de caché
@optimizer.cached(ttl=300)  # Cache por 5 minutos
def expensive_detection(image):
    # Operación costosa
    return detect_objects(image)

# Medir tiempo de ejecución
@measure_time
def process_image(image):
    return inspector.inspect_frame(image)

# Obtener estadísticas
stats = optimizer.get_performance_stats("expensive_detection")
print(f"Tiempo promedio: {stats['avg_time']:.4f}s")
```

## Notas

- El sistema requiere una cámara conectada o imágenes para procesar
- Los modelos de ML (YOLOv8, autoencoder) son opcionales pero mejoran la precisión
- La configuración puede ajustarse según el tipo de producto a inspeccionar
- El sistema es extensible y permite agregar nuevos tipos de defectos

## Autor

Blatam Academy

