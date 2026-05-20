# 🚀 Onyx AI Video System

**Sistema de Generación de Videos con IA completamente integrado con Onyx**

El sistema AI Video ha sido completamente adaptado para aprovechar la infraestructura, utilidades y capacidades existentes de Onyx, proporcionando una integración perfecta dentro del ecosistema Onyx mientras mantiene todas las características avanzadas del sistema de video.

## ✨ **Características Principales**

### 🔧 **Integración Completa con Onyx**
- **LLM Integration**: Utiliza el sistema LLM de Onyx para generación de texto y visión
- **Threading**: Aprovecha las utilidades de concurrencia de Onyx para procesamiento paralelo
- **Logging**: Integración con el sistema de logging estructurado de Onyx
- **Security**: Utiliza las utilidades de seguridad y validación de Onyx
- **Performance**: Integración con las utilidades de timing y GPU de Onyx
- **Telemetry**: Integración con el sistema de telemetría de Onyx

### 🎯 **Funcionalidades Avanzadas**
- **Generación de Video con IA**: Creación automática de videos a partir de texto
- **Procesamiento de Visión**: Análisis y generación con capacidades de visión
- **Sistema de Plugins**: Arquitectura modular y extensible
- **Workflow Inteligente**: Flujo de trabajo optimizado para generación de video
- **Monitoreo en Tiempo Real**: Métricas y telemetría completas
- **Escalabilidad**: Soporte para procesamiento distribuido

## 🏗️ **Arquitectura del Sistema**

```
┌─────────────────────────────────────────────────────────────┐
│                    Onyx AI Video System                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Onyx Main     │  │  Video Workflow │  │ Plugin Mgr   │ │
│  │   System        │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Onyx Integration│  │  LLM Manager    │  │ Task Manager │ │
│  │    Manager      │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Security Manager│  │ Performance Mgr │  │ File Manager │ │
│  │                 │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Onyx Infrastructure                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   LLM Factory   │  │ Threading Utils │  │ Logging Sys  │ │
│  │                 │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 **Instalación Rápida**

### **1. Instalación Automática**
```bash
# Clonar el repositorio
git clone <repository-url>
cd blatam-academy/agents/backend/onyx/server/features/ai_video

# Instalación automática
python install_onyx.py --install-dir /opt/onyx-ai-video

# Iniciar el sistema
python start_onyx_ai_video.py start
```

### **2. Instalación Manual**
```bash
# Instalar dependencias
pip install -r requirements_unified.txt

# Configurar el sistema
python onyx_config.py

# Inicializar el sistema
python onyx_main.py
```

### **3. Verificar Instalación**
```bash
# Verificar estado del sistema
python start_onyx_ai_video.py status

# Ejecutar pruebas
python start_onyx_ai_video.py test

# Verificar salud del sistema
python start_onyx_ai_video.py health
```

## 📖 **Uso Básico**

### **Generación de Video Simple**
```python
from onyx_ai_video import get_system
from models import VideoRequest

# Obtener sistema
system = await get_system()

# Crear solicitud
request = VideoRequest(
    input_text="Crea un video sobre inteligencia artificial",
    user_id="usuario123"
)

# Generar video
response = await system.generate_video(request)
print(f"Video generado: {response.output_url}")
```

### **Generación con Plugins**
```python
# Generar video con plugins específicos
request = VideoRequest(
    input_text="Crea un video sobre tecnología",
    user_id="usuario123",
    plugins=["content_analyzer", "visual_enhancer"]
)

response = await system.generate_video(request)
print(f"Resultados de plugins: {response.metadata['plugin_results']}")
```

### **Generación con Visión**
```python
# Generar video con capacidades de visión
with open("imagen.jpg", "rb") as f:
    image_data = f.read()

response = await system.generate_video_with_vision(request, image_data)
print(f"Video con visión generado: {response.output_url}")
```

## 🔧 **Configuración**

### **Archivo de Configuración Principal**
```json
{
  "use_onyx_logging": true,
  "use_onyx_llm": true,
  "use_onyx_telemetry": true,
  "use_onyx_encryption": true,
  "use_onyx_threading": true,
  "use_onyx_retry": true,
  "use_onyx_gpu": true,
  "max_workers": 10,
  "timeout_seconds": 300,
  "default_quality": "medium",
  "default_duration": 60,
  "enable_plugins": true
}
```

### **Variables de Entorno**
```bash
# Configuración del sistema
export AI_VIDEO_ENVIRONMENT=production
export AI_VIDEO_LOG_LEVEL=INFO
export AI_VIDEO_DEBUG_MODE=false

# Configuración de rendimiento
export AI_VIDEO_MAX_WORKERS=10
export AI_VIDEO_TIMEOUT=300
export AI_VIDEO_RETRY_ATTEMPTS=3

# Configuración de video
export AI_VIDEO_DEFAULT_QUALITY=medium
export AI_VIDEO_DEFAULT_DURATION=60
export AI_VIDEO_DEFAULT_OUTPUT_FORMAT=mp4

# Integración con Onyx
export AI_VIDEO_USE_ONYX_LOGGING=true
export AI_VIDEO_USE_ONYX_LLM=true
export AI_VIDEO_USE_ONYX_TELEMETRY=true
export AI_VIDEO_USE_ONYX_GPU=true
```

## 🔌 **Sistema de Plugins**

### **Plugins Incluidos**
- **Content Analyzer**: Análisis de contenido usando LLM de Onyx
- **Visual Enhancer**: Mejora visual con aceleración GPU
- **Audio Processor**: Procesamiento y mejora de audio

### **Crear Plugin Personalizado**
```python
from onyx_plugin_manager import OnyxPluginBase, OnyxPluginContext

class MiPluginPersonalizado(OnyxPluginBase):
    version = "1.0.0"
    description = "Mi plugin personalizado"
    author = "Tu Nombre"
    category = "personalizado"
    
    async def _initialize_plugin(self):
        # Inicializar recursos del plugin
        pass
    
    async def process(self, context: OnyxPluginContext):
        # Lógica del plugin
        return {"resultado": "procesado"}
```

## 📊 **Monitoreo y Métricas**

### **Estado del Sistema**
```bash
# Ver estado completo
python start_onyx_ai_video.py status

# Ver métricas
python start_onyx_ai_video.py metrics

# Verificar salud
python start_onyx_ai_video.py health
```

### **Métricas Disponibles**
- **Rendimiento**: Tiempo de procesamiento, tasa de éxito
- **Recursos**: Uso de GPU, memoria, CPU
- **Plugins**: Estado, rendimiento, errores
- **Sistema**: Uptime, requests, errores

## 🛠️ **Comandos CLI**

### **Gestión del Sistema**
```bash
# Iniciar sistema
python start_onyx_ai_video.py start

# Ver estado
python start_onyx_ai_video.py status

# Ver métricas
python start_onyx_ai_video.py metrics

# Verificar salud
python start_onyx_ai_video.py health
```

### **Gestión de Configuración**
```bash
# Mostrar configuración
python start_onyx_ai_video.py config show

# Validar configuración
python start_onyx_ai_video.py config validate

# Crear configuración por defecto
python start_onyx_ai_video.py config create

# Guardar configuración
python start_onyx_ai_video.py config save
```

### **Generación de Video**
```bash
# Generar video simple
python start_onyx_ai_video.py generate "Texto para el video"

# Generar con opciones
python start_onyx_ai_video.py generate "Texto" --quality high --duration 120

# Generar con plugins
python start_onyx_ai_video.py generate "Texto" --plugins "content_analyzer,visual_enhancer"
```

### **Gestión de Plugins**
```bash
# Listar plugins
python start_onyx_ai_video.py plugins list

# Ver estado de plugins
python start_onyx_ai_video.py plugins status

# Habilitar plugin
python start_onyx_ai_video.py plugins enable content_analyzer

# Deshabilitar plugin
python start_onyx_ai_video.py plugins disable content_analyzer
```

### **Pruebas**
```bash
# Ejecutar pruebas básicas
python start_onyx_ai_video.py test

# Ejecutar pruebas con generación
python start_onyx_ai_video.py test --test-generation
```

## 🔍 **Troubleshooting**

### **Problemas Comunes**

#### **1. Error de Inicialización**
```bash
# Verificar logs
tail -f logs/onyx_ai_video.log

# Verificar configuración
python start_onyx_ai_video.py config validate

# Verificar estado del sistema
python start_onyx_ai_video.py health
```

#### **2. Error de GPU**
```bash
# Verificar disponibilidad de GPU
python -c "from onyx.utils.gpu_utils import is_gpu_available; print(is_gpu_available())"

# Deshabilitar GPU si es necesario
export AI_VIDEO_USE_ONYX_GPU=false
```

#### **3. Error de Plugins**
```bash
# Verificar plugins
python start_onyx_ai_video.py plugins status

# Reiniciar plugins
python start_onyx_ai_video.py plugins reload
```

#### **4. Problemas de Rendimiento**
```bash
# Ver métricas de rendimiento
python start_onyx_ai_video.py metrics

# Ajustar configuración
export AI_VIDEO_MAX_WORKERS=5
export AI_VIDEO_TIMEOUT=600
```

## 📚 **Documentación Adicional**

### **Guías Específicas**
- **[Guía de Integración Onyx](ONYX_INTEGRATION_GUIDE.md)**: Integración completa con Onyx
- **[Guía de Producción](PRODUCTION_GUIDE.md)**: Despliegue en producción
- **[Guía de Plugins](README.md)**: Desarrollo de plugins
- **[Vista General del Sistema](SYSTEM_OVERVIEW.md)**: Arquitectura del sistema

### **Ejemplos de Código**
- **Ejemplos básicos**: `examples/basic_usage.py`
- **Ejemplos avanzados**: `examples/advanced_usage.py`
- **Ejemplos de plugins**: `examples/plugin_examples.py`

## 🤝 **Contribución**

### **Desarrollo Local**
```bash
# Clonar repositorio
git clone <repository-url>
cd blatam-academy/agents/backend/onyx/server/features/ai_video

# Instalar en modo desarrollo
python install_onyx.py --development --debug

# Ejecutar pruebas
python start_onyx_ai_video.py test --test-generation
```

### **Crear Plugin**
1. Crear archivo en `plugins/` o `custom_plugins/`
2. Heredar de `OnyxPluginBase`
3. Implementar métodos requeridos
4. Probar con `python start_onyx_ai_video.py plugins list`

## 📄 **Licencia**

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 **Soporte**

### **Canales de Soporte**
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentación**: [Documentación Completa](https://docs.onyx-ai-video.com)
- **Comunidad**: [Discord/Slack](https://community.onyx-ai-video.com)

### **Contacto**
- **Email**: support@onyx-ai-video.com
- **Twitter**: [@OnyxAIVideo](https://twitter.com/OnyxAIVideo)

---

**¡Gracias por usar Onyx AI Video System! 🎉**

Este sistema representa la integración perfecta entre la potencia de Onyx y las capacidades avanzadas de generación de video con IA, proporcionando una solución empresarial completa y escalable. 