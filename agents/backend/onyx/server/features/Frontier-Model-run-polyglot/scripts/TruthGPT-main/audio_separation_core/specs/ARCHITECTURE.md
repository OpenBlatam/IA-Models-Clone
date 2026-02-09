# Arquitectura de Audio Separation Core

## 📋 Resumen

`audio_separation_core` es un framework modular para separación y mezcla de audio con IA, basado en la arquitectura de `optimization_core`.

## 🎯 Objetivos

1. **Separación de Audio**: Separar audio en componentes (voces, música, efectos)
2. **Mezcla de Audio**: Mezclar componentes con control de volúmenes y efectos
3. **Procesamiento de Video**: Extraer audio de videos
4. **Modularidad**: Arquitectura basada en interfaces para fácil extensión
5. **Alto Rendimiento**: Optimizado para procesamiento eficiente

## 🏛️ Arquitectura General

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                          │
│              (Funciones de conveniencia, APIs)                 │
└────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│                    Core Layer                                    │
│              (Interfaces, Config, Factories)                     │
└─────────────────────────────┬──────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐    ┌─────────▼─────────┐   ┌──────▼──────┐
│ Separators   │    │     Mixers        │   │ Processors  │
│              │    │                   │   │             │
├──────────────┤    ├───────────────────┤   ├─────────────┤
│ • Spleeter   │    │ • Simple Mixer    │   │ • Video     │
│ • Demucs     │    │ • Advanced Mixer │   │   Extractor │
│ • LALAL      │    │                   │   │ • Converter │
└──────────────┘    └───────────────────┘   └─────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│                    Utils Layer                                   │
│              (Audio Utils, Format Utils, Validation)            │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Componentes Principales

### 1. Core Layer (`core/`)

**Interfaces** (`interfaces.py`):
- `IAudioComponent`: Interfaz base para todos los componentes
- `IAudioSeparator`: Interfaz para separadores de audio
- `IAudioMixer`: Interfaz para mezcladores de audio
- `IAudioProcessor`: Interfaz para procesadores de audio

**Base Component** (`base_component.py`):
- `BaseComponent`: Clase base con gestión de ciclo de vida
  - **Responsabilidades**: Inicialización, limpieza, estado, salud
  - **Beneficios**: Elimina duplicación de código en BaseSeparator y BaseMixer
  - **Métodos principales**:
    - `initialize()`: Inicializa el componente
    - `cleanup()`: Limpia recursos
    - `get_status()`: Obtiene estado y salud
    - `_do_initialize()`: Método abstracto para inicialización específica
    - `_do_cleanup()`: Método opcional para limpieza específica

**Configuración** (`config.py`):
- `AudioConfig`: Configuración base de audio
- `SeparationConfig`: Configuración para separación (hereda de `AudioConfig`)
- `MixingConfig`: Configuración para mezcla (hereda de `AudioConfig`)
- `ProcessorConfig`: Configuración para procesamiento (hereda de `AudioConfig`)

**Factories** (`factories.py`):
- `BaseFactory`: Factory genérico con lógica común
  - **Responsabilidades**: Registro, creación dinámica, validación de tipos
  - **Beneficios**: Elimina ~200 líneas de código duplicado
- `AudioSeparatorFactory`: Factory para separadores (hereda de `BaseFactory`)
- `AudioMixerFactory`: Factory para mezcladores (hereda de `BaseFactory`)
- `AudioProcessorFactory`: Factory para procesadores (hereda de `BaseFactory`)

**Excepciones** (`exceptions.py`):
- Jerarquía de excepciones específicas del dominio:
  - `AudioSeparationError`: Excepción base
  - `AudioProcessingError`: Errores de procesamiento
  - `AudioFormatError`: Errores de formato
  - `AudioModelError`: Errores de modelos
  - `AudioValidationError`: Errores de validación
  - `AudioIOError`: Errores de I/O
  - `AudioConfigurationError`: Errores de configuración

### 2. Separators (`separators/`)

**BaseSeparator**: Clase base abstracta para separadores
- **Refactorizado**: Ahora hereda de `BaseComponent` (elimina duplicación)
- **Responsabilidades**:
  - Validación de formatos y componentes
  - Lógica específica de separación
  - Gestión de modelos de IA
  - Delegación de ciclo de vida a `BaseComponent`
- **Métodos abstractos**:
  - `_load_model()`: Carga el modelo de separación
  - `_cleanup_model()`: Limpia el modelo
  - `_perform_separation()`: Realiza la separación
  - `_get_supported_components()`: Obtiene componentes soportados

**Implementaciones**:
- `SpleeterSeparator`: Usa Spleeter de Deezer
- `DemucsSeparator`: Usa Demucs de Facebook Research
- `LALALSeparator`: Usa LALAL.AI API

### 3. Mixers (`mixers/`)

**BaseMixer**: Clase base abstracta para mezcladores
- **Refactorizado**: Ahora hereda de `BaseComponent` (elimina duplicación)
- **Responsabilidades**:
  - Validación de archivos y volúmenes
  - Lógica específica de mezcla
  - Aplicación de efectos
  - Delegación de ciclo de vida a `BaseComponent`
- **Métodos abstractos**:
  - `_perform_mixing()`: Realiza la mezcla
  - `_apply_effect()`: Aplica efectos a archivos

**Implementaciones**:
- `SimpleMixer`: Mezcla básica con control de volúmenes
- `AdvancedMixer`: Mezcla avanzada con efectos (reverb, EQ, compresor)

### 4. Processors (`processors/`)

**Implementaciones**:
- `VideoAudioExtractor`: Extrae audio de videos usando ffmpeg
- `AudioFormatConverter`: Convierte entre formatos de audio
- `AudioEnhancer`: Mejora calidad de audio

### 5. Utils (`utils/`)

- `audio_utils.py`: Utilidades para procesamiento de audio
- `format_utils.py`: Utilidades para formatos
- `validation_utils.py`: Utilidades de validación

## 🔄 Flujos de Trabajo

### Flujo de Separación

```
1. Usuario → create_audio_separator()
2. Factory → crea instancia del separador
3. Separator → carga modelo de IA
4. Separator → separa audio en componentes
5. Separator → retorna rutas a componentes separados
```

### Flujo de Mezcla

```
1. Usuario → create_audio_mixer()
2. Factory → crea instancia del mezclador
3. Mixer → carga archivos de audio
4. Mixer → aplica volúmenes y efectos
5. Mixer → mezcla componentes
6. Mixer → retorna ruta al archivo mezclado
```

### Flujo Completo (Video → Separación → Mezcla)

```
1. Usuario → process_video_audio()
2. VideoExtractor → extrae audio del video
3. Separator → separa audio en componentes
4. Mixer → mezcla componentes con volúmenes personalizados
5. Sistema → retorna rutas a todos los archivos generados
```

## 🎨 Principios de Diseño

1. **Interfaces Primero**: Todas las funcionalidades se definen como interfaces
2. **Factory Pattern**: Creación de instancias mediante factories (consolidado en `BaseFactory`)
3. **Configuración Centralizada**: Todas las configuraciones en clases dedicadas
4. **DRY (Don't Repeat Yourself)**: Eliminada duplicación mediante `BaseComponent` y `BaseFactory`
5. **Single Responsibility Principle**: Cada clase tiene una responsabilidad única y clara
6. **Extensibilidad**: Fácil agregar nuevos separadores, mezcladores o procesadores
7. **Separación de Concerns**: Cada componente tiene una responsabilidad clara

## 🔄 Refactorización y Mejoras

### Cambios Principales

#### Antes de la Refactorización
- `BaseSeparator` y `BaseMixer` duplicaban código de ciclo de vida (~50 líneas cada uno)
- Tres factories con código casi idéntico (~200 líneas duplicadas)
- Validación redundante en configuraciones

#### Después de la Refactorización
- `BaseSeparator` y `BaseMixer` heredan de `BaseComponent` (eliminadas ~100 líneas)
- Factories heredan de `BaseFactory` (eliminadas ~200 líneas)
- **Total**: ~300 líneas de código duplicado eliminadas

### Estructura de Herencia Refactorizada

```
IAudioComponent (interface)
    └── BaseComponent (implementación base)
        ├── BaseSeparator (hereda de BaseComponent, implementa IAudioSeparator)
        │   ├── SpleeterSeparator
        │   ├── DemucsSeparator
        │   └── LALALSeparator
        └── BaseMixer (hereda de BaseComponent, implementa IAudioMixer)
            ├── SimpleMixer
            └── AdvancedMixer

BaseFactory (factory genérico)
    ├── AudioSeparatorFactory
    ├── AudioMixerFactory
    └── AudioProcessorFactory
```

### Beneficios de la Refactorización

1. **Mantenibilidad**: Cambios en ciclo de vida solo requieren modificar `BaseComponent`
2. **Consistencia**: Todos los componentes siguen el mismo patrón de ciclo de vida
3. **Testabilidad**: `BaseComponent` y `BaseFactory` pueden probarse independientemente
4. **Extensibilidad**: Agregar nuevos componentes es más simple (heredar de `BaseComponent`)
5. **Legibilidad**: Menos código, intención más clara

## 🔌 Extensibilidad

### Agregar un Nuevo Separador

1. Crear clase que herede de `BaseSeparator`
2. Implementar métodos abstractos:
   - `_load_model()`: Cargar el modelo de IA
   - `_cleanup_model()`: Limpiar recursos del modelo
   - `_perform_separation()`: Lógica de separación
   - `_get_supported_components()`: Lista de componentes soportados
3. Registrar en `AudioSeparatorFactory` (o usar auto-detección)

**Ejemplo**:
```python
class MySeparator(BaseSeparator):
    def _load_model(self, **kwargs):
        # Cargar modelo específico
        pass
    
    def _perform_separation(self, input_path, output_dir, components, **kwargs):
        # Implementar separación
        pass
    
    # ... otros métodos abstractos
```

### Agregar un Nuevo Mezclador

1. Crear clase que herede de `BaseMixer`
2. Implementar `_perform_mixing()` (y opcionalmente `_apply_effect()`)
3. Registrar en `AudioMixerFactory` (o usar auto-detección)

**Ejemplo**:
```python
class MyMixer(BaseMixer):
    def _perform_mixing(self, audio_files, output_path, volumes, effects, **kwargs):
        # Implementar mezcla
        pass
```

### Agregar un Nuevo Procesador

1. Crear clase que implemente `IAudioProcessor` (o herede de `BaseComponent` si necesita ciclo de vida)
2. Implementar métodos requeridos:
   - `process()`: Procesamiento principal
   - `get_metadata()`: Obtener metadatos
   - `validate()`: Validar entrada
3. Registrar en `AudioProcessorFactory`

## 📊 Rendimiento

- **Separación**: Depende del modelo (Demucs > Spleeter > LALAL)
- **Mezcla**: Procesamiento en memoria, muy rápido
- **Extracción de Video**: Depende de ffmpeg y tamaño del video

## 🔒 Manejo de Errores

- Excepciones específicas del dominio
- Validación de entrada
- Manejo de recursos (cleanup automático)
- Mensajes de error descriptivos




