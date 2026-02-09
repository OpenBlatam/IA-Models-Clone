# Arquitectura - Contabilidad Mexicana AI

## Visión General

El sistema está diseñado con una arquitectura modular que separa las responsabilidades en diferentes capas:

```
┌─────────────────────────────────────────┐
│         API Layer (FastAPI)            │
│      /api/contador/* endpoints         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Core Layer (ContadorAI)            │
│   - Lógica de negocio principal          │
│   - Orquestación de servicios            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  Infrastructure Layer (OpenRouter)      │
│   - Cliente HTTP con pooling            │
│   - Manejo de conexiones                │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Services Layer (Especializados)    │
│   - CalculadoraImpuestos                 │
│   - Otros servicios futuros              │
└─────────────────────────────────────────┘
```

## Componentes Principales

### 1. Core (`core/`)

**ContadorAI** (`contador_ai.py`): Clase principal que orquesta todas las operaciones.

- `calcular_impuestos()`: Cálculo de impuestos
- `asesoria_fiscal()`: Asesoría personalizada
- `guia_fiscal()`: Generación de guías
- `tramite_sat()`: Información de trámites
- `ayuda_declaracion()`: Ayuda con declaraciones

### 2. Infrastructure (`infrastructure/`)

**OpenRouterClient** (`openrouter/openrouter_client.py`): Cliente para OpenRouter API.

- Connection pooling
- Manejo de errores
- Timeouts configurables

### 3. Services (`services/`)

**CalculadoraImpuestos** (`calculadora_impuestos.py`): Cálculos especializados.

- Cálculo ISR RESICO
- Cálculo IVA
- Desglose por tramos

### 4. API (`api/`)

**FastAPI Router** (`contador_api.py`): Endpoints REST.

- Validación con Pydantic
- Manejo de errores
- Documentación automática

### 5. Config (`config/`)

**ContadorConfig** (`contador_config.py`): Configuración centralizada.

- Configuración OpenRouter
- Regímenes fiscales
- Tipos de impuestos
- Servicios disponibles

## Flujo de Datos

### Ejemplo: Cálculo de Impuestos

```
1. Cliente → POST /api/contador/calcular-impuestos
2. FastAPI → Valida request con Pydantic
3. ContadorAI → Recibe request
4. ContadorAI → Construye prompt especializado
5. OpenRouterClient → Llama a OpenRouter API
6. OpenRouter → Procesa con IA
7. OpenRouter → Retorna respuesta
8. ContadorAI → Procesa respuesta
9. FastAPI → Retorna JSON al cliente
```

## System Prompts

El sistema usa prompts especializados para cada tipo de servicio:

- **calculo_impuestos**: Enfocado en cálculos precisos con fórmulas
- **asesoria_fiscal**: Análisis y recomendaciones personalizadas
- **guias_fiscales**: Guías completas con ejemplos
- **tramites_sat**: Información detallada de procedimientos
- **declaraciones**: Ayuda paso a paso con declaraciones

## Extensibilidad

El sistema está diseñado para ser extensible:

1. **Nuevos Regímenes**: Agregar a `regimenes_fiscales` en config
2. **Nuevos Servicios**: Agregar método en `ContadorAI` y endpoint en API
3. **Nuevos Cálculos**: Agregar métodos en `CalculadoraImpuestos`
4. **Nuevos Prompts**: Agregar a `system_prompts` en `ContadorAI`

## Seguridad

- API keys mediante variables de entorno
- Validación de inputs con Pydantic
- Manejo seguro de errores
- No almacenamiento de datos sensibles

## Performance

- Connection pooling en HTTP client
- Caché opcional (futuro)
- Timeouts configurables
- Async/await para operaciones I/O
