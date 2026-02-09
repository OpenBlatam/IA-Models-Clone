# Contabilidad Mexicana AI SAM3

Sistema de contabilidad mexicana con arquitectura SAM3, integrado con OpenRouter y TruthGPT.

## Características

- ✅ Arquitectura SAM3 para procesamiento paralelo y continuo
- ✅ Integración con OpenRouter para LLM de alta calidad
- ✅ Integración con TruthGPT para optimización avanzada
- ✅ Operación continua 24/7
- ✅ Ejecución paralela de tareas
- ✅ Gestión automática de tareas con cola de prioridades
- ✅ Servicios de contabilidad mexicana:
  - Cálculo de impuestos
  - Asesoría fiscal
  - Guías fiscales
  - Trámites SAT
  - Ayuda con declaraciones

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Configura las variables de entorno:

```bash
export OPENROUTER_API_KEY="tu-api-key"
export TRUTHGPT_ENDPOINT="opcional-endpoint"  # Opcional
```

## Uso Básico

```python
import asyncio
from contabilidad_mexicana_ai_sam3 import ContadorSAM3Agent, ContadorSAM3Config

async def main():
    # Crear configuración
    config = ContadorSAM3Config()
    
    # Crear agente
    agent = ContadorSAM3Agent(config=config)
    
    # Iniciar agente (modo 24/7)
    # await agent.start()  # En producción
    
    # O usar métodos directos
    task_id = await agent.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={"ingresos": 100000, "gastos": 30000}
    )
    
    # Esperar resultado
    import time
    while True:
        status = await agent.get_task_status(task_id)
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            print(result)
            break
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Arquitectura

Para una documentación completa de la arquitectura, consulta:
- **[COMPLETE_ARCHITECTURE.md](docs/COMPLETE_ARCHITECTURE.md)** - Documentación completa y detallada
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Resumen de arquitectura

### Estructura de Directorios

```
contabilidad_mexicana_ai_sam3/
├── core/
│   ├── contador_sam3_agent.py    # Agente principal (orchestrator)
│   ├── task_manager.py            # Gestión de tareas y cola de prioridades
│   ├── parallel_executor.py       # Ejecución paralela con worker pool
│   ├── prompt_builder.py          # Construcción de prompts de usuario
│   ├── system_prompts_builder.py # Prompts del sistema especializados
│   └── helpers.py                 # Utilidades comunes
├── infrastructure/
│   ├── openrouter_client.py       # Cliente OpenRouter (LLM)
│   ├── truthgpt_client.py         # Cliente TruthGPT (optimización)
│   └── retry_helpers.py           # Helpers de reintentos con backoff
├── config/
│   └── contador_sam3_config.py    # Configuración centralizada
├── api/
│   └── contador_sam3_api.py       # API REST (FastAPI, opcional)
├── utils/
│   ├── formatters.py              # Formateo de datos fiscales
│   └── validators.py              # Validación de entrada
├── tests/
│   └── test_contador_sam3_agent.py # Suite de tests
├── examples/
│   ├── example_usage.py           # Ejemplos básicos
│   └── advanced_examples.py        # Ejemplos avanzados
└── docs/
    ├── COMPLETE_ARCHITECTURE.md   # Arquitectura completa
    ├── ARCHITECTURE.md            # Resumen de arquitectura
    └── API.md                     # Documentación de API
```

## Servicios Disponibles

### 1. Cálculo de Impuestos

```python
task_id = await agent.calcular_impuestos(
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={
        "ingresos": 100000,
        "gastos": 30000,
        "periodo": "2024-01"
    }
)
```

### 2. Asesoría Fiscal

```python
task_id = await agent.asesoria_fiscal(
    pregunta="¿Puedo deducir gastos de home office?",
    contexto={"regimen": "RESICO", "ingresos_anuales": 500000}
)
```

### 3. Guía Fiscal

```python
task_id = await agent.guia_fiscal(
    tema="Deducciones RESICO",
    nivel_detalle="completo"
)
```

### 4. Trámites SAT

```python
task_id = await agent.tramite_sat(
    tipo_tramite="Alta en RFC",
    detalles={"persona_fisica": True}
)
```

### 5. Ayuda con Declaraciones

```python
task_id = await agent.ayuda_declaracion(
    tipo_declaracion="mensual",
    periodo="2024-01",
    datos={"rfc": "ABC123456789"}
)
```

## Características Avanzadas

### Modo 24/7 Continuo

El agente puede ejecutarse en modo continuo procesando tareas automáticamente:

```python
agent = ContadorSAM3Agent(config=config)
await agent.start()  # Ejecuta indefinidamente
```

### Priorización de Tareas

Las tareas pueden tener diferentes prioridades:

```python
# Alta prioridad
task_id = await agent.calcular_impuestos(
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={...},
    priority=10  # Mayor prioridad
)
```

### Integración con TruthGPT

El agente optimiza automáticamente las consultas usando TruthGPT cuando está disponible.

## Requisitos

- Python 3.8+
- OpenRouter API key
- TruthGPT (opcional pero recomendado)

## Licencia

MIT
