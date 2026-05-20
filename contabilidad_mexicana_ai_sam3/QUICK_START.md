# Quick Start Guide

## Instalación Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar variables de entorno
export OPENROUTER_API_KEY="tu-api-key"
export TRUTHGPT_ENDPOINT="opcional"  # Opcional
```

## Uso Básico en 3 Pasos

### Paso 1: Importar y Configurar

```python
from contabilidad_mexicana_ai_sam3 import ContadorSAM3Agent, ContadorSAM3Config

config = ContadorSAM3Config()
agent = ContadorSAM3Agent(config=config)
```

### Paso 2: Enviar Tarea

```python
task_id = await agent.calcular_impuestos(
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={"ingresos": 100000, "gastos": 30000}
)
```

### Paso 3: Obtener Resultado

```python
import asyncio

while True:
    status = await agent.get_task_status(task_id)
    if status["status"] == "completed":
        result = await agent.get_task_result(task_id)
        print(result)
        break
    await asyncio.sleep(1)
```

## Ejemplos Completos

Ver `examples/example_usage.py` y `examples/advanced_examples.py` para más ejemplos.

## API REST

```bash
# Iniciar servidor
uvicorn contabilidad_mexicana_ai_sam3.api.contador_sam3_api:app --reload

# Usar endpoints
curl -X POST http://localhost:8000/calcular-impuestos \
  -H "Content-Type: application/json" \
  -d '{"regimen": "RESICO", "tipo_impuesto": "ISR", "datos": {"ingresos": 100000}}'
```

## Modo Continuo 24/7

```python
agent = ContadorSAM3Agent(config=config)
await agent.start()  # Ejecuta indefinidamente
```

## Próximos Pasos

- Lee el [README.md](README.md) para documentación completa
- Revisa [ARCHITECTURE.md](docs/ARCHITECTURE.md) para entender la arquitectura
- Consulta [API.md](docs/API.md) para documentación de la API
- Explora [FEATURES.md](FEATURES.md) para ver todas las características
