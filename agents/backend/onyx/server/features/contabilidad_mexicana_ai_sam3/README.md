# Mexican Accounting AI SAM3

> Part of the [Blatam Academy Integrated Platform](../README.md)

Mexican accounting system with SAM3 architecture, integrated with OpenRouter and TruthGPT.

## Features

- ✅ SAM3 architecture for parallel and continuous processing
- ✅ OpenRouter integration for high-quality LLMs
- ✅ TruthGPT integration for advanced optimization
- ✅ Continuous 24/7 operation
- ✅ Parallel task execution
- ✅ Automatic task management with priority queue
- ✅ Mexican accounting services:
  - Tax calculation
  - Tax advice
  - Tax guides
  - SAT procedures
  - Tax return help

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Configure environment variables:

```bash
export OPENROUTER_API_KEY="your-api-key"
export TRUTHGPT_ENDPOINT="optional-endpoint"  # Optional
```

## Basic Usage

```python
import asyncio
from contabilidad_mexicana_ai_sam3 import ContadorSAM3Agent, ContadorSAM3Config

async def main():
    # Create configuration
    config = ContadorSAM3Config()
    
    # Create agent
    agent = ContadorSAM3Agent(config=config)
    
    # Start agent (24/7 mode)
    # await agent.start()  # In production
    
    # Or use direct methods
    task_id = await agent.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={"ingresos": 100000, "gastos": 30000}
    )
    
    # Wait for result
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

## Architecture

For complete architecture documentation, see:
- **[COMPLETE_ARCHITECTURE.md](docs/COMPLETE_ARCHITECTURE.md)** - Complete and detailed documentation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Architecture summary

### Directory Structure

```
contabilidad_mexicana_ai_sam3/
├── core/
│   ├── contador_sam3_agent.py    # Main agent (orchestrator)
│   ├── task_manager.py            # Task management and priority queue
│   ├── parallel_executor.py       # Parallel execution with worker pool
│   ├── prompt_builder.py          # User prompt builder
│   ├── system_prompts_builder.py # Specialized system prompts
│   └── helpers.py                 # Common utilities
├── infrastructure/
│   ├── openrouter_client.py       # OpenRouter client (LLM)
│   ├── truthgpt_client.py         # TruthGPT client (optimization)
│   └── retry_helpers.py           # Retry helpers with backoff
├── config/
│   └── contador_sam3_config.py    # Centralized configuration
├── api/
│   └── contador_sam3_api.py       # REST API (FastAPI, optional)
├── utils/
│   ├── formatters.py              # Tax data formatting
│   └── validators.py              # Input validation
├── tests/
│   └── test_contador_sam3_agent.py # Test suite
├── examples/
│   ├── example_usage.py           # Basic examples
│   └── advanced_examples.py        # Advanced examples
└── docs/
    ├── COMPLETE_ARCHITECTURE.md   # Complete architecture
    ├── ARCHITECTURE.md            # Architecture summary
    └── API.md                     # API documentation
```

## Available Services

### 1. Tax Calculation

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

### 2. Tax Advice

```python
task_id = await agent.asesoria_fiscal(
    pregunta="Can I deduct home office expenses?",
    contexto={"regimen": "RESICO", "ingresos_anuales": 500000}
)
```

### 3. Tax Guide

```python
task_id = await agent.guia_fiscal(
    tema="RESICO Deductions",
    nivel_detalle="completo"
)
```

### 4. SAT Procedures

```python
task_id = await agent.tramite_sat(
    tipo_tramite="RFC Registration",
    detalles={"persona_fisica": True}
)
```

### 5. Tax Return Help

```python
task_id = await agent.ayuda_declaracion(
    tipo_declaracion="mensual",
    periodo="2024-01",
    datos={"rfc": "ABC123456789"}
)
```

## Advanced Features

### Continuous 24/7 Mode

The agent can run in continuous mode processing tasks automatically:

```python
agent = ContadorSAM3Agent(config=config)
await agent.start()  # Runs indefinitely
```

### Task Prioritization

Tasks can have different priorities:

```python
# High priority
task_id = await agent.calcular_impuestos(
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={...},
    priority=10  # Higher priority
)
```

### TruthGPT Integration

The agent automatically optimizes queries using TruthGPT when available.

## Requirements

- Python 3.8+
- OpenRouter API key
- TruthGPT (optional but recommended)

## License

Proprietary - Blatam Academy

---

[← Back to Main README](../README.md)
