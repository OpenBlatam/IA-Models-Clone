# Quick Start - Contabilidad Mexicana AI

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar o navegar al directorio
cd contabilidad_mexicana_ai

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key
export OPENROUTER_API_KEY="tu-api-key"
# O crear archivo .env:
# OPENROUTER_API_KEY=tu-api-key
```

### 2. Uso Básico (Python)

```python
import asyncio
from contabilidad_mexicana_ai import ContadorAI, ContadorConfig

async def main():
    # Inicializar
    config = ContadorConfig()
    contador = ContadorAI(config)
    
    # Calcular impuestos
    resultado = await contador.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={"ingresos_mensuales": 50000}
    )
    
    print(resultado['resultado'])
    
    # Cerrar conexiones
    await contador.close()

asyncio.run(main())
```

### 3. Iniciar API

```bash
# Opción 1: Usando uvicorn directamente
uvicorn main:app --reload --port 8000

# Opción 2: Ejecutar main.py
python main.py
```

### 4. Probar Endpoints

```bash
# Health check
curl http://localhost:8000/api/contador/health

# Calcular impuestos
curl -X POST http://localhost:8000/api/contador/calcular-impuestos \
  -H "Content-Type: application/json" \
  -d '{
    "regimen": "RESICO",
    "tipo_impuesto": "ISR",
    "datos": {"ingresos_mensuales": 50000}
  }'

# Asesoría fiscal
curl -X POST http://localhost:8000/api/contador/asesoria-fiscal \
  -H "Content-Type: application/json" \
  -d '{
    "pregunta": "¿Qué deducciones puedo aplicar?",
    "contexto": {"regimen": "RESICO"}
  }'
```

### 5. Documentación Interactiva

Una vez iniciado el servidor, visita:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📚 Ejemplos Completos

Ver `examples/example_usage.py` para ejemplos completos de todas las funcionalidades.

## ⚙️ Configuración

### Variables de Entorno

```bash
OPENROUTER_API_KEY=tu-api-key
```

### Configuración Avanzada

```python
from contabilidad_mexicana_ai import ContadorConfig, OpenRouterConfig

config = ContadorConfig(
    openrouter=OpenRouterConfig(
        default_model="anthropic/claude-3.5-sonnet",
        temperature=0.3,
        max_tokens=4000
    )
)
```

## 🔍 Verificación

```python
# Verificar configuración
from contabilidad_mexicana_ai import ContadorConfig

config = ContadorConfig()
config.validate()  # Lanza error si falta API key
```

## 📖 Siguiente Paso

Lee el [README.md](README.md) para documentación completa.
