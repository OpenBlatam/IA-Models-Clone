# 🚀 Inicio Rápido - 3D Prototype AI

## Instalación Rápida

```bash
# Navegar al directorio
cd agents/backend/onyx/server/features/3d_prototype_ai

# Instalar dependencias
pip install -r requirements.txt
```

## Uso del Chat Interface

La forma más fácil de usar el sistema es a través del chat interactivo:

```bash
python chat_interface.py
```

Luego simplemente escribe qué quieres hacer:

```
💬 Tú: Quiero hacer una nueva licuadora
```

El sistema generará automáticamente:
- ✅ Lista de materiales con precios
- ✅ Modelos CAD por partes
- ✅ Instrucciones de ensamblaje
- ✅ Opciones según presupuesto
- ✅ Documentos completos en JSON

## Uso de la API REST

### Iniciar el servidor

```bash
python main.py
```

### Hacer una petición

```bash
curl -X POST "http://localhost:8030/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "product_description": "Quiero hacer una nueva licuadora",
    "product_type": "licuadora",
    "budget": 150.0
  }'
```

### Ver documentación de la API

Abre en tu navegador: http://localhost:8030/docs

## Ejemplos de Uso

### Ejemplo 1: Licuadora

```python
from core.prototype_generator import PrototypeGenerator
from models.schemas import PrototypeRequest, ProductType
import asyncio

async def main():
    generator = PrototypeGenerator()
    request = PrototypeRequest(
        product_description="Quiero hacer una nueva licuadora potente",
        product_type=ProductType.LICUADORA,
        budget=150.0
    )
    response = await generator.generate_prototype(request)
    print(f"Producto: {response.product_name}")
    print(f"Costo: ${response.total_cost_estimate}")

asyncio.run(main())
```

### Ejemplo 2: Estufa

```python
request = PrototypeRequest(
    product_description="Necesito diseñar una estufa de gas de 4 quemadores",
    product_type=ProductType.ESTUFA,
    budget=300.0
)
```

### Ejemplo 3: Máquina Personalizada

```python
request = PrototypeRequest(
    product_description="Quiero crear una máquina para cortar madera",
    product_type=ProductType.MAQUINA,
    requirements=["Segura", "Precisa"]
)
```

## Ejecutar Ejemplos

```bash
python example_usage.py
```

## Estructura de Respuesta

La respuesta incluye:

- **product_name**: Nombre del producto
- **specifications**: Especificaciones técnicas
- **materials**: Lista de materiales con precios y fuentes
- **cad_parts**: Partes del modelo CAD
- **assembly_instructions**: Instrucciones paso a paso
- **budget_options**: Opciones según presupuesto (bajo, medio, alto, premium)
- **total_cost_estimate**: Costo total estimado
- **estimated_build_time**: Tiempo estimado de construcción
- **difficulty_level**: Nivel de dificultad
- **documents**: Rutas a documentos generados

## Documentos Generados

Los documentos se guardan en `output/prototypes/`:

- `{producto}_completo.json`: Documento completo con toda la información
- `{producto}_materiales.json`: Lista detallada de materiales

## Próximos Pasos

1. Ejecuta `python chat_interface.py` para probar el sistema
2. Revisa los documentos generados en `output/prototypes/`
3. Personaliza los materiales en `core/prototype_generator.py`
4. Integra con tu sistema usando la API REST




