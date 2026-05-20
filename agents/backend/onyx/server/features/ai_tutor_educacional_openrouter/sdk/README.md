# Python SDK - AI Tutor Educacional

SDK oficial de Python para interactuar con la API de AI Tutor Educacional.

## 📦 Instalación

```bash
pip install -r requirements.txt
```

O instala directamente desde el directorio:

```bash
cd sdk
pip install -e .
```

## 🚀 Uso Básico

```python
from sdk import TutorClient

# Inicializar cliente
client = TutorClient(
    base_url="http://localhost:8000",
    api_key="optional-api-key"
)

# Hacer una pregunta
response = client.ask_question(
    question="¿Qué es la fotosíntesis?",
    subject="ciencias",
    difficulty="intermedio"
)

print(response["data"]["answer"])

# Explicar un concepto
explanation = client.explain_concept(
    concept="derivadas",
    subject="matematicas",
    difficulty="avanzado"
)

# Generar ejercicios
exercises = client.generate_exercises(
    topic="algebra",
    subject="matematicas",
    num_exercises=5
)

# Generar quiz
quiz = client.generate_quiz(
    topic="biologia",
    subject="ciencias",
    num_questions=10
)

# Cerrar cliente
client.close()
```

## 📚 Métodos Disponibles

### Preguntas y Respuestas
- `ask_question()` - Hacer una pregunta
- `explain_concept()` - Explicar un concepto
- `generate_exercises()` - Generar ejercicios
- `generate_quiz()` - Generar quiz

### Sistema
- `get_metrics()` - Obtener métricas
- `get_health()` - Verificar salud del sistema

## 🔧 Configuración

```python
client = TutorClient(
    base_url="https://api.tutor.example.com",
    api_key="your-api-key",
    timeout=120  # segundos
)
```

## 📝 Ejemplos Completos

Ver `examples/` para más ejemplos de uso.






