# 🔌 Guía de Integración - AI Tutor Educacional

## 📋 Integración Básica

### Python SDK

```python
from sdk import TutorClient

client = TutorClient(base_url="https://api.tutor.example.com")
response = client.ask_question("¿Qué es la fotosíntesis?", subject="ciencias")
```

### REST API

```bash
curl -X POST http://localhost:8000/api/tutor/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué es la fotosíntesis?",
    "subject": "ciencias"
  }'
```

## 🔗 Integración con LMS

### Moodle

```python
from core.lms_integration import LMSIntegration, LMSType

lms = LMSIntegration(
    lms_type=LMSType.MOODLE,
    api_key="moodle-api-key",
    base_url="https://moodle.example.com"
)

# Sincronizar estudiante
result = lms.sync_student("student_001", student_data)
```

### Canvas

```python
lms = LMSIntegration(
    lms_type=LMSType.CANVAS,
    api_key="canvas-api-key",
    base_url="https://canvas.example.com"
)

# Sincronizar calificaciones
result = lms.sync_grades("student_001", grades_data)
```

## 🔔 Webhooks

### Registrar Webhook

```python
from core.webhooks import WebhookManager, WebhookEvent

webhook_manager = WebhookManager()

webhook = webhook_manager.register_webhook(
    url="https://your-app.com/webhooks/tutor",
    events=[
        WebhookEvent.QUESTION_ASKED,
        WebhookEvent.QUIZ_COMPLETED,
        WebhookEvent.ACHIEVEMENT_UNLOCKED
    ],
    secret="your-webhook-secret"
)
```

### Recibir Eventos

```python
# En tu aplicación webhook
@app.post("/webhooks/tutor")
async def receive_webhook(request: Request):
    payload = await request.json()
    
    # Verificar firma
    signature = request.headers.get("X-Webhook-Signature")
    # ... verificación ...
    
    event = payload["event"]
    data = payload["data"]
    
    if event == "question.asked":
        # Procesar pregunta
        pass
    elif event == "quiz.completed":
        # Procesar quiz completado
        pass
```

## 📊 Integración con Analytics

### Enviar Datos a Analytics

```python
from core.dashboard_analytics import DashboardAnalytics

analytics = DashboardAnalytics(metrics_collector, learning_analyzer, gamification_system)

# Obtener estadísticas
overview = analytics.get_overview_stats()
engagement = analytics.get_engagement_metrics()
insights = analytics.get_learning_insights()
```

## 🔐 Autenticación

### Registrar Usuario

```python
from core.auth import AuthManager

auth = AuthManager()

user = auth.register_user(
    email="student@example.com",
    username="student123",
    password="secure_password",
    role="student"
)
```

### Login

```python
session_token = auth.authenticate("student@example.com", "secure_password")

# Usar token en requests
headers = {"Authorization": f"Bearer {session_token}"}
```

## 📤 Exportación de Datos

### Exportar Reportes

```python
from core.report_generator import ReportGenerator

report_gen = ReportGenerator(learning_analyzer, metrics_collector)

# Generar reporte
report = report_gen.generate_student_report("student_001")

# Exportar
report_gen.export_report(report, format="json", output_path="report.json")
report_gen.export_report(report, format="html", output_path="report.html")
```

## 🎮 Gamificación

### Registrar Acciones

```python
from core.gamification import GamificationSystem

gamification = GamificationSystem()

# Registrar acción
gamification.record_action("student_001", "ask_question")

# Obtener perfil
profile = gamification.get_student_profile("student_001")
print(f"Puntos: {profile['points']}, Nivel: {profile['level']}")
```

## 🔄 Sincronización en Tiempo Real

### Usar WebSockets (Futuro)

```python
# Ejemplo de integración WebSocket
import websocket

def on_message(ws, message):
    data = json.loads(message)
    if data["type"] == "answer_received":
        # Procesar respuesta
        pass

ws = websocket.WebSocketApp("ws://api.tutor.example.com/ws")
ws.on_message = on_message
ws.run_forever()
```

## 📱 Integración Mobile

### React Native

```javascript
import axios from 'axios';

const tutorAPI = axios.create({
  baseURL: 'https://api.tutor.example.com',
  headers: { 'Content-Type': 'application/json' }
});

// Hacer pregunta
const response = await tutorAPI.post('/api/tutor/ask', {
  question: '¿Qué es la fotosíntesis?',
  subject: 'ciencias'
});
```

## 🌐 Integración Web

### JavaScript/TypeScript

```typescript
class TutorClient {
  constructor(private baseUrl: string) {}
  
  async askQuestion(question: string, subject?: string) {
    const response = await fetch(`${this.baseUrl}/api/tutor/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, subject })
    });
    return response.json();
  }
}

const client = new TutorClient('https://api.tutor.example.com');
const result = await client.askQuestion('¿Qué es la fotosíntesis?', 'ciencias');
```

## 🔒 Seguridad

### Validar Inputs

```python
from core.advanced_validation import AdvancedValidator

validator = AdvancedValidator()

try:
    validator.validate_question(question)
    validator.validate_subject(subject, allowed_subjects)
    validator.validate_difficulty(difficulty)
except ValidationError as e:
    # Manejar error
    pass
```

## 📈 Monitoreo

### Health Checks

```python
import requests

def check_health():
    response = requests.get("https://api.tutor.example.com/api/tutor/health")
    return response.json()["status"] == "healthy"
```

## 🆘 Soporte

Para más información sobre integración:
- Documentación API: `/docs`
- Ejemplos: `examples/`
- SDK: `sdk/README.md`






