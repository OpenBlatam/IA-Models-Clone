# 🎯 AI Job Replacement Helper

Sistema inteligente que ayuda a las personas cuando una IA les quita su trabajo. Incluye gamificación, pasos guiados personalizados y búsqueda de trabajo estilo Tinder con integración de LinkedIn API.

## ✨ Características Principales

### 🎮 Sistema de Gamificación
- **Puntos y niveles**: Gana puntos por completar acciones y sube de nivel
- **Badges y logros**: Desbloquea badges por hitos importantes
- **Rachas (Streaks)**: Mantén rachas de días consecutivos activos
- **Leaderboards**: Compite con otros usuarios
- **Sistema de recompensas**: Obtén puntos por cada acción importante

### 📋 Pasos Guiados Personalizados
- **Roadmap completo**: 10 pasos estructurados desde evaluación hasta aplicación
- **Progreso visual**: Ve tu progreso en cada categoría
- **Recursos integrados**: Acceso a artículos, videos, herramientas y plantillas
- **Prerrequisitos inteligentes**: Los pasos se desbloquean según tu progreso
- **Tracking detallado**: Registra cuándo iniciaste y completaste cada paso

### 💼 Búsqueda de Trabajo Estilo Tinder
- **Swipe de trabajos**: Like/dislike trabajos como en Tinder
- **Integración con LinkedIn**: Busca trabajos reales de LinkedIn
- **Matching inteligente**: Score de compatibilidad con cada trabajo
- **Trabajos guardados**: Guarda trabajos para revisar después
- **Aplicación directa**: Aplica a trabajos desde la plataforma
- **Matches mutuos**: Ve cuando hay interés mutuo

### 🤖 Recomendaciones Inteligentes
- **Habilidades recomendadas**: IA sugiere qué habilidades aprender
- **Trabajos personalizados**: Recomendaciones basadas en tu perfil
- **Próximos pasos**: Sugerencias de qué hacer a continuación
- **Análisis de gaps**: Identifica qué habilidades te faltan

### 🔔 Sistema de Notificaciones
- **Notificaciones inteligentes**: Alertas personalizadas
- **Recordatorios**: No pierdas tu racha ni pasos importantes
- **Logros**: Notificaciones cuando desbloqueas badges o subes de nivel
- **Matches de trabajo**: Alertas cuando hay interés mutuo

### 👨‍🏫 Mentoría y Coaching con IA
- **Coaches especializados**: Career coach, tech mentor, interview coach
- **Sesiones personalizadas**: Chats con IA para guía profesional
- **Consejos de carrera**: Análisis de tu situación y objetivos
- **Tips para entrevistas**: Preparación específica por trabajo
- **Mensajes motivacionales**: Mantén la motivación alta

### 📄 Análisis de CV con IA
- **Análisis completo**: Score general y por sección
- **Feedback detallado**: Fortalezas y áreas de mejora
- **Score ATS**: Compatibilidad con sistemas de tracking
- **Análisis de keywords**: Coincidencia con trabajos objetivo
- **Sugerencias específicas**: Mejoras concretas para tu CV

### 🎤 Simulador de Entrevistas
- **Entrevistas simuladas**: Practica con IA
- **Múltiples tipos**: Técnicas, de comportamiento, cultural fit
- **Feedback en tiempo real**: Análisis de tus respuestas
- **Score y mejoras**: Identifica qué mejorar
- **Banco de preguntas**: Preguntas reales de entrevistas

### 🏆 Sistema de Desafíos
- **Desafíos diarios**: Misiones diarias para mantenerte activo
- **Desafíos semanales**: Objetivos semanales más grandes
- **Logros especiales**: Badges únicos por hitos importantes
- **Recompensas**: Puntos, XP y badges por completar desafíos
- **Tracking de progreso**: Ve tu avance en tiempo real

### 📊 Dashboard y Analytics
- **Métricas completas**: Vista 360° de tu progreso
- **Tendencias**: Gráficos de actividad y crecimiento
- **Estadísticas de actividad**: Análisis de tus acciones
- **Leaderboards**: Compara tu progreso con otros

### ✍️ Generador de Contenido con IA
- **Cartas de presentación**: Genera cartas personalizadas automáticamente
- **Posts de LinkedIn**: Crea posts profesionales para compartir logros
- **Emails de seguimiento**: Genera emails de follow-up profesionales
- **Notas de agradecimiento**: Crea notas post-entrevista
- **Mejora de texto**: Mejora textos con diferentes estilos

### 🔔 Alertas de Trabajos Inteligentes
- **Alertas personalizadas**: Crea alertas con keywords, ubicación y tipo
- **Búsqueda automática**: Verifica periódicamente nuevos trabajos
- **Frecuencias configurables**: Daily, weekly, o real-time
- **Rango salarial**: Filtros por rango salarial
- **Tracking de matches**: Cuenta cuántos trabajos coinciden

### 🔗 Integraciones con APIs Externas
- **GitHub**: Muestra proyectos y contribuciones
- **Stack Overflow**: Integra reputación y respuestas
- **Medium**: Muestra artículos publicados
- **Sincronización**: Sincroniza datos de plataformas integradas

### 📱 Notificaciones Inteligentes
- **Canales múltiples**: In-app, email, push, SMS
- **Prioridades**: Low, medium, high, urgent
- **Tiempo óptimo**: Calcula el mejor momento para enviar
- **Preferencias de usuario**: Respeta horarios y canales preferidos
- **Agrupación**: Agrupa notificaciones para evitar spam

## 🚀 Instalación

### Requisitos
- Python 3.10+
- PostgreSQL (opcional, para producción)
- Redis (opcional, para cache)

### Instalación Local

```bash
# Clonar o navegar al directorio
cd ai_job_replacement_helper

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Ejecutar servidor
python main.py
```

El servidor estará disponible en `http://localhost:8030`

## 📚 API Endpoints

### Gamificación

- `GET /api/v1/gamification/progress/{user_id}` - Obtener progreso completo
- `POST /api/v1/gamification/points/{user_id}?action={action}&amount={amount}` - Agregar puntos
- `GET /api/v1/gamification/leaderboard?limit={limit}` - Ver leaderboard
- `GET /api/v1/gamification/badges/{user_id}` - Ver badges del usuario

### Pasos Guiados

- `GET /api/v1/steps/roadmap/{user_id}` - Obtener roadmap completo
- `GET /api/v1/steps/progress/{user_id}` - Obtener progreso de pasos
- `POST /api/v1/steps/start/{user_id}` - Iniciar un paso
- `POST /api/v1/steps/complete/{user_id}` - Completar un paso

### Trabajos (LinkedIn)

- `GET /api/v1/jobs/search/{user_id}?keywords={keywords}&location={location}` - Buscar trabajos
- `POST /api/v1/jobs/swipe/{user_id}` - Hacer swipe (like/dislike/save)
- `POST /api/v1/jobs/apply/{user_id}?job_id={job_id}` - Aplicar a trabajo
- `GET /api/v1/jobs/saved/{user_id}` - Trabajos guardados
- `GET /api/v1/jobs/liked/{user_id}` - Trabajos que te gustaron
- `GET /api/v1/jobs/matches/{user_id}` - Matches (interés mutuo)
- `GET /api/v1/jobs/statistics/{user_id}` - Estadísticas del usuario

### Recomendaciones

- `GET /api/v1/recommendations/skills/{user_id}?target_industry={industry}` - Recomendaciones de habilidades
- `GET /api/v1/recommendations/jobs/{user_id}?location={location}` - Recomendaciones de trabajos
- `GET /api/v1/recommendations/next-steps/{user_id}` - Próximos pasos recomendados

### Notificaciones

- `GET /api/v1/notifications/{user_id}?unread_only={bool}&limit={limit}` - Obtener notificaciones
- `GET /api/v1/notifications/unread-count/{user_id}` - Contar no leídas
- `POST /api/v1/notifications/mark-read/{user_id}/{notification_id}` - Marcar como leída
- `POST /api/v1/notifications/mark-all-read/{user_id}` - Marcar todas como leídas

### Mentoría

- `POST /api/v1/mentoring/start/{user_id}?session_type={type}&mentor_type={type}` - Iniciar sesión
- `POST /api/v1/mentoring/ask/{user_id}/{session_id}?question={question}` - Preguntar al mentor
- `GET /api/v1/mentoring/career-advice/{user_id}?current_situation={situation}&goals={goals}` - Consejo de carrera
- `GET /api/v1/mentoring/interview-tips/{user_id}?job_title={title}&company={company}` - Tips para entrevista
- `GET /api/v1/mentoring/motivation/{user_id}?current_mood={mood}` - Mensaje motivacional

### Análisis de CV

- `POST /api/v1/cv/analyze/{user_id}` - Analizar CV (body: cv_content, target_job opcional)

### Simulador de Entrevistas

- `POST /api/v1/interview/start/{user_id}?interview_type={type}&job_title={title}&company={company}` - Iniciar entrevista
- `POST /api/v1/interview/answer/{user_id}/{session_id}?question_id={id}&answer={answer}` - Enviar respuesta
- `POST /api/v1/interview/complete/{user_id}/{session_id}` - Completar y obtener resultados

### Desafíos

- `GET /api/v1/challenges/available/{user_id}?challenge_type={type}` - Desafíos disponibles
- `POST /api/v1/challenges/start/{user_id}/{challenge_id}` - Iniciar desafío
- `POST /api/v1/challenges/progress/{user_id}/{challenge_id}?progress={0.0-1.0}` - Actualizar progreso
- `POST /api/v1/challenges/complete/{user_id}/{challenge_id}` - Completar desafío

### Dashboard

- `GET /api/v1/dashboard/{user_id}` - Dashboard completo
- `GET /api/v1/dashboard/metrics/{user_id}` - Métricas del usuario
- `GET /api/v1/dashboard/activity/{user_id}?days={days}` - Estadísticas de actividad

### Generador de Contenido

- `POST /api/v1/content/cover-letter` - Generar carta de presentación
- `POST /api/v1/content/linkedin-post` - Generar post de LinkedIn
- `POST /api/v1/content/follow-up-email` - Generar email de seguimiento
- `POST /api/v1/content/improve-text` - Mejorar texto con IA

### Alertas de Trabajos

- `POST /api/v1/job-alerts/create/{user_id}` - Crear alerta
- `GET /api/v1/job-alerts/{user_id}` - Obtener alertas del usuario
- `POST /api/v1/job-alerts/check/{user_id}` - Verificar alertas y encontrar matches

### Health

- `GET /health` - Health check básico
- `GET /health/detailed` - Health check detallado

## 📖 Ejemplos de Uso

### Buscar trabajos y hacer swipe

```bash
# Buscar trabajos
curl "http://localhost:8030/api/v1/jobs/search/user123?keywords=Python&location=Madrid"

# Hacer like a un trabajo
curl -X POST "http://localhost:8030/api/v1/jobs/swipe/user123" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job_1", "action": "like"}'

# Guardar un trabajo
curl -X POST "http://localhost:8030/api/v1/jobs/swipe/user123" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job_2", "action": "save"}'
```

### Completar pasos y ganar puntos

```bash
# Iniciar un paso
curl -X POST "http://localhost:8030/api/v1/steps/start/user123" \
  -H "Content-Type: application/json" \
  -d '{"step_id": "step_1"}'

# Completar un paso
curl -X POST "http://localhost:8030/api/v1/steps/complete/user123" \
  -H "Content-Type: application/json" \
  -d '{"step_id": "step_1", "notes": "Completé la evaluación"}'

# Ver progreso
curl "http://localhost:8030/api/v1/gamification/progress/user123"
```

### Obtener recomendaciones

```bash
# Recomendaciones de habilidades
curl "http://localhost:8030/api/v1/recommendations/skills/user123?target_industry=tech"

# Recomendaciones de trabajos
curl "http://localhost:8030/api/v1/recommendations/jobs/user123?location=Madrid"

# Próximos pasos
curl "http://localhost:8030/api/v1/recommendations/next-steps/user123"
```

## 🏗️ Arquitectura

```
ai_job_replacement_helper/
├── core/                    # Lógica de negocio
│   ├── gamification.py     # Sistema de gamificación
│   ├── steps_guide.py      # Pasos guiados
│   ├── linkedin_integration.py  # Integración LinkedIn
│   └── recommendations.py  # Recomendaciones IA
├── api/                    # API REST
│   ├── app_factory.py     # Factory de FastAPI
│   └── routes/            # Endpoints
│       ├── gamification.py
│       ├── steps.py
│       ├── jobs.py
│       ├── recommendations.py
│       └── health.py
├── models/                 # Modelos Pydantic
│   └── schemas.py
├── main.py                 # Punto de entrada
└── requirements.txt        # Dependencias
```

## 🎮 Sistema de Gamificación

### Puntos por Acción

- Completar perfil: 50 puntos
- Completar paso: 25 puntos
- Aplicar a trabajo: 100 puntos
- Guardar trabajo: 10 puntos
- Aprender habilidad: 75 puntos
- Contacto de networking: 30 puntos
- Completar desafío: 150 puntos
- Ayudar comunidad: 50 puntos
- Login diario: 20 puntos

### Badges Disponibles

- 🎯 **Primer Paso**: Completaste tu primer paso
- ✅ **Perfil Completo**: Perfil al 100%
- 📝 **Primera Aplicación**: Enviaste tu primera aplicación
- 🔥 **Semana de Dedicación**: 7 días consecutivos
- ⭐ **Mes de Excelencia**: 30 días consecutivos
- 🎓 **Habilidad Aprendida**: Aprendiste una nueva habilidad
- 🤝 **Networking Master**: Muchos contactos
- 💼 **Listo para Entrevista**: Preparado para entrevistas
- 🎉 **Oferta de Trabajo**: Recibiste una oferta
- 👨‍🏫 **Mentor**: Ayudaste a otros

### Niveles

Los niveles van del 1 al 10, con experiencia requerida creciente:
- Nivel 1: 100 XP
- Nivel 2: 250 XP
- Nivel 3: 500 XP
- ... hasta nivel 10: 20,000 XP

## 📋 Roadmap de Pasos

1. **Evalúa tu situación actual** - Auto-análisis profesional
2. **Identifica nuevas habilidades** - Investigación de mercado
3. **Crea un plan de aprendizaje** - Diseño estructurado
4. **Actualiza tu perfil de LinkedIn** - Optimización
5. **Construye tu red profesional** - Networking
6. **Busca oportunidades de trabajo** - Búsqueda activa
7. **Prepara tu CV y carta** - Documentos profesionales
8. **Practica para entrevistas** - Preparación
9. **Aplica a trabajos** - Aplicaciones
10. **Mantén una mentalidad positiva** - Motivación continua

## 🔧 Configuración

### Variables de Entorno

```env
# LinkedIn API
LINKEDIN_API_KEY=your_linkedin_api_key
LINKEDIN_API_SECRET=your_linkedin_api_secret

# Database (opcional)
DATABASE_URL=postgresql://user:password@localhost/dbname

# Redis (opcional)
REDIS_URL=redis://localhost:6379

# App
APP_ENV=development
DEBUG=True
```

## ✅ Funcionalidades Implementadas

- [x] Sistema de gamificación completo
- [x] Pasos guiados y roadmap
- [x] Integración LinkedIn estilo Tinder
- [x] Recomendaciones inteligentes
- [x] Sistema de notificaciones
- [x] Mentoría y coaching con IA
- [x] Análisis de CV con IA
- [x] Simulador de entrevistas
- [x] Sistema de desafíos
- [x] Dashboard y analytics
- [x] Generador de contenido con IA ⭐ NUEVO
- [x] Alertas de trabajos inteligentes ⭐ NUEVO
- [x] Integraciones con APIs externas ⭐ NUEVO
- [x] Notificaciones inteligentes ⭐ NUEVO
- [x] Sistema de comunidad y foros
- [x] Multi-plataforma (LinkedIn, Indeed, Glassdoor, Remote.com)
- [x] Tracking de aplicaciones completo
- [x] Sistema de mensajería
- [x] Eventos y webinars
- [x] Biblioteca de recursos
- [x] Sistema de reportes y exportación
- [x] Autenticación y usuarios
- [x] Sistema de suscripciones
- [x] Sistema de referidos
- [x] Integración social
- [x] Analytics avanzados
- [x] Certificados
- [x] Sistema de feedback
- [x] Internacionalización (i18n)
- [x] A/B Testing
- [x] Integración con calendarios
- [x] Motor de búsqueda avanzado
- [x] Sistema de recordatorios
- [x] Learning paths personalizados
- [x] AI Coach avanzado
- [x] Evaluación de habilidades
- [x] Sistema de colaboración
- [x] Personalidad de IA
- [x] Tracking de progreso avanzado

## 🚧 Próximas Mejoras

- [ ] Integración real con LinkedIn Jobs API (actualmente simulado)
- [ ] Base de datos persistente (PostgreSQL)
- [ ] Sistema de autenticación completo
- [ ] Frontend React con UI moderna
- [ ] Notificaciones push
- [ ] Sistema de comunidad y foros
- [ ] Integración con más plataformas (Indeed, Glassdoor, etc.)
- [ ] Modelos de IA más avanzados (GPT-4, Claude)
- [ ] Video entrevistas simuladas
- [ ] Sistema de networking avanzado

## 📝 Licencia

Propietaria - Blatam Academy

## 👥 Autor

Blatam Academy

---

**Versión**: 2.0.0  
**Última actualización**: 2024

---

## 📚 Documentación Adicional

- [LATEST_IMPROVEMENTS.md](LATEST_IMPROVEMENTS.md) - Últimas mejoras agregadas
- [COMPLETE_FEATURES_LIST.md](COMPLETE_FEATURES_LIST.md) - Lista completa de funcionalidades
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura del sistema
- [DEPLOYMENT.md](DEPLOYMENT.md) - Guía de despliegue
- [QUICK_START.md](QUICK_START.md) - Guía de inicio rápido

