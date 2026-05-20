# 🔗 Guía de Integración - Validación con Proyecto Principal

Guía para integrar el folder de validación con el proyecto principal de Dermatology AI.

## 🎯 Objetivo

Conectar el frontend de validación con el backend del proyecto principal de forma seamless.

---

## 📋 Setup de Integración

### 1. Verificar Backend

**Ubicación del backend**:
```
agents/backend/onyx/server/features/dermatology_ai/
```

**Puerto por defecto**: `8006`

**Verificar que funciona**:
```bash
cd agents/backend/onyx/server/features/dermatology_ai
python main.py
```

Debería mostrar:
```
INFO:     Uvicorn running on http://0.0.0.0:8006
```

---

### 2. Configurar Frontend de Validación

**Ubicación del frontend**:
```
agents/backend/onyx/server/features/dermatology_ai/validation/frontend/
```

**Configurar URL del backend**:

Edita `app.js`:
```javascript
// Línea 2
const API_BASE_URL = 'http://localhost:8006'; // Ajusta si es necesario
```

**Si el backend está en otro puerto o servidor**:
```javascript
const API_BASE_URL = 'http://tu-servidor:8006';
```

---

### 3. Endpoints Disponibles

### Análisis de Imagen
```
POST /dermatology/analyze-image
Content-Type: multipart/form-data

Parameters:
- file: [imagen]
- enhance: true/false
- use_advanced: true/false
- use_cache: true/false
```

### Recomendaciones
```
POST /dermatology/get-recommendations
Content-Type: multipart/form-data

Parameters:
- file: [imagen]
- include_routine: true/false
```

### Health Check
```
GET /health
```

---

## 🔧 Configuración Avanzada

### CORS (Cross-Origin Resource Sharing)

Si tienes problemas de CORS, verifica que el backend permita requests del frontend.

**En el backend**, asegúrate de tener:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### Variables de Entorno

**Backend** (`.env`):
```env
HOST=0.0.0.0
PORT=8006
DEBUG=true
```

**Frontend** (si necesitas):
```javascript
// En app.js, puedes usar variables de entorno
const API_BASE_URL = process.env.API_URL || 'http://localhost:8006';
```

---

## 🚀 Deployment

### Opción 1: Desarrollo Local

**Backend**:
```bash
cd agents/backend/onyx/server/features/dermatology_ai
python main.py
```

**Frontend**:
```bash
cd validation/frontend
python -m http.server 8080
```

Accede a: `http://localhost:8080`

---

### Opción 2: Producción Simple

**Backend en servidor**:
- Deploy backend a servidor (Heroku, Railway, etc.)
- Actualiza `API_BASE_URL` en frontend

**Frontend estático**:
- Sube `frontend/` a Netlify, Vercel, GitHub Pages
- Configura `API_BASE_URL` para apuntar a tu backend

---

### Opción 3: Integración Completa

**Usar frontend del proyecto principal**:
- El proyecto principal ya tiene un frontend completo
- El frontend de validación es solo para validación rápida
- Para producción, usa el frontend principal

---

## 📊 Uso de Métricas

### Integrar con Sistema Principal

**Opción 1: Archivo JSON compartido**
```javascript
// El frontend de validación guarda en results/metrics.json
// El sistema principal puede leer este archivo
```

**Opción 2: API de métricas**
```javascript
// Si el backend tiene endpoint de métricas
fetch('http://localhost:8006/metrics')
```

**Opción 3: Base de datos compartida**
- Si el proyecto principal usa DB
- Guarda métricas de validación ahí también

---

## 🔄 Flujo de Integración

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO DE INTEGRACIÓN                     │
└─────────────────────────────────────────────────────────────┘

1. VALIDACIÓN (validation/)
   │
   ├─ Frontend de validación
   ├─ Scripts de métricas
   └─ Documentación
   │
   ↓ (Después de validar)
   │
2. PROYECTO PRINCIPAL (../)
   │
   ├─ Backend completo
   ├─ Frontend completo
   ├─ ML models
   └─ Features completas
   │
   ↓
   │
3. PRODUCCIÓN
   │
   └─ Sistema completo desplegado
```

---

## 🎯 Casos de Uso

### Caso 1: Validación Rápida

**Setup**:
1. Backend del proyecto principal corriendo
2. Frontend de validación apuntando al backend
3. Validar con usuarios

**Resultado**: Feedback rápido sin construir frontend completo

---

### Caso 2: Validación Continua

**Setup**:
1. Integrar métricas de validación con sistema principal
2. Trackear métricas en producción también
3. Comparar validación vs producción

**Resultado**: Validación continua incluso después de lanzar

---

### Caso 3: A/B Testing

**Setup**:
1. Usar frontend de validación como variante A
2. Usar frontend principal como variante B
3. Comparar métricas

**Resultado**: Test de diferentes UX

---

## 🔧 Troubleshooting de Integración

### El frontend no se conecta al backend

**Solución**:
1. Verifica que backend esté corriendo
2. Verifica URL en `app.js`
3. Verifica CORS en backend
4. Revisa consola del navegador (F12)

Ver [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para más.

---

### Las métricas no se sincronizan

**Solución**:
1. Verifica que `results/metrics.json` se actualice
2. Si usas DB, verifica conexión
3. Si usas API, verifica endpoint

---

### El análisis no funciona

**Solución**:
1. Verifica que endpoint `/dermatology/analyze-image` exista
2. Verifica formato de request
3. Revisa logs del backend

---

## 📝 Checklist de Integración

### Setup Básico
- [ ] Backend del proyecto principal funcionando
- [ ] Frontend de validación configurado
- [ ] URL del backend correcta en `app.js`
- [ ] CORS configurado (si es necesario)

### Testing
- [ ] Frontend se conecta al backend
- [ ] Análisis funciona end-to-end
- [ ] Resultados se muestran correctamente
- [ ] Métricas se recopilan

### Deployment (si aplica)
- [ ] Backend desplegado
- [ ] Frontend desplegado
- [ ] URLs configuradas correctamente
- [ ] CORS configurado para producción

---

## 💡 Tips de Integración

1. **Mantén separado**: Validación separada de producción
2. **Reutiliza backend**: No dupliques backend
3. **Documenta cambios**: Si modificas algo, documenta
4. **Versiona**: Usa git para trackear cambios
5. **Testa**: Prueba integración antes de validar

---

## 🔗 Enlaces Útiles

- **Backend**: [../main.py](../main.py)
- **API Docs**: `http://localhost:8006/docs` (cuando backend corre)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Frontend Principal**: [../frontend/](../frontend/)

---

**Próximo paso**: Configura la integración y comienza a validar! 🔗






