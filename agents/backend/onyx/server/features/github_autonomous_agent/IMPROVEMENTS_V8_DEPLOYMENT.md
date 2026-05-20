# Guía de Deployment - Mejoras V8

## Proceso Completo de Deployment

---

## 📋 Pre-Deployment Checklist

### Verificación de Código

```bash
# 1. Verificar constantes
python scripts/verify-constants-usage.py

# 2. Verificar decoradores
python scripts/analyze-decorator-usage.py

# 3. Buscar strings hardcodeados
python scripts/find-hardcoded-strings.py

# 4. Ejecutar tests
pytest tests/ -v --cov=core --cov=api

# 5. Type checking
mypy core/ api/

# 6. Linting
ruff check core/ api/
# o
flake8 core/ api/
```

**Resultado esperado**: Todos los checks pasan ✅

---

### Verificación de Documentación

- [ ] Documentación actualizada
- [ ] Changelog actualizado
- [ ] README actualizado (si aplica)
- [ ] Ejemplos verificados

---

### Verificación de Tests

- [ ] Todos los tests unitarios pasan
- [ ] Todos los tests de integración pasan
- [ ] Cobertura >= 80%
- [ ] Tests de regresión pasan

---

## 🚀 Proceso de Deployment

### Fase 1: Preparación

**1.1 Crear Release Branch**
```bash
git checkout -b release/v8.0.0
git push origin release/v8.0.0
```

**1.2 Verificar Versión**
```bash
# Verificar que la versión es correcta
grep -r "V8\|v8.0.0" --include="*.md"
```

**1.3 Actualizar Changelog**
```bash
# Verificar que CHANGELOG está actualizado
cat IMPROVEMENTS_V8_CHANGELOG.md
```

---

### Fase 2: Build

**2.1 Build Local**
```bash
# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Ejecutar tests
pytest tests/ -v

# Verificar que todo funciona
python -m pytest tests/ --cov
```

**2.2 Build Docker (si aplica)**
```bash
# Build imagen
docker build -t github-autonomous-agent:v8.0.0 .

# Test imagen
docker run --rm github-autonomous-agent:v8.0.0 pytest tests/ -v
```

---

### Fase 3: Staging Deployment

**3.1 Deploy a Staging**
```bash
# Ejemplo con Kubernetes
kubectl apply -f k8s/staging/

# Verificar deployment
kubectl get pods -n staging
kubectl logs -f deployment/github-autonomous-agent -n staging
```

**3.2 Verificación en Staging**
```bash
# Health check
curl https://staging-api.example.com/health

# Verificar logs
kubectl logs -f deployment/github-autonomous-agent -n staging | grep ERROR

# Verificar métricas
# (usar tu herramienta de métricas)
```

**3.3 Smoke Tests**
```bash
# Ejecutar smoke tests
pytest tests/smoke/ -v

# Verificar funcionalidad crítica
# (tests específicos de tu aplicación)
```

---

### Fase 4: Production Deployment

**4.1 Pre-Deployment**
```bash
# Backup de producción
# (proceso específico de tu infraestructura)

# Verificar que staging está estable
# (mínimo 24 horas sin issues críticos)
```

**4.2 Deploy a Production**
```bash
# Deploy gradual (recomendado)
kubectl set image deployment/github-autonomous-agent \
  app=github-autonomous-agent:v8.0.0 \
  -n production

# O deploy completo
kubectl apply -f k8s/production/
```

**4.3 Verificación Post-Deployment**
```bash
# Health check
curl https://api.example.com/health

# Verificar logs
kubectl logs -f deployment/github-autonomous-agent -n production

# Verificar métricas
# (usar tu herramienta de métricas)

# Monitoreo activo (primeras 2 horas)
```

---

## 🔍 Monitoreo Post-Deployment

### Métricas a Monitorear

**Primera Hora:**
- Tasa de errores
- Tiempo de respuesta
- Uso de recursos
- Logs de error

**Primeras 24 Horas:**
- Tasa de errores (comparar con baseline)
- Performance (comparar con baseline)
- Uso de constantes (verificar que se usan)
- Uso de decoradores (verificar que funcionan)

---

### Alertas Configuradas

**Alertas Críticas:**
- Error rate > 5%
- Response time > 2s (p95)
- CPU > 80%
- Memory > 90%

**Alertas de Advertencia:**
- Error rate > 2%
- Response time > 1s (p95)
- CPU > 60%
- Memory > 70%

---

## 🔄 Rollback Plan

### Criterios de Rollback

**Rollback inmediato si:**
- Error rate > 10%
- Sistema no responde
- Data loss detectado
- Security issue detectado

---

### Proceso de Rollback

**1. Detener Deployment**
```bash
# Pausar rollout
kubectl rollout pause deployment/github-autonomous-agent -n production
```

**2. Rollback a Versión Anterior**
```bash
# Rollback
kubectl rollout undo deployment/github-autonomous-agent -n production

# Verificar rollback
kubectl rollout status deployment/github-autonomous-agent -n production
```

**3. Verificación Post-Rollback**
```bash
# Health check
curl https://api.example.com/health

# Verificar logs
kubectl logs -f deployment/github-autonomous-agent -n production

# Verificar que sistema está estable
```

**4. Documentar Rollback**
```markdown
## Rollback V8.0.0 - [Fecha]

### Razón
[Por qué se hizo rollback]

### Acciones Tomadas
[Qué se hizo]

### Lecciones Aprendidas
[Qué aprendimos]
```

---

## 📊 Verificación de Deployment

### Checklist Post-Deployment

**Inmediato (1 hora):**
- [ ] Health check pasa
- [ ] No hay errores críticos en logs
- [ ] Métricas dentro de rangos normales
- [ ] Funcionalidad básica funciona

**Corto plazo (24 horas):**
- [ ] Tasa de errores <= baseline
- [ ] Performance <= baseline
- [ ] No hay regresiones reportadas
- [ ] Uso de constantes verificado
- [ ] Uso de decoradores verificado

**Largo plazo (1 semana):**
- [ ] No hay issues críticos
- [ ] Feedback del equipo positivo
- [ ] Métricas mejoradas (si aplica)
- [ ] Documentación actualizada

---

## 🎯 Métricas de Éxito

### KPIs de Deployment

**Técnicos:**
- ✅ Error rate: <= baseline
- ✅ Response time: <= baseline
- ✅ Uptime: >= 99.9%
- ✅ Cobertura de tests: >= 80%

**Funcionales:**
- ✅ Todas las features funcionan
- ✅ No hay regresiones
- ✅ Performance mejorada o igual

**De Negocio:**
- ✅ Tiempo de debugging: ⬇️ 60%
- ✅ Tiempo de desarrollo: ⬇️ 40%
- ✅ Satisfacción del equipo: ⬆️

---

## 🔧 Troubleshooting de Deployment

### Problema 1: Tests Fallan en CI/CD

**Síntoma**: Tests pasan localmente pero fallan en CI/CD

**Solución**:
```bash
# Verificar entorno
python --version
pip list

# Ejecutar tests con verbose
pytest tests/ -v -s

# Verificar variables de entorno
env | grep GITHUB
```

---

### Problema 2: Deployment Falla

**Síntoma**: Deployment no completa

**Solución**:
```bash
# Verificar logs de deployment
kubectl describe deployment/github-autonomous-agent -n production

# Verificar eventos
kubectl get events -n production --sort-by='.lastTimestamp'

# Verificar recursos
kubectl get pods -n production
```

---

### Problema 3: Errores Post-Deployment

**Síntoma**: Errores después de deployment

**Solución**:
```bash
# Verificar logs
kubectl logs -f deployment/github-autonomous-agent -n production

# Verificar constantes
# (usar scripts de verificación)

# Verificar decoradores
# (usar scripts de verificación)

# Rollback si es necesario
```

---

## 📚 Recursos

- [IMPROVEMENTS_V8.md](IMPROVEMENTS_V8.md) - Documentación completa
- [IMPROVEMENTS_V8_TROUBLESHOOTING.md](IMPROVEMENTS_V8_TROUBLESHOOTING.md) - Troubleshooting
- [IMPROVEMENTS_V8_CHANGELOG.md](IMPROVEMENTS_V8_CHANGELOG.md) - Changelog

---

**Última actualización**: Enero 2025  
**Versión**: V8



