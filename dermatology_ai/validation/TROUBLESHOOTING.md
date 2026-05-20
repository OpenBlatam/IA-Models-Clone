# 🔧 Troubleshooting - Validación Dermatology AI

Soluciones a problemas comunes durante la validación.

## 🐛 Problemas Técnicos

### Backend no responde

**Síntomas**:
- Frontend muestra error de conexión
- `curl http://localhost:8006/health` falla

**Soluciones**:
```bash
# 1. Verifica que esté corriendo
ps aux | grep python  # Linux/Mac
tasklist | findstr python  # Windows

# 2. Reinicia el backend
cd agents/backend/onyx/server/features/dermatology_ai
python main.py

# 3. Verifica el puerto
netstat -an | grep 8006  # Linux/Mac
netstat -an | findstr 8006  # Windows

# 4. Cambia el puerto si está ocupado
# Edita main.py y cambia el puerto
```

---

### Frontend no se conecta al backend

**Síntomas**:
- Error CORS en consola
- "No se puede conectar al backend"

**Soluciones**:
1. **Verifica la URL en app.js**:
```javascript
const API_BASE_URL = 'http://localhost:8006'; // Debe ser correcta
```

2. **Usa un servidor local** (no file://):
```bash
cd validation/frontend
python -m http.server 8080
```

3. **Verifica CORS en backend**:
- Asegúrate que el backend permita CORS
- Revisa configuración de middleware

---

### El análisis tarda mucho

**Síntomas**:
- Más de 30 segundos
- Timeout errors

**Soluciones**:
1. **Verifica logs del backend**:
```bash
# Revisa los logs para ver qué está pasando
```

2. **Optimiza imagen**:
- Usa imágenes más pequeñas (< 5MB)
- Comprime antes de subir

3. **Revisa recursos del servidor**:
- CPU/Memoria disponibles
- Otros procesos corriendo

4. **Cache**:
- Asegúrate que `use_cache=true` en la request

---

### Los resultados no se muestran

**Síntomas**:
- Análisis completa pero no hay resultados
- Error en consola del navegador

**Soluciones**:
1. **Revisa consola del navegador (F12)**:
- Busca errores JavaScript
- Verifica formato de respuesta del backend

2. **Verifica respuesta del backend**:
```bash
curl -X POST http://localhost:8006/dermatology/analyze-image \
  -F "file=@test.jpg" \
  -F "enhance=true"
```

3. **Revisa app.js**:
- Verifica que `displayResults()` funcione
- Revisa estructura de datos esperada

---

## 📊 Problemas de Métricas

### No puedo recopilar métricas

**Síntomas**:
- `collect_metrics.py` no funciona
- No sé qué métricas medir

**Soluciones**:
1. **Usa el script interactivo**:
```bash
python scripts/collect_metrics.py
```

2. **Empieza simple**:
- Solo mide: visitas, conversiones, feedback
- Agrega más métricas después

3. **Usa Google Analytics**:
- Configura tracking básico
- Mide automáticamente

---

### Métricas no tienen sentido

**Síntomas**:
- Números muy bajos/altos
- No sé cómo interpretarlos

**Soluciones**:
1. **Compara con benchmarks**:
- Revisa [METRICS_GUIDE.md](METRICS_GUIDE.md)
- Compara con industria

2. **Contexto es clave**:
- ¿Cuánto tiempo llevas validando?
- ¿Cuántos usuarios probaron?

3. **Busca patrones**:
- ¿Mejoran con el tiempo?
- ¿Hay días mejores que otros?

---

## 👥 Problemas con Usuarios

### No consigo usuarios para probar

**Síntomas**:
- Nadie quiere probar
- Baja respuesta

**Soluciones**:
1. **Empieza con círculo cercano**:
- Amigos, familia
- Más dispuestos a ayudar

2. **Ofrece incentivo**:
- Acceso gratis
- Descuento futuro
- Agradecimiento público

3. **Usa múltiples canales**:
- Reddit, Twitter, Facebook
- Grupos relevantes
- Comunidades online

4. **Sé claro sobre el tiempo**:
- "Solo toma 2 minutos"
- "Es gratis"
- "Tu feedback es valioso"

---

### Usuarios no completan el análisis

**Síntomas**:
- Suben foto pero no ven resultados
- Abandonan a mitad del proceso

**Soluciones**:
1. **Optimiza tiempo de análisis**:
- Debe ser < 15 segundos
- Muestra progreso

2. **Mejora UX**:
- Instrucciones claras
- Feedback visual
- Mensajes de error claros

3. **Pregunta por qué**:
- Envía email de seguimiento
- Pregunta qué los detuvo

---

### Feedback es muy negativo

**Síntomas**:
- Promedio < 3/5
- Muchas críticas

**Soluciones**:
1. **No te desanimes**:
- Feedback negativo es valioso
- Mejor saberlo temprano

2. **Identifica patrones**:
- ¿Qué se critica más?
- ¿Hay algo común?

3. **Pregunta específicamente**:
- "¿Qué mejorarías?"
- "¿Qué te haría cambiar de opinión?"

4. **Itera rápido**:
- Haz cambios rápidos
- Prueba de nuevo

---

## 💰 Problemas de Precios

### No sé qué precio poner

**Síntomas**:
- Duda entre múltiples precios
- No hay datos

**Soluciones**:
1. **Valida con usuarios**:
- Usa [PRICING_STRATEGY.md](PRICING_STRATEGY.md)
- Pregunta en entrevistas

2. **Analiza competencia**:
- ¿Qué cobran otros?
- ¿Cómo te posicionas?

3. **Empieza conservador**:
- Mejor aumentar que reducir
- Puedes ofrecer descuento beta

---

### Usuarios dicen que es muy caro

**Síntomas**:
- Baja intención de pago
- "Muy caro" en feedback

**Soluciones**:
1. **Comunica valor mejor**:
- ¿Qué ahorran?
- ¿Qué obtienen?

2. **Considera freemium**:
- Versión gratis
- Premium pagado

3. **Valida precio específico**:
- "¿Cuánto pagarías?"
- "¿Qué precio sería justo?"

---

## 📱 Problemas de Marketing

### No genero tráfico

**Síntomas**:
- Pocas visitas
- Bajo engagement

**Soluciones**:
1. **Usa múltiples canales**:
- No solo uno
- Prueba diferentes

2. **Mejora mensaje**:
- Más claro
- Más específico
- Más valor

3. **Timing**:
- Publica cuando audiencia está activa
- Diferentes horarios

4. **Consistencia**:
- Publica regularmente
- No te rindas rápido

---

### Posts no tienen engagement

**Síntomas**:
- Pocos likes/comentarios
- Bajo click-through

**Soluciones**:
1. **Mejora contenido**:
- Más visual
- Más específico
- Más valor

2. **Pide engagement**:
- "¿Qué opinas?"
- "Comparte si te gusta"

3. **Participa en comunidad**:
- No solo promociona
- Responde comentarios
- Ayuda a otros

---

## 🎤 Problemas de Presentación

### Nervios antes de presentar

**Síntomas**:
- Ansiedad
- Miedo a fallar

**Soluciones**:
1. **Practica mucho**:
- Mínimo 10 veces
- Con diferentes personas

2. **Prepárate bien**:
- Conoce tu material
- Ten backup

3. **Visualiza éxito**:
- Imagina que va bien
- Respira profundamente

4. **Recuerda**:
- Es solo validación
- Feedback es valioso
- No es vida o muerte

---

### No sé cómo responder preguntas

**Síntomas**:
- Preguntas difíciles
- No tengo todas las respuestas

**Soluciones**:
1. **Prepárate**:
- Anticipa preguntas comunes
- Prepara respuestas

2. **Es OK no saber**:
- "Excelente pregunta, déjame investigar"
- "Eso es algo que estamos explorando"

3. **Sé honesto**:
- No inventes
- Admite limitaciones

---

## 🔍 Problemas de Análisis

### No sé cómo interpretar resultados

**Síntomas**:
- Métricas confusas
- No sé qué hacer

**Soluciones**:
1. **Usa el script de análisis**:
```bash
python scripts/analyze_metrics.py
```

2. **Compara con benchmarks**:
- Revisa [METRICS_GUIDE.md](METRICS_GUIDE.md)
- Compara con industria

3. **Busca ayuda**:
- Pregunta a mentores
- Comparte en comunidades
- Lee casos de estudio

---

### No sé si debo continuar

**Síntomas**:
- Métricas mixtas
- Feedback contradictorio

**Soluciones**:
1. **Revisa criterios**:
- [STRATEGY.md](STRATEGY.md) tiene criterios claros
- [CHECKLIST.md](CHECKLIST.md) tiene decisión final

2. **Más datos**:
- Más usuarios
- Más tiempo
- Más feedback

3. **Habla con mentores**:
- Perspectiva externa
- Experiencia previa

---

## 💡 Tips Generales

1. **Documenta problemas**: Anota qué pasó y cómo lo resolviste
2. **Pide ayuda**: No tengas miedo de preguntar
3. **Itera rápido**: Mejor probar que esperar
4. **No te rindas**: Problemas son normales
5. **Aprende**: Cada problema es una oportunidad

---

## 🆘 ¿Aún no funciona?

### Recursos Adicionales

- **Backend**: Revisa logs y documentación del proyecto principal
- **Frontend**: Revisa consola del navegador (F12)
- **Métricas**: Revisa [METRICS_GUIDE.md](METRICS_GUIDE.md)
- **Estrategia**: Revisa [STRATEGY.md](STRATEGY.md)

### Comunidades de Ayuda

- Reddit: r/startups, r/Entrepreneur
- Stack Overflow: Para problemas técnicos
- Discord/Slack: Comunidades de startups

---

**¿Problema no listado?** Documenta el problema y la solución para ayudar a otros! 🔧






