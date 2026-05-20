# ❓ FAQ - Validación Dermatology AI

Preguntas frecuentes sobre el proceso de validación.

## 🚀 Inicio

### ¿Por dónde empiezo?

**Respuesta**: 
Empieza con [START_HERE.md](START_HERE.md) para una guía de inicio rápido, o [QUICK_GUIDE.md](QUICK_GUIDE.md) para validar en 1 hora.

**Pasos rápidos**:
1. Lee [START_HERE.md](START_HERE.md) (5 min)
2. Prueba el frontend tú mismo (5 min)
3. Comparte con 3 personas (10 min)

---

### ¿Cuánto tiempo toma validar?

**Respuesta**:
- **Mínimo**: 1 hora (prueba básica)
- **Básico**: 1 semana (validación inicial)
- **Completo**: 2-4 semanas (validación profunda)

Depende de qué tan profundo quieres ir.

---

### ¿Necesito saber programar?

**Respuesta**:
No necesariamente. El frontend ya está listo, solo necesitas:
- El backend corriendo (ya está configurado)
- Abrir el frontend en el navegador
- Compartir el link

Si quieres personalizar, ayuda saber HTML/CSS/JS básico.

---

## 🛠️ Técnico

### ¿Cómo inicio el backend?

**Respuesta**:
```bash
cd agents/backend/onyx/server/features/dermatology_ai
python main.py
```

El servidor estará en `http://localhost:8006`

---

### ¿Cómo abro el frontend?

**Respuesta**:
**Opción 1** (Simple):
- Doble clic en `frontend/index.html`

**Opción 2** (Recomendado):
```bash
cd validation/frontend
python -m http.server 8080
```
Luego abre: `http://localhost:8080`

---

### El backend no responde, ¿qué hago?

**Respuesta**:
1. Verifica que esté corriendo:
```bash
curl http://localhost:8006/health
```

2. Si no responde, reinicia:
```bash
python main.py
```

3. Verifica que el puerto 8006 no esté ocupado

Ver más en [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

### ¿Puedo usar esto en producción?

**Respuesta**:
El frontend de validación es para **validación**, no producción. Para producción:
- Usa el frontend completo del proyecto principal
- O crea uno nuevo basado en este

---

## 👥 Usuarios

### ¿Dónde consigo usuarios para probar?

**Respuesta**:
1. **Círculo cercano**: Amigos, familia, colegas
2. **Redes sociales**: Twitter, Reddit, Facebook Groups
3. **Comunidades online**: Grupos de interés, foros
4. **Eventos**: Networking, meetups

Ver [SOCIAL_MEDIA_GUIDE.md](SOCIAL_MEDIA_GUIDE.md) para más ideas.

---

### ¿Cuántos usuarios necesito?

**Respuesta**:
- **Mínimo**: 5-10 usuarios
- **Ideal**: 20-50 usuarios
- **Profundo**: 50-100+ usuarios

Más usuarios = más confianza en resultados, pero calidad > cantidad.

---

### ¿Qué si nadie quiere probar?

**Respuesta**:
1. **Revisa tu mensaje**: ¿Es claro? ¿Ofrece valor?
2. **Empieza con cercanos**: Más dispuestos a ayudar
3. **Ofrece incentivo**: Acceso gratis, descuento futuro
4. **Sé específico**: "Solo toma 2 minutos"

Ver [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para más soluciones.

---

## 📊 Métricas

### ¿Qué métricas debo medir?

**Respuesta**:
**Mínimas**:
- Visitas
- Conversiones (subidas de foto)
- Feedback promedio
- % que pagarían

**Completas**:
- Funnel completo
- NPS
- Retención
- Tiempo de análisis

Ver [METRICS_GUIDE.md](METRICS_GUIDE.md) para guía completa.

---

### ¿Cómo recopilo métricas?

**Respuesta**:
**Opción 1** (Manual):
```bash
python scripts/collect_metrics.py
```

**Opción 2** (Automático):
- Google Analytics
- Mixpanel
- Hotjar

Ver [METRICS_GUIDE.md](METRICS_GUIDE.md) para más opciones.

---

### ¿Qué métricas son buenas?

**Respuesta**:
**Buenas**:
- Tasa de conversión > 20%
- Satisfacción > 3.5/5
- NPS > 30
- Intención de pago > 30%

Ver [STRATEGY.md](STRATEGY.md) para criterios completos.

---

## 💰 Precios

### ¿Cómo defino el precio?

**Respuesta**:
1. **Analiza competencia**: ¿Qué cobran otros?
2. **Valida con usuarios**: Pregunta en entrevistas
3. **Calcula costos**: Asegúrate de ser rentable
4. **Prueba diferentes precios**: A/B testing

Ver [PRICING_STRATEGY.md](PRICING_STRATEGY.md) para guía completa.

---

### ¿Qué si dicen que es muy caro?

**Respuesta**:
1. **Comunica valor mejor**: ¿Qué ahorran? ¿Qué obtienen?
2. **Considera freemium**: Versión gratis + premium
3. **Valida precio específico**: "¿Cuánto pagarías?"
4. **Revisa costos**: ¿Puedes reducir precio?

---

## 🎯 Decisión

### ¿Cómo sé si debo continuar?

**Respuesta**:
**Continúa si**:
- Tasa de conversión > 20%
- Satisfacción > 3.5/5
- NPS > 30
- Intención de pago > 30%

**Itera si**:
- Métricas mixtas pero hay interés
- Problemas técnicos solucionables

**Pivotar si**:
- Métricas muy bajas
- No hay interés
- Feedback muy negativo

Ver [STRATEGY.md](STRATEGY.md) para criterios completos.

---

### ¿Qué si el feedback es negativo?

**Respuesta**:
1. **No te desanimes**: Feedback negativo es valioso
2. **Identifica patrones**: ¿Qué se critica más?
3. **Pregunta específicamente**: "¿Qué mejorarías?"
4. **Itera rápido**: Haz cambios y prueba de nuevo

Mejor saberlo temprano que tarde.

---

## 📱 Marketing

### ¿Dónde comparto el producto?

**Respuesta**:
1. **Reddit**: r/SkincareAddiction, r/startups
2. **Twitter**: Posts con hashtags relevantes
3. **Facebook Groups**: Grupos de interés
4. **LinkedIn**: Para audiencia profesional

Ver [SOCIAL_MEDIA_GUIDE.md](SOCIAL_MEDIA_GUIDE.md) para guía completa.

---

### ¿Qué mensaje uso?

**Respuesta**:
Usa las plantillas de [TEMPLATES.md](TEMPLATES.md) o [SOCIAL_MEDIA_GUIDE.md](SOCIAL_MEDIA_GUIDE.md).

**Elementos clave**:
- Claro y específico
- Ofrece valor
- CTA claro
- No suena como spam

---

## 🎤 Presentación

### ¿Cómo hago un pitch?

**Respuesta**:
1. **Usa el template**: [PITCH_DECK_TEMPLATE.md](PITCH_DECK_TEMPLATE.md)
2. **Sigue la guía**: [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)
3. **Practica mucho**: Mínimo 10 veces

---

### ¿Qué incluyo en el pitch?

**Respuesta**:
**Mínimo**:
- Problema
- Solución
- Tracción
- CTA

**Completo**:
- 12 slides según [PITCH_DECK_TEMPLATE.md](PITCH_DECK_TEMPLATE.md)

---

## 🔧 Problemas

### ¿Dónde encuentro ayuda con problemas técnicos?

**Respuesta**:
1. **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Logs del backend**: Revisa errores
3. **Consola del navegador**: F12 para ver errores
4. **Comunidades**: Stack Overflow, Reddit

---

### ¿Qué si algo no funciona?

**Respuesta**:
1. **Revisa troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Documenta el problema**: Anota qué pasó
3. **Busca ayuda**: Comunidades, mentores
4. **Itera**: Prueba soluciones alternativas

---

## 📚 Recursos

### ¿Qué archivo debo leer primero?

**Respuesta**:
1. **[START_HERE.md](START_HERE.md)** - Si quieres empezar rápido
2. **[QUICK_GUIDE.md](QUICK_GUIDE.md)** - Si quieres validar en 1 hora
3. **[INDEX.md](INDEX.md)** - Si quieres ver todos los recursos

---

### ¿Hay un checklist?

**Respuesta**:
Sí, [CHECKLIST.md](CHECKLIST.md) tiene un checklist completo paso a paso.

---

### ¿Dónde encuento plantillas?

**Respuesta**:
- **Mensajes**: [TEMPLATES.md](TEMPLATES.md)
- **Emails**: [EMAIL_TEMPLATES.md](EMAIL_TEMPLATES.md)
- **Redes sociales**: [SOCIAL_MEDIA_GUIDE.md](SOCIAL_MEDIA_GUIDE.md)
- **Pitch deck**: [PITCH_DECK_TEMPLATE.md](PITCH_DECK_TEMPLATE.md)

---

## 💡 Tips Generales

### ¿Cuál es el error más común?

**Respuesta**:
Construir sin validar. Valida primero, construye después.

---

### ¿Qué hago si me siento abrumado?

**Respuesta**:
1. **Empieza pequeño**: Un paso a la vez
2. **Usa quick wins**: [QUICK_WINS.md](QUICK_WINS.md)
3. **Enfócate**: Una cosa a la vez
4. **Pide ayuda**: No tengas miedo

---

### ¿Cuánto tiempo debo dedicar?

**Respuesta**:
- **Mínimo**: 1 hora esta semana
- **Ideal**: 2-3 horas esta semana
- **Completo**: 5-10 horas en 2-4 semanas

Consistencia > Intensidad.

---

## 🆘 ¿Pregunta no respondida?

1. **Revisa el índice**: [INDEX.md](INDEX.md)
2. **Busca en archivos**: Usa búsqueda en tu editor
3. **Documenta la pregunta**: Ayuda a mejorar el FAQ

---

**¿Tienes más preguntas?** Documenta tu pregunta y respuesta para ayudar a otros! ❓






