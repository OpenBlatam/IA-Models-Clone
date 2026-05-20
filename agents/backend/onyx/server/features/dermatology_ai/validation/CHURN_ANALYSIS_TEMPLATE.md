# 📉 Template de Análisis de Churn - Validación Dermatology AI

Cómo analizar churn (pérdida de usuarios) para entender causas y reducir pérdidas.

## 🎯 Qué es Churn

Pérdida de usuarios que dejan de usar el producto.

**Tipos**:
- Churn de usuarios (dejan de usar)
- Churn de revenue (pierden ingresos)
- Churn voluntario (deciden irse)
- Churn involuntario (problemas técnicos)

---

## 📋 Métricas de Churn

### Tasa de Churn

**Fórmula**:
```
Churn Rate = Usuarios que se van / Usuarios totales
```

**Objetivo**: < 5% mensual

---

### Churn por Cohort

**Análisis**:
- Comparar churn entre cohortes
- Identificar tendencias
- Encontrar problemas

---

### Churn por Canal

**Análisis**:
- Comparar churn por canal
- Identificar canales de mejor calidad
- Optimizar adquisición

---

### Churn por Segmento

**Análisis**:
- Comparar churn por segmento
- Identificar segmentos de alto valor
- Enfocar en mejores segmentos

---

## 🔍 Causas Comunes

### 1. Problemas de Producto

**Causas**:
- No resuelve problema
- Muy complicado
- Bugs frecuentes
- Performance lenta

**Solución**:
- Mejorar producto
- Simplificar UX
- Arreglar bugs
- Optimizar performance

---

### 2. Problemas de Precio

**Causas**:
- Precio muy alto
- No ven valor
- Competencia más barata

**Solución**:
- Revisar precio
- Comunicar valor
- Mejorar propuesta

---

### 3. Problemas de Onboarding

**Causas**:
- Onboarding confuso
- No entienden valor
- No activan features

**Solución**:
- Mejorar onboarding
- Educar sobre valor
- Guiar activación

---

### 4. Problemas de Engagement

**Causas**:
- No usan producto
- No ven valor continuo
- Olvidan producto

**Solución**:
- Aumentar engagement
- Recordatorios
- Features que generen hábito

---

## 📝 Template de Análisis

```
═══════════════════════════════════════════════════════════════
                    ANÁLISIS DE CHURN
                    Dermatology AI
═══════════════════════════════════════════════════════════════

PERÍODO: [Fecha inicio] - [Fecha fin]
USUARIOS INICIALES: [X]
USUARIOS FINALES: [Y]
CHURN: [Z]

───────────────────────────────────────────────────────────────
1. TASA DE CHURN
───────────────────────────────────────────────────────────────

Churn Mensual: [X]% (Objetivo: < 5%)
Churn Anual: [Y]% (Objetivo: < 50%)

Tendencia:
- [Tendencia 1]
- [Tendencia 2]

───────────────────────────────────────────────────────────────
2. CHURN POR COHORT
───────────────────────────────────────────────────────────────

Cohorte | Tamaño | Churn | Tasa | Estado
--------|--------|-------|------|-------
Ene 2024| 100    | 15    | 15%  | ❌ Alto
Feb 2024| 150    | 10    | 7%   | ⚠️ Medio
Mar 2024| 200    | 8     | 4%   | ✅ Bajo

Análisis:
- [Tendencia 1]
- [Tendencia 2]

───────────────────────────────────────────────────────────────
3. CHURN POR CANAL
───────────────────────────────────────────────────────────────

Canal     | Usuarios | Churn | Tasa | Estado
----------|----------|-------|------|-------
SEO       | 100      | 5     | 5%   | ✅ Bajo
Paid Ads  | 150      | 12    | 8%   | ⚠️ Medio
Social    | 80       | 10    | 13%  | ❌ Alto

Análisis:
- [Insight 1]
- [Insight 2]

───────────────────────────────────────────────────────────────
4. CHURN POR SEGMENTO
───────────────────────────────────────────────────────────────

Segmento      | Usuarios | Churn | Tasa | Estado
--------------|----------|-------|------|-------
Premium       | 50       | 2     | 4%   | ✅ Bajo
Free          | 200      | 15    | 8%   | ⚠️ Medio
Trial         | 100      | 20    | 20%  | ❌ Alto

Análisis:
- [Insight 1]
- [Insight 2]

───────────────────────────────────────────────────────────────
5. CAUSAS DE CHURN
───────────────────────────────────────────────────────────────

Causa              | Frecuencia | %    | Prioridad
-------------------|------------|------|----------
Precio muy alto    | 20         | 40%  | Alta
No resuelve problema| 15        | 30%  | Alta
Muy complicado     | 10         | 20%  | Media
Problemas técnicos | 5          | 10%  | Baja

Análisis:
- [Causa principal]
- [Causa secundaria]

───────────────────────────────────────────────────────────────
6. MOMENTO DE CHURN
───────────────────────────────────────────────────────────────

Momento        | Churn | %    | Insight
---------------|-------|------|--------
Primera semana | 25    | 50%  | Onboarding
Primer mes     | 15    | 30%  | Activación
Después        | 10    | 20%  | Engagement

Análisis:
- [Insight 1]
- [Insight 2]

───────────────────────────────────────────────────────────────
7. ACCIONES
───────────────────────────────────────────────────────────────

Corto Plazo:
- [ ] [Acción 1] - [Causa]
- [ ] [Acción 2] - [Causa]

Mediano Plazo:
- [ ] [Acción 1] - [Causa]
- [ ] [Acción 2] - [Causa]

───────────────────────────────────────────────────────────────
8. MÉTRICAS DE ÉXITO
───────────────────────────────────────────────────────────────

Objetivo: Reducir churn a [X]% en [Y] meses

Métricas:
- Churn actual: [A]%
- Churn objetivo: [B]%
- Reducción necesaria: [C]%
```

---

## 💡 Tips

1. **Mide regularmente**: Mensual o semanal
2. **Analiza causas**: No solo números
3. **Segmenta**: Diferentes causas por segmento
4. **Acciona**: Implementa soluciones
5. **Mide impacto**: Verifica mejoras

---

## 🎯 Checklist

- [ ] Tasa de churn calculada
- [ ] Análisis por cohort completado
- [ ] Análisis por canal completado
- [ ] Análisis por segmento completado
- [ ] Causas identificadas
- [ ] Acciones planificadas
- [ ] Métricas de éxito definidas

---

**Próximo paso**: Analiza tu churn usando este template! 📉






