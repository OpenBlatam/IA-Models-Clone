# 📈 Template de Análisis de Cohortes - Validación Dermatology AI

Cómo analizar comportamiento de usuarios por cohortes para entender retención y valor.

## 🎯 Qué es Análisis de Cohortes

Análisis de grupos de usuarios que comparten característica común (fecha de registro, canal, etc.) para entender:
- Retención
- Valor a lo largo del tiempo
- Comportamiento
- Tendencias

---

## 📋 Tipos de Cohortes

### 1. Por Fecha

**Cohortes**:
- Por mes de registro
- Por semana de registro
- Por día de registro

**Uso**: Entender retención y valor a lo largo del tiempo

---

### 2. Por Canal

**Cohortes**:
- SEO
- Paid ads
- Social media
- Referrals

**Uso**: Comparar calidad de canales

---

### 3. Por Segmento

**Cohortes**:
- Usuarios premium
- Usuarios free
- Por geografía
- Por tipo de usuario

**Uso**: Entender diferencias entre segmentos

---

## 📊 Métricas de Cohortes

### Retención

**Qué mide**: % de usuarios que vuelven

**Fórmula**:
```
Retención D1 = Usuarios activos D1 / Usuarios totales
Retención D7 = Usuarios activos D7 / Usuarios totales
Retención D30 = Usuarios activos D30 / Usuarios totales
```

**Objetivos**:
- D1: > 40%
- D7: > 20%
- D30: > 10%

---

### Revenue por Cohort

**Qué mide**: Revenue generado por cohorte

**Fórmula**:
```
Revenue Cohort = Suma de revenue de usuarios en cohorte
```

**Uso**: Comparar valor de diferentes cohortes

---

### LTV por Cohort

**Qué mide**: Lifetime Value por cohorte

**Fórmula**:
```
LTV Cohort = Revenue promedio por usuario en cohorte
```

**Uso**: Comparar valor de diferentes cohortes

---

## 📝 Template de Análisis

```
═══════════════════════════════════════════════════════════════
                    ANÁLISIS DE COHORTES
                    Dermatology AI
═══════════════════════════════════════════════════════════════

TIPO: [Por fecha / Canal / Segmento]
PERÍODO: [Fecha inicio] - [Fecha fin]

───────────────────────────────────────────────────────────────
1. RETENCIÓN POR COHORTE
───────────────────────────────────────────────────────────────

Cohorte | Tamaño | D1  | D7  | D14 | D30 | D60 | D90
---------|--------|-----|-----|-----|-----|-----|-----
Ene 2024 | 100    | 45% | 25% | 18% | 12% | 8%  | 5%
Feb 2024 | 150    | 50% | 28% | 20% | 14% | 10% | 7%
Mar 2024 | 200    | 48% | 26% | 19% | 13% | 9%  | 6%

Análisis:
- [Tendencia 1]
- [Tendencia 2]

───────────────────────────────────────────────────────────────
2. REVENUE POR COHORTE
───────────────────────────────────────────────────────────────

Cohorte | Mes 1 | Mes 2 | Mes 3 | Total | LTV
---------|-------|-------|-------|-------|-----
Ene 2024 | $500  | $300  | $200  | $1,000| $10
Feb 2024 | $750  | $450  | $300  | $1,500| $10
Mar 2024 | $1,000| $600  | $400  | $2,000| $10

Análisis:
- [Tendencia 1]
- [Tendencia 2]

───────────────────────────────────────────────────────────────
3. COMPARACIÓN DE CANALES
───────────────────────────────────────────────────────────────

Canal     | Tamaño | D1  | D7  | D30 | LTV  | CAC
----------|--------|-----|-----|-----|------|-----
SEO       | 100    | 50% | 28% | 14% | $12  | $5
Paid Ads  | 150    | 45% | 25% | 12% | $10  | $20
Social    | 80     | 40% | 22% | 10% | $8   | $15

Análisis:
- [Insight 1]
- [Insight 2]

───────────────────────────────────────────────────────────────
4. TENDENCIAS
───────────────────────────────────────────────────────────────

Retención:
- [Tendencia 1]
- [Tendencia 2]

Revenue:
- [Tendencia 1]
- [Tendencia 2]

───────────────────────────────────────────────────────────────
5. INSIGHTS Y ACCIONES
───────────────────────────────────────────────────────────────

Insights:
- [Insight 1]
- [Insight 2]

Acciones:
- [ ] [Acción 1]
- [ ] [Acción 2]
```

---

## 🛠️ Herramientas

### Manuales

**Excel/Google Sheets**:
- Crear tablas
- Calcular métricas
- Visualizar

**Python/R**:
- Análisis avanzado
- Visualizaciones
- Automatización

---

### Automatizadas

**Analytics**:
- Google Analytics
- Mixpanel
- Amplitude

**BI Tools**:
- Tableau
- Looker
- Metabase

---

## 💡 Tips

1. **Sé consistente**: Mismo método siempre
2. **Sé paciente**: Datos toman tiempo
3. **Compara**: Entre cohortes
4. **Visualiza**: Gráficos ayudan
5. **Acciona**: Usa insights

---

## 🎯 Checklist

- [ ] Tipo de cohorte definido
- [ ] Métricas calculadas
- [ ] Análisis completado
- [ ] Insights identificados
- [ ] Acciones planificadas

---

**Próximo paso**: Analiza tus cohortes usando este template! 📈






