# Scripts de Refactorización - ChatInterface.tsx

Colección de scripts para automatizar y facilitar la refactorización del componente `ChatInterface.tsx`.

---

## 📋 Scripts Disponibles

### 1. `analyze-unused-states.js` 🔍
**Propósito:** Analiza estados no usados en el componente

**Uso:**
```bash
node scripts/analyze-unused-states.js
```

**Output:**
- Reporte de estados no usados
- Estados duplicados
- Estados obsoletos
- Archivo: `ChatInterface_STATE_ANALYSIS_REPORT.txt`

**Cuándo usar:**
- Al inicio de la refactorización
- Antes de eliminar código
- Para limpieza inicial

---

### 2. `extract-hook-template.js` 🪝
**Propósito:** Genera template de hook personalizado

**Uso:**
```bash
node scripts/extract-hook-template.js <hook-name> <state1> <state2> ...
```

**Ejemplo:**
```bash
node scripts/extract-hook-template.js useSearch searchQuery filteredMessages currentSearchIndex
```

**Output:**
- Archivo de hook con estructura completa
- Types e interfaces
- Template de tests
- Ubicación: `ChatInterface/hooks/<hook-name>.ts`

**Cuándo usar:**
- Al extraer lógica de useState/useEffect
- Al crear nuevos hooks personalizados
- Para mantener consistencia

---

### 3. `find-state-dependencies.js` 🔗
**Propósito:** Encuentra dependencias entre estados

**Uso:**
```bash
node scripts/find-state-dependencies.js [ruta-al-archivo]
```

**Output:**
- Grupos de estados relacionados
- Dependencias entre estados
- Sugerencias de consolidación
- Archivo: `ChatInterface_DEPENDENCIES_REPORT.txt`

**Cuándo usar:**
- Para identificar estados a consolidar
- Antes de usar useReducer
- Para entender relaciones entre estados

---

### 4. `extract-component.js` 🧩
**Propósito:** Extrae componentes de JSX a archivos separados

**Uso:**
```bash
node scripts/extract-component.js <component-name> <start-line> <end-line> [props...]
```

**Ejemplo:**
```bash
node scripts/extract-component.js MessageList 500 800 messages onMessageClick isFavorite
```

**Output:**
- Componente extraído
- Props interface
- Template de tests
- README del componente
- Ubicación: `ChatInterface/components/<component-name>/`

**Cuándo usar:**
- Al extraer bloques grandes de JSX
- Al crear componentes reutilizables
- Para reducir tamaño del componente principal

---

### 5. `validate-refactoring.js` ✅
**Propósito:** Valida que la refactorización esté correcta

**Uso:**
```bash
node scripts/validate-refactoring.js
```

**Output:**
- Verificación de estructura
- Validación de hooks y contexts
- Verificación de componentes
- Análisis de archivo principal
- Score de refactorización
- Archivo: `ChatInterface_VALIDATION_REPORT.txt`

**Cuándo usar:**
- Después de cada fase de refactorización
- Antes de hacer commit
- Para verificar progreso

---

### 6. `performance-analyzer.js` ⚡
**Propósito:** Analiza problemas de performance

**Uso:**
```bash
node scripts/performance-analyzer.js
```

**Output:**
- Análisis de re-renders
- Cálculos costosos sin memo
- Efectos problemáticos
- Sugerencias de optimización
- Score de performance
- Archivo: `ChatInterface_PERFORMANCE_REPORT.txt`

**Cuándo usar:**
- Después de refactorización
- Cuando hay problemas de performance
- Para identificar optimizaciones

---

### 7. `refactoring-assistant.js` 🤖
**Propósito:** Asistente interactivo paso a paso

**Uso:**
```bash
node scripts/refactoring-assistant.js
```

**Features:**
- Guía interactiva
- Análisis inicial
- Creación de estructura
- Extracción asistida de hooks/componentes/contexts
- Validación automática

**Cuándo usar:**
- Si eres nuevo en la refactorización
- Para guía paso a paso
- Para automatizar flujo completo

---

## 🚀 Flujo de Trabajo Recomendado

### Inicio de Refactorización

```bash
# 1. Análisis inicial
node scripts/analyze-unused-states.js
node scripts/find-state-dependencies.js

# 2. Revisar reportes
cat ChatInterface_STATE_ANALYSIS_REPORT.txt
cat ChatInterface_DEPENDENCIES_REPORT.txt

# 3. Usar asistente interactivo (opcional)
node scripts/refactoring-assistant.js
```

### Durante Refactorización

```bash
# Extraer hook
node scripts/extract-hook-template.js useSearch searchQuery filteredMessages

# Extraer componente
node scripts/extract-component.js MessageList 500 800 messages onMessageClick

# Validar progreso
node scripts/validate-refactoring.js
```

### Después de Refactorización

```bash
# Validación final
node scripts/validate-refactoring.js

# Análisis de performance
node scripts/performance-analyzer.js

# Revisar reportes
cat ChatInterface_VALIDATION_REPORT.txt
cat ChatInterface_PERFORMANCE_REPORT.txt
```

---

## 📊 Reportes Generados

Todos los scripts generan reportes en formato texto:

- `ChatInterface_STATE_ANALYSIS_REPORT.txt` - Análisis de estados
- `ChatInterface_DEPENDENCIES_REPORT.txt` - Dependencias entre estados
- `ChatInterface_VALIDATION_REPORT.txt` - Validación de refactorización
- `ChatInterface_PERFORMANCE_REPORT.txt` - Análisis de performance

---

## 🔧 Requisitos

- Node.js 14+
- Acceso al archivo `ChatInterface.tsx`
- Permisos de escritura para crear archivos

---

## ⚠️ Notas Importantes

1. **Backup:** Siempre haz backup antes de ejecutar scripts que modifican código
2. **Revisión:** Los scripts generan templates - siempre revisa y ajusta el código
3. **Tests:** Los templates incluyen tests básicos - complétalos según necesidad
4. **Validación:** Ejecuta `validate-refactoring.js` después de cambios importantes

---

## 🐛 Troubleshooting

### Error: "File not found"
- Verifica que estás en el directorio correcto
- Verifica que `ChatInterface.tsx` existe

### Error: "Permission denied"
- Verifica permisos de escritura
- Ejecuta con permisos adecuados

### Script no genera output esperado
- Revisa los argumentos pasados
- Verifica formato del archivo fuente
- Revisa logs de error

---

## 📚 Documentación Relacionada

- `ChatInterface_COMPLETE_GUIDE.md` - Guía completa
- `ChatInterface_MIGRATION_STEPS.md` - Pasos detallados
- `ChatInterface_REFACTORING_EXAMPLES.md` - Ejemplos de código
- `ChatInterface_BEST_PRACTICES.md` - Mejores prácticas

---

**Versión:** 1.0  
**Última actualización:** 2024




