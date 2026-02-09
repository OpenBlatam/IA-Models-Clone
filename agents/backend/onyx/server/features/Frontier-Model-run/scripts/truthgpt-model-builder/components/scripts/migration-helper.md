# Migration Helper - Guía de Uso de Scripts

## 🛠️ Scripts Disponibles

### 1. `analyze-unused-states.js`

**Propósito:** Analizar y encontrar estados no usados en ChatInterface.tsx

**Uso:**
```bash
# Desde el directorio del script
node scripts/analyze-unused-states.js

# O especificar ruta
node scripts/analyze-unused-states.js ../ChatInterface.tsx
```

**Output:**
- Lista de estados usados
- Lista de estados no usados (para eliminar)
- Lista de estados solo lectura
- Lista de estados solo escritura
- Lista de estados sospechosos
- Reporte en archivo `ChatInterface_STATE_ANALYSIS_REPORT.txt`

**Ejemplo de output:**
```
📊 Analizando: ChatInterface.tsx

📈 Total de estados encontrados: 200

📊 RESULTADOS DEL ANÁLISIS
════════════════════════════════════════════════════════════

✅ Estados USADOS: 30

❌ Estados NO USADOS: 120
   Estados que pueden ELIMINARSE:
   - workflowManager (línea 500)
   - integrationManager (línea 502)
   ...

💡 Potencial de reducción: 150 estados (75%)
```

---

### 2. `extract-hook-template.js`

**Propósito:** Generar template de hook desde estados existentes

**Uso:**
```bash
# Generar hook useSearch con estados searchQuery y filteredMessages
node scripts/extract-hook-template.js useSearch searchQuery filteredMessages

# Generar hook useMessageManagement con múltiples estados
node scripts/extract-hook-template.js useMessageManagement favoriteMessages pinnedMessages archivedMessages
```

**Output:**
- Crea `ChatInterface/hooks/[nombre-hook].ts`
- Crea `ChatInterface/hooks/__tests__/[nombre-hook].test.ts`
- Template completo con tipos y estructura

**Ejemplo:**
```bash
$ node scripts/extract-hook-template.js useSearch searchQuery filteredMessages

📝 Generando template para hook: useSearch
📦 Estados a incluir: searchQuery, filteredMessages

✅ Hook creado: ChatInterface/hooks/useSearch.ts
✅ Test creado: ChatInterface/hooks/__tests__/useSearch.test.ts

📝 Próximos pasos:
   1. Completar tipos TypeScript
   2. Implementar lógica del hook
   3. Escribir tests
   4. Integrar en componente principal
```

---

## 📋 Workflow Recomendado

### Paso 1: Análisis

```bash
# Analizar estados no usados
node scripts/analyze-unused-states.js

# Revisar reporte generado
cat ChatInterface_STATE_ANALYSIS_REPORT.txt
```

### Paso 2: Limpieza

1. Identificar estados no usados del reporte
2. Eliminar estados no usados manualmente (con cuidado)
3. Verificar que no hay errores

### Paso 3: Extracción de Hooks

```bash
# Extraer hook de búsqueda
node scripts/extract-hook-template.js useSearch searchQuery filteredMessages currentSearchIndex

# Extraer hook de gestión de mensajes
node scripts/extract-hook-template.js useMessageManagement favoriteMessages pinnedMessages archivedMessages

# Extraer hook de voz
node scripts/extract-hook-template.js useVoiceFeatures voiceInputEnabled isRecording dictationMode
```

### Paso 4: Completar Hooks

1. Abrir hook generado
2. Completar tipos TypeScript
3. Implementar lógica
4. Escribir tests
5. Integrar en componente

---

## 🔧 Personalización de Scripts

### Modificar analyze-unused-states.js

Puedes modificar el script para:
- Cambiar umbrales de "sospechoso"
- Agregar más análisis
- Exportar en diferentes formatos (JSON, CSV)

### Modificar extract-hook-template.js

Puedes modificar el template para:
- Agregar más estructura
- Incluir ejemplos de código
- Agregar documentación JSDoc

---

## 📊 Ejemplo de Workflow Completo

```bash
# 1. Analizar
node scripts/analyze-unused-states.js

# 2. Revisar reporte
# (Eliminar estados no usados manualmente)

# 3. Generar hooks
node scripts/extract-hook-template.js useChatState input isLoading messages error
node scripts/extract-hook-template.js useSearch searchQuery filteredMessages filterRole
node scripts/extract-hook-template.js useMessageManagement favoriteMessages pinnedMessages

# 4. Completar hooks generados
# (Editar archivos generados)

# 5. Integrar en componente
# (Seguir ejemplos de ChatInterface_REFACTORING_EXAMPLES.md)
```

---

## ⚠️ Notas Importantes

1. **Backup antes de ejecutar:** Siempre haz backup del archivo original
2. **Revisar manualmente:** Los scripts son herramientas, no reemplazan revisión humana
3. **Tests:** Siempre ejecuta tests después de cambios
4. **Incremental:** Haz cambios pequeños e incrementales

---

**Versión:** 1.0  
**Fecha:** 2024




