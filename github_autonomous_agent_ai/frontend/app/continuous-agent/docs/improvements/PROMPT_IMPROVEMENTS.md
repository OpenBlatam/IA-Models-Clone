# Prompt/Goal Field Improvements

## Overview

Se ha agregado soporte completo para prompts estilo Perplexity en el módulo continuous-agent. Los agentes ahora pueden tener un campo `goal` opcional que define su objetivo o prompt de sistema.

## Cambios Realizados

### 1. Tipos Actualizados

**`types/index.ts`**
- Agregado campo opcional `goal?: string` a `AgentConfig`

### 2. Constantes

**`constants/config.ts`**
- Agregado `GOAL: ""` a `FORM_DEFAULTS`

**`constants/prompt-templates.ts`** (NUEVO)
- Plantilla completa estilo Perplexity con todas las reglas de formato
- Plantillas simplificadas para diferentes casos de uso:
  - Research Assistant
  - Content Generator
  - Data Analyst
  - Custom (vacía para personalización)

### 3. Validación

**`utils/validation.ts`**
- Nueva función `validateGoal()` que valida el campo goal (opcional)
- Límite máximo de 10,000 caracteres

**`utils/validation/zod-schemas.ts`**
- Actualizado `agentConfigSchema` para incluir campo `goal` opcional

### 4. Componentes

**`components/forms/AgentGoalField.tsx`** (NUEVO)
- Componente completo para gestionar el campo goal
- Características:
  - Validación en tiempo real
  - Contador de caracteres con indicadores visuales
  - Selector de plantillas organizadas por categoría
  - Botón para limpiar el campo
  - Textarea grande (12 filas) para prompts largos

### 5. Hooks

**`hooks/useAgentForm.ts`**
- Agregado estado `goal` y `setGoal`
- Agregado validación de goal en `validate()`
- Actualizado `getConfig()` para incluir goal solo si no está vacío
- Actualizado `reset()` para limpiar goal

**`hooks/useFormFocus.ts`**
- Agregado referencia `goal` a `FieldRefs`
- Agregado "goal" al orden de campos

### 6. Modal

**`components/CreateAgentModal.tsx`**
- Agregado `AgentGoalField` al formulario
- Importado desde `./forms`

## Características

### Plantillas de Prompts

Las plantillas están organizadas en categorías:

- **Research**: Plantillas para investigación y búsqueda
  - Perplexity Base (completa)
  - Research Assistant (simplificada)

- **Content**: Plantillas para generación de contenido
  - Content Generator

- **Analysis**: Plantillas para análisis de datos
  - Data Analyst

- **Custom**: Plantilla vacía para personalización

### Validación

- Campo opcional (puede estar vacío)
- Máximo 10,000 caracteres
- Validación en tiempo real con feedback visual
- Indicadores de longitud (normal, advertencia, error)

### UI/UX

- Textarea grande para prompts largos
- Contador de caracteres visible
- Plantillas organizadas por categoría
- Botón para limpiar rápidamente
- Validación con mensajes de error claros

## Uso

### Crear un agente con goal

```typescript
const request: CreateAgentRequest = {
  name: "Research Agent",
  description: "Agent for research tasks",
  config: {
    taskType: "automated_research",
    frequency: 3600,
    parameters: {},
    goal: "<goal>You are a research assistant...</goal>"
  }
};
```

### Usar una plantilla

1. Abrir el modal de creación de agente
2. Hacer clic en "Plantillas" en el campo Goal
3. Seleccionar una plantilla de la categoría deseada
4. La plantilla se aplicará automáticamente
5. Personalizar según sea necesario

### Validación

El campo goal es completamente opcional. Si se proporciona:
- Debe ser una cadena de texto válida
- No puede exceder 10,000 caracteres
- Se valida en tiempo real mientras se escribe

## Compatibilidad

- ✅ Compatible con agentes existentes (campo opcional)
- ✅ No requiere cambios en el backend (se almacena en `config.goal`)
- ✅ Validación completa en frontend
- ✅ Type-safe con TypeScript

## Próximos Pasos

1. **Backend**: Asegurar que el backend acepte y almacene el campo `goal` en `config`
2. **Visualización**: Mostrar el goal en `AgentCard` si está presente
3. **Edición**: Permitir editar el goal en agentes existentes
4. **Templates**: Agregar más plantillas según necesidades

## Archivos Modificados

- `types/index.ts`
- `constants/config.ts`
- `constants/prompt-templates.ts` (NUEVO)
- `constants/index.ts`
- `utils/validation.ts`
- `utils/validation/zod-schemas.ts`
- `components/forms/AgentGoalField.tsx` (NUEVO)
- `components/forms/index.ts`
- `components/CreateAgentModal.tsx`
- `hooks/useAgentForm.ts`
- `hooks/useFormFocus.ts`

## Notas Técnicas

- El campo `goal` se almacena como parte de `config` en el backend
- Si el goal está vacío o es solo espacios en blanco, no se incluye en la configuración
- Las plantillas incluyen el prompt completo estilo Perplexity con todas las reglas de formato
- La validación usa debounce para evitar validaciones excesivas mientras se escribe


