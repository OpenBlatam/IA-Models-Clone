# Guía de Migración - Nueva Estructura Modular

## 📋 Cambios Principales

### 1. Model Files → `modules/management/`

**Antes:**
```typescript
import { analyzeModelDescription } from '@/lib/model-analyzer'
import { ModelManager } from '@/lib/model-manager'
import { ModelSpec } from '@/lib/model-analyzer'
```

**Después:**
```typescript
import { analyzeModelDescription, ModelSpec } from '@/lib/modules/management'
import { ModelManager } from '@/lib/modules/management'
```

### 2. TruthGPT Adaptation → `modules/adaptation/`

**Antes:**
```typescript
import { adaptToTruthGPT } from '@/lib/truthgpt-adapter'
import { integrateWithTruthGPT } from '@/lib/truthgpt-integrator'
```

**Después:**
```typescript
import { adaptToTruthGPT, integrateWithTruthGPT } from '@/lib/modules/adaptation'
```

### 3. Cache → `modules/storage/`

**Antes:**
```typescript
import { Cache } from '@/lib/cache'
// o
import { SmartCache } from '@/lib/smart-cache'
```

**Después:**
```typescript
import { UnifiedCache, createCache } from '@/lib/modules/storage'
```

### 4. ChatInterface → Estructura Modular

**Antes:**
```typescript
import ChatInterface from '@/components/ChatInterface'
```

**Después:**
```typescript
// Sigue funcionando igual, pero internamente está modularizado
import ChatInterface from '@/components/ChatInterface'
// O usar directamente la nueva estructura
import ChatInterface from '@/components/ChatInterface/index'
```

## 🔄 Mapeo de Imports

| Antes | Después |
|-------|---------|
| `@/lib/model-analyzer` | `@/lib/modules/management` |
| `@/lib/model-manager` | `@/lib/modules/management` |
| `@/lib/model-optimizer` | `@/lib/modules/management` |
| `@/lib/model-validator` | `@/lib/modules/management` |
| `@/lib/model-templates` | `@/lib/modules/management` |
| `@/lib/model-versioning` | `@/lib/modules/management` |
| `@/lib/model-exporter` | `@/lib/modules/management` |
| `@/lib/truthgpt-adapter` | `@/lib/modules/adaptation` |
| `@/lib/truthgpt-integrator` | `@/lib/modules/adaptation` |
| `@/lib/cache` | `@/lib/modules/storage` |
| `@/lib/smart-cache` | `@/lib/modules/storage` |
| `@/lib/advanced-cache` | `@/lib/modules/storage` |

## ✅ Compatibilidad

Los archivos antiguos en el root level **aún existen** pero están **deprecados**. Funcionarán pero mostrarán warnings. Se recomienda migrar a la nueva estructura.

## 📝 Ejemplos de Migración

### Ejemplo 1: Usar Model Analyzer

```typescript
// ❌ Antes
import { analyzeModelDescription, ModelSpec } from '@/lib/model-analyzer'

// ✅ Después
import { analyzeModelDescription, ModelSpec } from '@/lib/modules/management'
```

### Ejemplo 2: Usar TruthGPT Adapter

```typescript
// ❌ Antes
import { adaptToTruthGPT } from '@/lib/truthgpt-adapter'

// ✅ Después
import { adaptToTruthGPT } from '@/lib/modules/adaptation'
```

### Ejemplo 3: Usar Cache

```typescript
// ❌ Antes
import { Cache } from '@/lib/cache'

// ✅ Después
import { UnifiedCache, createCache } from '@/lib/modules/storage'

// Uso
const cache = createCache({ maxSize: 100, strategy: 'lru' })
```

## 🚨 Archivos Deprecados

Los siguientes archivos están deprecados pero aún funcionan:

- `lib/model-analyzer.ts` → Usar `modules/management`
- `lib/model-manager.ts` → Usar `modules/management`
- `lib/model-optimizer.ts` → Usar `modules/management`
- `lib/model-validator.ts` → Usar `modules/management`
- `lib/model-templates.ts` → Usar `modules/management`
- `lib/model-versioning.ts` → Usar `modules/management`
- `lib/model-exporter.ts` → Usar `modules/management`
- `lib/truthgpt-adapter.ts` → Usar `modules/adaptation`
- `lib/truthgpt-integrator.ts` → Usar `modules/adaptation`
- `lib/cache.ts` → Usar `modules/storage`
- `lib/github-service.ts` → Ya re-exporta desde `modules/github`

## 🔍 Cómo Encontrar Imports a Actualizar

Busca en tu código:

```bash
# Buscar imports antiguos
grep -r "from '@/lib/model-" .
grep -r "from '@/lib/truthgpt-adapter" .
grep -r "from '@/lib/cache" .
```

## 📚 Estructura de Módulos

```
lib/modules/
├── management/      # Gestión de modelos
│   ├── model-analyzer.ts
│   ├── model-manager.ts
│   ├── model-optimizer.ts
│   ├── model-validator.ts
│   ├── model-templates.ts
│   ├── model-versioning.ts
│   ├── model-exporter.ts
│   └── index.ts
├── adaptation/      # Adaptación a TruthGPT
│   ├── truthgpt-adapter.ts
│   ├── truthgpt-integrator.ts
│   └── index.ts
├── storage/         # Almacenamiento y cache
│   ├── cache.ts
│   └── index.ts
└── github/          # Integración con GitHub
    ├── github-service.ts
    └── index.ts
```

## ⚡ Beneficios de la Nueva Estructura

1. **Organización Clara**: Archivos agrupados por funcionalidad
2. **Fácil de Encontrar**: Todo relacionado está en el mismo módulo
3. **Mejor Mantenibilidad**: Cambios aislados por módulo
4. **Type Safety**: Mejor soporte de TypeScript
5. **Escalabilidad**: Fácil agregar nuevos módulos

## 🆘 Soporte

Si encuentras problemas con la migración:

1. Revisa los ejemplos en esta guía
2. Consulta `REFACTORING_FINAL.md` para detalles técnicos
3. Los archivos antiguos aún funcionan como fallback

---

**¡Migración completada exitosamente! 🎉**

