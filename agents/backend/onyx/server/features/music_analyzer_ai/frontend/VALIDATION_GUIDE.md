# Guía de Validaciones

## 📋 Overview

El sistema de validaciones utiliza **Zod** para validación en tiempo de ejecución con type safety completo.

## 🎯 Características

- ✅ Validación en tiempo de ejecución con Zod
- ✅ Type safety completo
- ✅ Mensajes de error personalizados
- ✅ Validación de formularios con hooks
- ✅ Validación automática en API calls
- ✅ Componentes de formulario con validación integrada

## 📚 Esquemas de Validación

### Validaciones Comunes

```typescript
import {
  emailSchema,
  urlSchema,
  nonEmptyStringSchema,
  positiveIntegerSchema,
  stringWithLengthRange,
} from '@/lib/validations';

// Email
const email = emailSchema.parse('user@example.com');

// URL
const url = urlSchema.parse('https://example.com');

// String no vacío
const name = nonEmptyStringSchema.parse('John Doe');

// Entero positivo
const age = positiveIntegerSchema.parse(25);

// String con rango
const title = stringWithLengthRange(1, 100).parse('My Title');
```

### Validaciones de Música

```typescript
import {
  searchQuerySchema,
  trackIdSchema,
  trackIdsSchema,
  analyzeTrackRequestSchema,
  compareTracksRequestSchema,
} from '@/lib/validations';

// Búsqueda
const query = searchQuerySchema.parse('artist name');

// Track ID
const trackId = trackIdSchema.parse('spotify:track:123');

// Análisis de track
const analysisRequest = analyzeTrackRequestSchema.parse({
  trackId: 'spotify:track:123',
  includeCoaching: true,
});

// Comparación de tracks
const comparisonRequest = compareTracksRequestSchema.parse({
  trackIds: ['id1', 'id2'],
  comparisonType: 'all',
});
```

## 🎣 Hook de Validación de Formularios

### Uso Básico

```typescript
'use client';

import { useFormValidation } from '@/lib/hooks';
import { searchRequestSchema } from '@/lib/validations';

function SearchForm() {
  const {
    values,
    errors,
    touched,
    isValid,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
  } = useFormValidation({
    schema: searchRequestSchema,
    initialValues: { query: '', limit: 10 },
    onSubmit: async (data) => {
      // Handle submission
      console.log('Valid data:', data);
    },
    validateOnBlur: true,
  });

  return (
    <form onSubmit={handleSubmit}>
      <input
        value={values.query || ''}
        onChange={(e) => handleChange('query')(e.target.value)}
        onBlur={handleBlur('query')}
      />
      {touched.query && errors.query && (
        <div className="text-red-400">
          {errors.query[0]}
        </div>
      )}
      <button type="submit" disabled={!isValid || isSubmitting}>
        Search
      </button>
    </form>
  );
}
```

### Con Componente FormField

```typescript
'use client';

import { useFormValidation } from '@/lib/hooks';
import { FormField } from '@/components/ui';
import { searchRequestSchema } from '@/lib/validations';

function SearchForm() {
  const {
    values,
    errors,
    touched,
    handleChange,
    handleBlur,
    handleSubmit,
  } = useFormValidation({
    schema: searchRequestSchema,
    initialValues: { query: '', limit: 10 },
    onSubmit: async (data) => {
      await searchTracks(data.query, data.limit);
    },
  });

  return (
    <form onSubmit={handleSubmit}>
      <FormField
        label="Search Query"
        value={values.query || ''}
        onChange={(e) => handleChange('query')(e.target.value)}
        onBlur={handleBlur('query')}
        errors={errors.query}
        touched={touched.query}
        required
      />
      <button type="submit">Search</button>
    </form>
  );
}
```

## 🔧 Utilidades de Validación

### Validar y Lanzar Error

```typescript
import { validateOrThrow } from '@/lib/utils/validation';
import { searchQuerySchema } from '@/lib/validations';

try {
  const query = validateOrThrow(searchQuerySchema, userInput, 'query');
  // query is now type-safe
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Validation failed:', error.message);
  }
}
```

### Validación Segura

```typescript
import { safeValidate } from '@/lib/utils/validation';
import { trackIdSchema } from '@/lib/validations';

const result = safeValidate(trackIdSchema, userInput);

if (result.success) {
  // result.data is type-safe
  console.log('Valid track ID:', result.data);
} else {
  // result.errors contains ZodError
  console.error('Validation errors:', result.errors);
}
```

### Verificar Validez

```typescript
import { isValid } from '@/lib/utils/validation';
import { emailSchema } from '@/lib/validations';

if (isValid(emailSchema, userInput)) {
  // TypeScript knows userInput is a valid email string
  sendEmail(userInput);
}
```

### Obtener Errores de Campo

```typescript
import { getFieldErrors } from '@/lib/utils/validation';

const errors = getFieldErrors(zodError, 'user.email');
// Returns: ['Invalid email address']
```

## 🎨 Componente FormField

El componente `FormField` proporciona:

- ✅ Label con indicador de requerido
- ✅ Estilos consistentes
- ✅ Muestra errores automáticamente
- ✅ Helper text opcional
- ✅ Accesibilidad (ARIA)

```typescript
import { FormField } from '@/components/ui';

<FormField
  label="Email Address"
  type="email"
  value={email}
  onChange={(e) => setEmail(e.target.value)}
  errors={errors.email}
  touched={touched.email}
  helperText="We'll never share your email"
  required
/>
```

## 🔄 Validación en API Calls

Las funciones de API validan automáticamente:

```typescript
import { searchTracks, analyzeTrack } from '@/lib/api';

// Validación automática de input
try {
  const results = await searchTracks('query', 10);
  // Validación automática de response
} catch (error) {
  if (error instanceof ValidationError) {
    // Error de validación
    console.error('Invalid input:', error.message);
  }
}

// Validación automática de opciones
try {
  const analysis = await analyzeTrack({
    trackId: 'spotify:track:123',
    includeCoaching: true,
  });
} catch (error) {
  // Manejo de errores
}
```

## 📝 Crear Nuevos Esquemas

### Esquema Simple

```typescript
import { z } from 'zod';
import { stringWithLengthRange } from '@/lib/validations/common';

export const myFieldSchema = stringWithLengthRange(
  1,
  50,
  'Field must be at least 1 character',
  'Field must be at most 50 characters'
);
```

### Esquema Complejo

```typescript
import { z } from 'zod';
import { nonEmptyStringSchema, positiveIntegerSchema } from '@/lib/validations/common';

export const myFormSchema = z.object({
  name: nonEmptyStringSchema,
  age: positiveIntegerSchema,
  email: z.string().email(),
}).refine(
  (data) => data.age >= 18,
  {
    message: 'Must be at least 18 years old',
    path: ['age'],
  }
);
```

## 🐛 Debugging

### Ver Errores de Validación

```typescript
import { formatZodErrors } from '@/lib/utils/validation';

try {
  schema.parse(data);
} catch (error) {
  if (error instanceof z.ZodError) {
    const formatted = formatZodErrors(error);
    console.log('Validation errors:', formatted);
    // {
    //   'user.email': ['Invalid email'],
    //   'user.age': ['Must be positive']
    // }
  }
}
```

### Validación Paso a Paso

```typescript
const result = schema.safeParse(data);

if (!result.success) {
  result.error.errors.forEach((err) => {
    console.log('Path:', err.path);
    console.log('Message:', err.message);
    console.log('Code:', err.code);
  });
}
```

## ✅ Mejores Prácticas

1. **Usa esquemas compartidos**: Reutiliza esquemas comunes
2. **Valida temprano**: Valida en el input, no solo en submit
3. **Mensajes claros**: Proporciona mensajes de error descriptivos
4. **Type safety**: Aprovecha el type inference de Zod
5. **Validación en ambos lados**: Valida en cliente y servidor
6. **Manejo de errores**: Usa try-catch para errores de validación

## 🔗 Recursos

- [Zod Documentation](https://zod.dev/)
- [React Hook Form + Zod](https://react-hook-form.com/get-started#SchemaValidation)
- [TypeScript + Zod](https://zod.dev/?id=typescript)

