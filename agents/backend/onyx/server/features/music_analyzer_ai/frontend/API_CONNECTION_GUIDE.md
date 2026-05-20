# Guía de Conexión de la API

## 🔌 Configuración de la Conexión

### Variables de Entorno

Crea un archivo `.env.local` en la raíz del proyecto:

```env
NEXT_PUBLIC_MUSIC_API_URL=http://localhost:8010
NEXT_PUBLIC_ROBOT_API_URL=http://localhost:8010
```

### Verificación de la Configuración

La configuración se valida automáticamente al iniciar la aplicación. Si hay problemas, verás errores en la consola.

Para verificar manualmente:

```typescript
import { validateApiConfig, getApiConnectionInfo } from '@/lib/api';

// Validar configuración
const validation = validateApiConfig();
if (!validation.isValid) {
  console.error('Problemas de configuración:', validation.issues);
}

// Ver información de conexión
const info = getApiConnectionInfo();
console.log('Configuración API:', info);
```

## 🏥 Health Check

### Verificación Automática

El componente `ApiStatus` verifica automáticamente el estado de la API cada 30 segundos.

### Verificación Manual

```typescript
import { checkApiHealth } from '@/lib/api';

const health = await checkApiHealth();
console.log(health.status); // 'healthy' | 'unhealthy'
console.log(health.message);
```

### Prueba de Conexión

```typescript
import { testApiConnection } from '@/lib/api';

const test = await testApiConnection();
console.log('Éxito:', test.success);
console.log('Tiempo de respuesta:', test.responseTime, 'ms');
console.log('Mensaje:', test.message);
```

## 🔄 Reintentos Automáticos

El cliente API incluye lógica de reintentos automáticos:

- **Reintentos por defecto**: 2
- **Backoff exponencial**: 1000ms * 2^intento
- **Solo reintenta en**: Errores de red o errores 5xx del servidor

### Personalizar Reintentos

```typescript
import { requestWithRetry } from '@/lib/api/client';

const result = await requestWithRetry(
  () => musicApiClient.get('/endpoint'),
  {
    retries: 3,
    retryDelay: 2000,
  }
);
```

## 🚨 Manejo de Errores

### Tipos de Error

```typescript
import { ApiError, NetworkError, ValidationError } from '@/lib/api';

try {
  await searchTracks('query');
} catch (error) {
  if (error instanceof NetworkError) {
    // Error de conexión
    console.error('Problema de red:', error.message);
  } else if (error instanceof ApiError) {
    // Error de API
    console.error('Error API:', error.statusCode, error.message);
  } else if (error instanceof ValidationError) {
    // Error de validación
    console.error('Validación fallida:', error.message);
  }
}
```

### Códigos de Estado HTTP

El cliente maneja automáticamente:

- **401**: No autorizado
- **403**: Prohibido
- **404**: No encontrado
- **429**: Demasiadas solicitudes
- **500-504**: Errores del servidor

## 📊 Monitoreo en Tiempo Real

### Hook de Health Check

```typescript
import { useApiHealth } from '@/lib/hooks';

function MyComponent() {
  const {
    isHealthy,
    isLoading,
    lastChecked,
    message,
    refreshHealth,
  } = useApiHealth({
    enabled: true,
    refetchInterval: 30000, // 30 segundos
  });

  return (
    <div>
      <p>Estado: {isHealthy ? 'Conectado' : 'Desconectado'}</p>
      <p>Última verificación: {lastChecked ? new Date(lastChecked).toLocaleTimeString() : 'Nunca'}</p>
      <button onClick={refreshHealth}>Verificar Ahora</button>
    </div>
  );
}
```

### Componente de Estado

El componente `ApiStatus` se muestra automáticamente en el layout:

```typescript
// Ya está incluido en app/layout.tsx
<ApiStatus position="top-right" />
```

O úsalo manualmente:

```typescript
import { ApiStatus } from '@/components/api-status';

<ApiStatus 
  showDetails={true} 
  position="top-right" 
/>
```

## 🔍 Debugging

### Logs en Desarrollo

En modo desarrollo, todas las solicitudes y respuestas se registran en la consola:

```
[API Request] GET /music/search
[API Response] GET /music/search { status: 200, duration: 150ms }
```

### Información de Request

Cada request tiene un ID único para rastreo:

```typescript
// El ID se muestra en los logs
[API Request] GET /music/search { requestId: '1234567890-abc123' }
```

## ⚙️ Configuración Avanzada

### Timeout Personalizado

```typescript
import { musicApiClient } from '@/lib/api/client';

const response = await musicApiClient.get('/endpoint', {
  timeout: 10000, // 10 segundos
});
```

### Headers Personalizados

```typescript
const response = await musicApiClient.get('/endpoint', {
  headers: {
    'Custom-Header': 'value',
  },
});
```

### Cancelación de Requests

```typescript
import { CancelTokenSource } from 'axios';
import { musicApiClient } from '@/lib/api/client';

const source = axios.CancelToken.source();

const request = musicApiClient.get('/endpoint', {
  cancelToken: source.token,
});

// Cancelar el request
source.cancel('Operación cancelada por el usuario');
```

## 🐛 Solución de Problemas

### Problema: "Network Error"

**Causas posibles:**
- El servidor API no está corriendo
- URL incorrecta en las variables de entorno
- Problemas de CORS
- Firewall bloqueando la conexión

**Solución:**
1. Verifica que el servidor API esté corriendo
2. Verifica la URL en `.env.local`
3. Verifica la configuración de CORS en el backend
4. Usa `testApiConnection()` para diagnosticar

### Problema: "CORS Error"

**Solución:**
Asegúrate de que el backend permita requests desde el frontend:

```python
# En el backend (FastAPI)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Problema: "Timeout"

**Solución:**
Aumenta el timeout en la configuración:

```typescript
// En lib/config/app.ts
export const apiConfig = {
  music: {
    timeout: 60000, // 60 segundos
  },
};
```

## 📝 Ejemplos Completos

### Ejemplo: Búsqueda con Manejo de Errores

```typescript
import { searchTracks } from '@/lib/api';
import { ApiError, NetworkError } from '@/lib/api';
import toast from 'react-hot-toast';

async function handleSearch(query: string) {
  try {
    const results = await searchTracks(query, 10);
    return results;
  } catch (error) {
    if (error instanceof NetworkError) {
      toast.error('Error de conexión. Verifica tu internet.');
    } else if (error instanceof ApiError) {
      toast.error(`Error del servidor: ${error.message}`);
    } else {
      toast.error('Error desconocido');
    }
    throw error;
  }
}
```

### Ejemplo: Health Check en Componente

```typescript
'use client';

import { useApiHealth } from '@/lib/hooks';
import { useEffect } from 'react';
import toast from 'react-hot-toast';

export function ApiHealthMonitor() {
  const { isHealthy, message } = useApiHealth();

  useEffect(() => {
    if (!isHealthy) {
      toast.error(`API desconectada: ${message}`);
    }
  }, [isHealthy, message]);

  return null; // Componente invisible
}
```

## 🔗 Recursos Adicionales

- [Documentación de Axios](https://axios-http.com/docs/intro)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Next.js Environment Variables](https://nextjs.org/docs/basic-features/environment-variables)

