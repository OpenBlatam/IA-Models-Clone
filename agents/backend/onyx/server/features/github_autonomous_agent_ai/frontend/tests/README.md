# Suite de Tests para Debug

Esta suite de tests está diseñada para encontrar el culpable del problema donde el stream solo devuelve `{}`.

## Estructura de Tests

### Unit Tests (`tests/unit/`)
- `stream-processor.test.ts` - Tests del procesador de streams
- `task-context.test.ts` - Tests de construcción de contexto
- `task-storage.test.ts` - Tests de almacenamiento de tareas

### Integration Tests (`tests/integration/`)
- `deepseek-stream.test.ts` - Tests del endpoint `/api/deepseek/stream`
- `backend-worker.test.ts` - Tests del backend worker completo
- `full-flow.test.ts` - Test del flujo completo end-to-end

### E2E Tests (`tests/e2e/`)
- `task-processing.spec.ts` - Tests E2E con Playwright del procesamiento
- `stream-debug.spec.ts` - Tests E2E específicos para debug del stream

## Cómo ejecutar

### Instalar dependencias
```bash
npm install
```

### Ejecutar todos los tests
```bash
npm run test:all
```

### Ejecutar por categoría
```bash
# Unit tests
npm run test:unit

# Integration tests
npm run test:integration

# E2E tests (requiere servidor corriendo)
npm run test:e2e

# E2E tests con UI
npm run test:e2e:ui
```

### Ejecutar tests de debug
```bash
npm run test:debug
```

### Ejecutar tests individuales
```bash
# Test del stream
npm run test:stream

# Test de DeepSeek
npm run test:deepseek

# Test del backend worker
npm run test:backend-worker
```

## Qué buscar en los resultados

### Si los unit tests fallan:
- **Problema**: Lógica incorrecta en las funciones
- **Solución**: Revisar el código de las funciones específicas

### Si los integration tests fallan:
- **Problema**: Problema en la comunicación entre componentes
- **Solución**: Revisar los endpoints y la comunicación entre ellos

### Si los E2E tests fallan:
- **Problema**: Problema en el flujo completo
- **Solución**: Revisar el flujo end-to-end y los logs del servidor

### Si el test detecta contenido = 0:
- **Problema**: El stream se está cerrando prematuramente
- **Solución**: Revisar los logs `[DEEPSEEK]` y `[BACKEND]` en el servidor

## Interpretación de resultados

### Caso exitoso:
```
✅ Tarea completada: pending_approval
✅ Contenido: 524 caracteres
```

### Caso problemático:
```
❌ PROBLEMA DETECTADO: Stream devolvió 0 caracteres
❌ Estado: processing
❌ Error: ninguno
```

En este caso, el test ejecutará diagnósticos automáticos para identificar el problema.

## Logs importantes

Los tests generan logs detallados que incluyen:
- Estado de cada paso del proceso
- Cantidad de contenido recibido
- Errores encontrados
- Diagnósticos automáticos

Revisa estos logs para identificar exactamente dónde falla el proceso.

