# Tests E2E con Playwright

## Ejecutar Tests

```bash
# Todos los tests
npm run test:e2e

# Con UI interactivo
npm run test:e2e:ui

# Con navegador visible
npm run test:e2e:headed

# Modo debug
npm run test:e2e:debug
```

## Tests Disponibles

- `chat-interface.spec.ts` - Tests de la interfaz de chat
- `api-integration.spec.ts` - Tests de integración con API
- `model-creation.spec.ts` - Tests del flujo de creación de modelos

## Requisitos

1. Servidor Next.js corriendo en `http://localhost:3000`
2. (Opcional) Servidor API en `http://localhost:8000` para tests completos











