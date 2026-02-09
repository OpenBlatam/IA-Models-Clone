# 🧪 Tests Playwright - TruthGPT Model Builder

## Configuración

### Instalar dependencias

```bash
npm install
# O si Playwright no está instalado:
npx playwright install
```

### Configurar variables de entorno

Crear `.env.local`:
```bash
PLAYWRIGHT_TEST_BASE_URL=http://localhost:3000
TRUTHGPT_API_URL=http://localhost:8000
```

## Ejecutar Tests

### Todos los tests

```bash
npm run test:e2e
```

### Tests con UI (interactivo)

```bash
npm run test:e2e:ui
```

### Tests en modo headed (con navegador visible)

```bash
npm run test:e2e:headed
```

### Tests en modo debug

```bash
npm run test:e2e:debug
```

### Tests específicos

```bash
# Solo tests de ChatInterface
npx playwright test tests/e2e/chat-interface.spec.ts

# Solo tests de integración API
npx playwright test tests/e2e/api-integration.spec.ts

# Tests en un navegador específico
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit
```

## Estructura de Tests

```
tests/
├── playwright.config.ts        # Configuración de Playwright
├── e2e/
│   ├── chat-interface.spec.ts   # Tests de ChatInterface
│   └── api-integration.spec.ts # Tests de integración con API
└── README.md                   # Este archivo
```

## Tests Incluidos

### ✅ chat-interface.spec.ts

#### ChatInterface Tests
- Display de la interfaz
- Mensaje de bienvenida
- Input field funcionalidad
- Validación de inputs
- Paneles (templates, historial)
- Sugerencias
- Atajos de teclado
- Botones habilitados/deshabilitados

#### Model Creation Flow
- Crear modelo y mostrar progreso
- Mostrar preview del modelo

#### Proactive Model Builder
- Toggle del builder proactivo
- Agregar modelos a la cola

#### API Integration
- Verificar estado de conexión
- Manejo de errores de API

#### Responsive Design
- Vista móvil
- Vista tablet

#### Accessibility
- Labels apropiados
- Navegación por teclado

### ✅ api-integration.spec.ts

#### API Integration Tests
- Health check endpoint
- Crear modelo via API
- Compilar modelo via API
- Listar modelos via API
- Manejo de errores de API

#### Frontend to API Integration
- Estado de conexión en frontend
- Crear modelo desde frontend y verificar en API

## Requisitos

### Servidor Frontend
El servidor Next.js debe estar corriendo en `http://localhost:3000`

```bash
npm run dev
```

### Servidor API (opcional)
Para tests completos de integración, el servidor API debe estar corriendo:

```bash
cd ../TruthGPT-main/truthgpt_api
python start_server.py
```

## Configuración Avanzada

### Ejecutar tests en CI

```bash
# En CI, los tests se ejecutan automáticamente
npm run test:all
```

### Generar reportes

```bash
# Reporte HTML
npx playwright test --reporter=html

# Reporte JSON
npx playwright test --reporter=json

# Ver reporte
npx playwright show-report
```

### Screenshots y videos

Los screenshots se toman automáticamente cuando los tests fallan.
Los videos se pueden habilitar en `playwright.config.ts`.

### Traces

Los traces se generan automáticamente en caso de retry.
Para verlos:

```bash
npx playwright show-trace trace.zip
```

## Solución de Problemas

### Error: "Browser not found"
```bash
npx playwright install
```

### Error: "Cannot connect to server"
- Verifica que el servidor Next.js esté corriendo
- Verifica la URL en `playwright.config.ts`

### Tests fallan intermitentemente
- Aumenta los timeouts en los tests
- Verifica que no haya otros procesos usando el puerto 3000

### Tests lentos
- Ejecuta tests en paralelo (ya configurado)
- Reduce el número de workers si hay problemas de recursos

## Mejores Prácticas

1. **Usar selectores estables**: Preferir `data-testid` o roles ARIA
2. **Esperar condiciones**: Usar `waitFor` en lugar de `sleep`
3. **Aislar tests**: Cada test debe ser independiente
4. **Limpiar después**: Limpiar datos creados durante los tests
5. **Verificar estado**: Verificar que el estado esperado se alcanza

## Agregar Nuevos Tests

1. Crear archivo en `tests/e2e/`
2. Importar `test` y `expect` de `@playwright/test`
3. Usar `test.describe` para agrupar tests relacionados
4. Usar `test.beforeEach` para setup común
5. Ejecutar `npm run test:e2e` para verificar

Ejemplo:
```typescript
import { test, expect } from '@playwright/test';

test.describe('Mi Nueva Funcionalidad', () => {
  test('debería hacer algo', async ({ page }) => {
    await page.goto('/');
    // ... tu test aquí
  });
});
```











