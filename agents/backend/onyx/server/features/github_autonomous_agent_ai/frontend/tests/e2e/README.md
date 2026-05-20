# E2E Tests - Estructura y Guía

## 📁 Estructura de Archivos

```
e2e/
├── complete-flow.spec.ts          # Tests principales del flujo completo
├── helpers.ts                      # Utilidades compartidas (constantes, helpers básicos)
├── page-objects/                   # Page Object Model
│   └── agent-control-page.ts      # Página de control del agente
├── test-helpers/                   # Helpers específicos para tests
│   ├── assertions.ts              # Assertions personalizadas
│   └── test-steps.ts              # Pasos reutilizables de tests
└── README.md                       # Esta documentación
```

## 🏗️ Arquitectura

### 1. **Helpers Base** (`helpers.ts`)
Contiene utilidades fundamentales compartidas:
- Constantes (TIMEOUTS, SELECTORS, etc.)
- Helpers de navegación
- Helpers de tareas
- Helpers de API
- Helpers de logging

### 2. **Page Object Model** (`page-objects/`)
Encapsula interacciones con páginas específicas:
- `AgentControlPage`: Interacciones con la página de control del agente
- Métodos para navegación, creación de tareas, etc.
- Facilita mantenimiento cuando cambia la UI

### 3. **Test Helpers** (`test-helpers/`)
Contiene helpers específicos para tests:
- **Assertions** (`assertions.ts`): Assertions personalizadas con mensajes claros
- **Test Steps** (`test-steps.ts`): Flujos reutilizables completos

### 4. **Tests** (`*.spec.ts`)
Tests limpios y legibles que usan los helpers:
- Fáciles de leer y entender
- Reutilizan pasos comunes
- Mantenibles y extensibles

## 📝 Ejemplo de Uso

### Test Simple

```typescript
test('debería completar el flujo completo', async ({ page }) => {
  await completeTaskFlow(page, 'Crea un archivo README.md');
});
```

### Test con Callback Personalizado

```typescript
test('debería monitorear progreso con logging', async ({ page }) => {
  await completeTaskFlow(page, 'Instrucción de prueba', (attempt, task) => {
    console.log(`Intento ${attempt}: ${task?.status}`);
  });
});
```

### Test con Page Object

```typescript
test('debería interactuar con la página directamente', async ({ page }) => {
  const agentPage = new AgentControlPage(page);
  await agentPage.goto();
  await agentPage.createTask('Test');
  const task = await agentPage.getLastTask();
  expectTaskHasContent(task);
});
```

## 🎯 Principios de Diseño

### 1. **Separación de Responsabilidades**
- Helpers base: Funcionalidad genérica
- Page Objects: Interacciones con UI
- Test Steps: Flujos completos
- Tests: Casos de prueba específicos

### 2. **Reutilización**
- Helpers compartidos entre todos los tests
- Pasos reutilizables para flujos comunes
- Page Objects para interacciones repetidas

### 3. **Mantenibilidad**
- Cambios en UI solo afectan Page Objects
- Cambios en flujos solo afectan Test Steps
- Tests permanecen estables

### 4. **Legibilidad**
- Tests leen como documentación
- Nombres descriptivos
- Estructura clara

## 🔧 Extensión

### Agregar Nueva Página

1. Crear Page Object en `page-objects/`:
```typescript
export class NewPage {
  constructor(private readonly page: Page) {}
  // Métodos de interacción
}
```

2. Usar en tests:
```typescript
const newPage = new NewPage(page);
await newPage.doSomething();
```

### Agregar Nuevo Test Step

1. Agregar función en `test-helpers/test-steps.ts`:
```typescript
export async function newTestStep(page: Page): Promise<void> {
  // Implementación
}
```

2. Usar en tests:
```typescript
await newTestStep(page);
```

### Agregar Nueva Assertion

1. Agregar función en `test-helpers/assertions.ts`:
```typescript
export function expectSomething(value: any): void {
  expect(value).toBeDefined();
}
```

2. Usar en tests:
```typescript
expectSomething(result);
```

## 📊 Ventajas de esta Estructura

1. **Mantenibilidad**: Cambios localizados
2. **Reutilización**: Código compartido
3. **Legibilidad**: Tests claros y concisos
4. **Escalabilidad**: Fácil agregar nuevos tests
5. **Testabilidad**: Helpers pueden testearse independientemente
6. **Consistencia**: Mismo patrón en todos los tests

## 📊 Sistema de Métricas y Reporting

### Métricas de Performance

El sistema incluye tracking automático de métricas:

```typescript
import { createMetricsTracker, collectPerformanceMetrics } from './test-helpers/metrics';

const tracker = createMetricsTracker('My Test');
tracker.start();

tracker.startStep('Navigate');
await page.goto('/page');
tracker.endStep(true);

const metrics = tracker.finish(true);
const performance = await collectPerformanceMetrics(page);
```

### Reportes Avanzados

Genera reportes en múltiples formatos:

```typescript
import { generateReport, generateTextReport, saveReport } from './test-helpers/reporting';

const report = generateReport(metrics);

// Reporte en texto
console.log(generateTextReport(report));

// Guardar reporte HTML
await saveReport(report, 'html', 'test-results');
```

### Formatos de Reporte

- **Texto**: Formato legible en consola
- **JSON**: Para procesamiento automatizado
- **HTML**: Reportes visuales interactivos

## 🚀 Próximos Pasos

- [x] Fixtures de Playwright para setup/teardown
- [x] Sistema de métricas y reporting
- [x] Assertions personalizadas
- [ ] Agregar más Page Objects según necesidad
- [ ] Documentar patrones comunes
- [ ] Crear ejemplos de tests avanzados
- [ ] Integración con CI/CD para reportes automáticos

