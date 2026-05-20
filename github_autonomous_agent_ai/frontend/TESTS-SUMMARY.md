# Resumen de Tests de Debug

## 🎯 Objetivo
Encontrar el culpable del problema donde el stream solo devuelve `{}` y termina inmediatamente.

## 📋 Tests Creados

### 1. Tests de Scripts Directos
- **`test-stream-debug.ts`** - Test del stream básico
- **`test-deepseek-stream-debug.ts`** - Test del endpoint DeepSeek
- **`test-backend-worker-stream.ts`** - Test que simula el backend worker

**Ejecutar:**
```bash
npm run test:stream
npm run test:deepseek
npm run test:backend-worker
```

### 2. Unit Tests (Vitest)
- **`tests/unit/stream-processor.test.ts`** - Tests del procesador de streams
- **`tests/unit/task-context.test.ts`** - Tests de construcción de contexto
- **`tests/unit/task-storage.test.ts`** - Tests de almacenamiento

**Ejecutar:**
```bash
npm run test:unit
```

### 3. Integration Tests (Vitest)
- **`tests/integration/deepseek-stream.test.ts`** - Test del endpoint
- **`tests/integration/backend-worker.test.ts`** - Test del worker completo
- **`tests/integration/full-flow.test.ts`** - Test del flujo completo

**Ejecutar:**
```bash
npm run test:integration
```

### 4. E2E Tests (Playwright)
- **`tests/e2e/task-processing.spec.ts`** - Test E2E del procesamiento
- **`tests/e2e/stream-debug.spec.ts`** - Test E2E específico para debug

**Ejecutar:**
```bash
npm run test:e2e
npm run test:e2e:ui  # Con interfaz gráfica
```

### 5. Script Maestro
- **`tests/run-all-debug.ts`** - Ejecuta todos los tests y genera un análisis

**Ejecutar:**
```bash
npm run test:debug:all
```

## 🚀 Instalación

Primero instala las dependencias:

```bash
npm install
```

## 📊 Cómo Interpretar los Resultados

### Si todos los tests pasan:
- El código funciona correctamente
- El problema podría ser intermitente
- Revisa los logs del servidor durante ejecución real

### Si solo fallan los tests del backend worker:
- **Culpable**: El procesamiento en `app/api/tasks/process/route.ts`
- **Solución**: Revisar cómo se procesa el stream en el worker

### Si fallan los tests del endpoint:
- **Culpable**: El endpoint `/api/deepseek/stream`
- **Solución**: Revisar `app/api/deepseek/stream/route.ts`

### Si fallan los unit tests:
- **Culpable**: Lógica incorrecta en las funciones
- **Solución**: Revisar las funciones específicas que fallan

## 🔍 Diagnóstico Automático

Los tests incluyen diagnósticos automáticos que:
1. Verifican si el endpoint funciona directamente
2. Comparan el comportamiento del worker vs llamada directa
3. Identifican dónde se pierde el contenido
4. Generan logs detallados para análisis

## 📝 Próximos Pasos

1. **Ejecuta todos los tests:**
   ```bash
   npm run test:debug:all
   ```

2. **Revisa el resumen final** que muestra qué tests fallaron

3. **Revisa los logs detallados** de cada test para ver exactamente dónde falla

4. **Ejecuta tests individuales** si necesitas más detalle:
   ```bash
   npm run test:unit
   npm run test:integration
   ```

5. **Revisa los logs del servidor** cuando ejecutes una tarea real para ver los logs `[DEEPSEEK]` y `[BACKEND]`

## 🎯 Encontrar el Culpable

El script `test:debug:all` ejecutará todos los tests y al final mostrará un análisis que identifica:
- Qué componente está fallando
- Dónde está el problema
- Qué revisar para solucionarlo

