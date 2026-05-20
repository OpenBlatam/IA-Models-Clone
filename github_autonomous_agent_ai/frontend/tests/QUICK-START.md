# Quick Start - Tests de Debug

## 🚀 Ejecutar Todos los Tests

```bash
# 1. Instalar dependencias
npm install

# 2. Ejecutar todos los tests de debug
npm run test:debug:all
```

## 📋 Tests Disponibles

### Tests Rápidos (sin servidor)
```bash
npm run test:unit              # Unit tests
npm run test:stream            # Test del stream básico
npm run test:deepseek          # Test del endpoint DeepSeek
npm run test:backend-worker    # Test del backend worker
```

### Tests de Integración (requiere servidor en puerto 3001)
```bash
npm run test:integration       # Integration tests
```

### Tests E2E (requiere servidor y Playwright)
```bash
npm run test:e2e               # E2E tests
npm run test:e2e:ui            # E2E tests con UI
```

## 🎯 Encontrar el Culpable

El script `test:debug:all` ejecutará todos los tests y al final mostrará:

1. **Qué tests pasaron/fallaron**
2. **Dónde está el problema**
3. **Qué componente revisar**
4. **Recomendaciones de solución**

## 🔍 Interpretación Rápida

### Si fallan los unit tests:
→ Problema en la lógica de las funciones

### Si fallan los integration tests:
→ Problema en la comunicación entre componentes

### Si fallan los E2E tests:
→ Problema en el flujo completo

### Si todos pasan pero el problema persiste:
→ Revisa los logs del servidor cuando ejecutas una tarea real

## 📊 Logs Importantes

Cuando ejecutes una tarea real, busca en los logs del servidor:

- `[DEEPSEEK]` - Logs del endpoint
- `[BACKEND]` - Logs del worker
- `[STREAM]` - Logs del procesador

Estos logs te dirán exactamente dónde está fallando.

