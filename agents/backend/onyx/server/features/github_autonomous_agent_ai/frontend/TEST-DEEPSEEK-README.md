# Test de Debug para Stream de DeepSeek

Este script de test está diseñado para diagnosticar el problema donde el stream de DeepSeek solo devuelve `{}` y termina inmediatamente.

## Cómo ejecutar

```bash
npm run test:deepseek
```

O directamente:

```bash
npx tsx test-deepseek-stream-debug.ts
```

## Qué hace el test

El script realiza dos tests:

### Test 1: Llamada directa a DeepSeek API
- Llama directamente a la API de DeepSeek sin pasar por el endpoint local
- Verifica si DeepSeek está respondiendo correctamente
- Muestra los chunks recibidos y el contenido extraído

### Test 2: Llamada al endpoint local
- Llama al endpoint `/api/deepseek/stream`
- Verifica qué está devolviendo el endpoint
- Analiza cada chunk recibido
- Identifica si hay problemas en el procesamiento del stream

## Qué buscar en los resultados

### Si el Test 1 falla:
- **Problema**: La API de DeepSeek no está respondiendo
- **Solución**: Verificar la API key y la conexión a internet

### Si el Test 1 pasa pero el Test 2 falla:
- **Problema**: El endpoint local está procesando mal el stream
- **Solución**: Revisar el código en `app/api/deepseek/stream/route.ts`

### Si ambos tests muestran solo `{}`:
- **Problema**: El stream se está cerrando prematuramente
- **Solución**: Revisar los logs `[DEEPSEEK]` en el servidor para ver dónde se está cerrando

## Logs importantes

El script muestra logs detallados de:
- Cada chunk recibido
- El contenido de cada línea procesada
- Errores de parsing JSON
- Estadísticas finales

## Interpretación de resultados

### Caso exitoso:
```
✅ [TEST] Stream terminado después de X chunks
📊 [TEST] Total de contenido: XXXX caracteres
✅ Éxito
```

### Caso problemático (solo {}):
```
📥 [TEST] Chunk #1 recibido: {}
✅ [TEST] Stream terminado después de 1 chunks
📊 [TEST] Total de contenido: 0 caracteres
❌ Falló
```

En este caso, revisa los logs del servidor que empiezan con `[DEEPSEEK]` para ver:
- Si el método `start()` se ejecuta
- Si la llamada a DeepSeek se realiza
- Si hay errores en la respuesta
- Si el stream se cierra antes de tiempo

