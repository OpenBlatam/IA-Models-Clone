# Test de Diagnóstico del Stream

Este script de prueba ayuda a diagnosticar el problema del stream que se cierra prematuramente con 0 caracteres.

## Instalación

Primero, instala las dependencias necesarias:

```bash
npm install
# o
yarn install
```

## Uso

### Opción 1: Usando el script npm

```bash
npm run test:stream
```

### Opción 2: Ejecutar directamente con tsx

```bash
npx tsx test-stream-debug.ts
```

### Opción 3: Ejecutar con Node.js (si tienes ts-node)

```bash
node --loader ts-node/esm test-stream-debug.ts
```

## Qué hace el test

El script ejecuta 3 tests diferentes:

### 1. Test: API Directa de DeepSeek
- Llama directamente a la API de DeepSeek
- Verifica que la API key sea válida
- Verifica que el stream funcione correctamente
- Mide cuánto contenido se recibe

### 2. Test: Endpoint de Stream
- Prueba el endpoint `/api/deepseek/stream` de tu aplicación
- Verifica que el stream se procese correctamente
- Mide chunks recibidos y contenido acumulado
- Detecta errores en el parseo del JSON

### 3. Test: Procesamiento Completo de Tarea
- Crea una tarea de prueba
- Inicia el procesamiento en el backend
- Espera 15 segundos
- Verifica que la tarea tenga contenido

## Interpretación de Resultados

### ✅ Test Exitoso
- El stream recibe contenido
- El contenido se parsea correctamente
- La tarea se procesa sin errores

### ❌ Test Fallido

#### "Stream terminó sin contenido"
- **Posibles causas:**
  - La API de DeepSeek no está respondiendo
  - La API key es inválida o expiró
  - Hay un problema de red
  - El stream se está cerrando prematuramente

#### "Error en el parseo"
- **Posibles causas:**
  - El formato del stream cambió
  - Hay un problema con el decoder
  - El buffer no se está procesando correctamente

#### "La respuesta no tiene body"
- **Posibles causas:**
  - El endpoint no está funcionando
  - Hay un error en el servidor
  - El stream no se está creando correctamente

## Variables de Entorno

Asegúrate de tener estas variables configuradas:

```bash
NEXT_PUBLIC_DEEPSEEK_API_KEY=tu_api_key
NEXT_PUBLIC_DEEPSEEK_API_BASE_URL=https://api.deepseek.com
NEXT_PUBLIC_DEEPSEEK_MODEL=deepseek-chat
NEXT_PUBLIC_APP_URL=http://localhost:3001
```

## Solución de Problemas

### El test falla en "API Directa de DeepSeek"
1. Verifica que tu API key sea válida
2. Verifica que tengas créditos en tu cuenta de DeepSeek
3. Verifica tu conexión a internet

### El test falla en "Endpoint de Stream"
1. Asegúrate de que el servidor Next.js esté corriendo
2. Verifica que el endpoint `/api/deepseek/stream` esté accesible
3. Revisa los logs del servidor para ver errores

### El test falla en "Procesamiento de Tarea"
1. Verifica que el backend esté procesando tareas
2. Revisa los logs del servidor
3. Verifica que el directorio `data/tasks` exista y tenga permisos de escritura

## Logs Detallados

El test muestra logs detallados de:
- Cada chunk recibido
- El contenido parseado
- Errores encontrados
- Estadísticas del stream

Usa estos logs para identificar exactamente dónde está fallando el stream.

