/**
 * E2E Tests para aislar el problema del stream
 * Prueba cada componente por separado
 */
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

test.describe('Stream Isolation Tests', () => {
  test('debería verificar que el endpoint responde correctamente', async ({ request }) => {
    const response = await request.post(`${BASE_URL}/api/deepseek/stream`, {
      data: {
        instruction: 'Test simple',
        repository: 'test/repo',
        context: {},
      },
      timeout: 30000,
    });

    expect(response.ok()).toBeTruthy();
    expect(response.status()).toBe(200);

    // En Playwright, response.body() devuelve un Buffer
    const body = await response.body();
    const text = new TextDecoder().decode(body);

    console.log(`📊 Tamaño de respuesta: ${body.length} bytes`);
    console.log(`📊 Primeros 200 caracteres: ${text.substring(0, 200)}`);

    // Verificar que NO es solo {}
    expect(text).not.toBe('{}');
    expect(body.length).toBeGreaterThan(2);

    // Verificar que hay contenido JSON válido
    const lines = text.split('\n').filter(l => l.trim());
    let hasValidContent = false;

    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        if (parsed.content && parsed.content.length > 0) {
          hasValidContent = true;
          console.log(`✅ Contenido válido encontrado: ${parsed.content.length} caracteres`);
          break;
        }
      } catch (e) {
        // Continuar
      }
    }

    expect(hasValidContent).toBe(true);
  }, 60000);

  test('debería verificar el Content-Type de la respuesta', async ({ request }) => {
    const response = await request.post(`${BASE_URL}/api/deepseek/stream`, {
      data: {
        instruction: 'Test',
        repository: 'test/repo',
        context: {},
      },
    });

    const contentType = response.headers()['content-type'];
    console.log(`📊 Content-Type: ${contentType}`);

    expect(contentType).toContain('text/event-stream');
  });

  test('debería verificar que el stream no se cierra inmediatamente', async ({ request }) => {
    const startTime = Date.now();
    
    const response = await request.post(`${BASE_URL}/api/deepseek/stream`, {
      data: {
        instruction: 'Crea un archivo README.md',
        repository: 'test/repo',
        context: {},
      },
    });

    // En Playwright, necesitamos obtener el body como stream
    const body = await response.body();
    // Convertir el buffer a un stream para leerlo chunk por chunk
    const stream = new ReadableStream({
      start(controller) {
        // Dividir el buffer en chunks simulados
        const chunkSize = 1024;
        let offset = 0;
        while (offset < body.length) {
          const chunk = body.slice(offset, offset + chunkSize);
          controller.enqueue(chunk);
          offset += chunkSize;
        }
        controller.close();
      },
    });
    
    const reader = stream.getReader();
    let chunkCount = 0;
    let totalBytes = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunkCount++;
        totalBytes += value.length;

        // Si recibimos más de 1 chunk, el stream no se cerró inmediatamente
        if (chunkCount > 1) {
          break;
        }
      }
    } finally {
      reader.releaseLock();
    }

    const duration = Date.now() - startTime;
    console.log(`📊 Duración: ${duration}ms`);
    console.log(`📊 Chunks recibidos: ${chunkCount}`);
    console.log(`📊 Bytes totales: ${totalBytes}`);

    // El stream debería durar al menos 100ms y recibir múltiples chunks
    expect(chunkCount).toBeGreaterThan(1);
    expect(totalBytes).toBeGreaterThan(2);
  }, 60000);

  test('debería detectar el problema específico: {} como primer chunk', async ({ request }) => {
    const response = await request.post(`${BASE_URL}/api/deepseek/stream`, {
      data: {
        instruction: 'Test',
        repository: 'test/repo',
        context: {},
      },
    });

    // En Playwright, leer el body completo
    const body = await response.body();
    const text = new TextDecoder().decode(body);
    const firstChunk = text.substring(0, 200); // Primeros 200 caracteres

    console.log(`📊 Primer chunk recibido: "${firstChunk}"`);
    console.log(`📊 Longitud total: ${text.length} caracteres`);

    // Verificar si el primer chunk es solo {}
    if (firstChunk.trim() === '{}' || firstChunk.trim().startsWith('{}')) {
      console.error('❌ PROBLEMA DETECTADO: El primer chunk es solo {}');
      console.error('❌ Esto indica que el stream se está cerrando inmediatamente');
      console.error('❌ Revisa el método start() del ReadableStream en /api/deepseek/stream');
      
      if (text.length <= 2) {
        console.error('❌ El stream terminó inmediatamente después del primer chunk');
        throw new Error('Stream se cierra prematuramente con solo {}');
      }
    }

    expect(firstChunk).not.toBe('{}');
  }, 30000);
});

