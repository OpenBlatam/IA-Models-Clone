/**
 * Tests de timing para identificar si el stream se cierra prematuramente
 */
import { describe, it, expect } from 'vitest';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

describe('Stream Timing Tests', () => {
  it('debería verificar el tiempo que tarda el stream en responder', async () => {
    const startTime = Date.now();

    const response = await fetch(`${BASE_URL}/api/deepseek/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: 'Test',
        repository: 'test/repo',
        context: {},
      }),
    });

    const responseTime = Date.now() - startTime;
    console.log(`📊 Tiempo de respuesta: ${responseTime}ms`);

    // Si la respuesta es muy rápida (< 100ms), podría indicar que se está cerrando inmediatamente
    if (responseTime < 100) {
      console.warn('⚠️ La respuesta fue muy rápida, podría indicar cierre prematuro');
    }

    expect(response.ok).toBe(true);

    // Leer el primer chunk
    const reader = response.body!.getReader();
    let firstChunkTime = 0;
    let chunkCount = 0;

    try {
      const firstChunkStart = Date.now();
      let done = false;
      
      while (true) {
        const { done: d, value } = await reader.read();
        done = d;
        
        if (chunkCount === 0) {
          firstChunkTime = Date.now() - firstChunkStart;
          console.log(`📊 Tiempo hasta primer chunk: ${firstChunkTime}ms`);
          
          if (firstChunkTime < 10 && done) {
            console.error('❌ PROBLEMA: El stream terminó inmediatamente');
            throw new Error('Stream se cerró inmediatamente');
          }
        }
        
        if (done) break;
        chunkCount++;
      }

      const totalTime = Date.now() - startTime;
      console.log(`📊 Tiempo total: ${totalTime}ms, Chunks: ${chunkCount}`);

      expect(chunkCount).toBeGreaterThan(1);
    } finally {
      reader.releaseLock();
    }
  }, 60000);

  it('debería verificar si hay un timeout que está cerrando el stream', async () => {
    // Este test verifica si hay un timeout configurado que está cerrando el stream
    const response = await fetch(`${BASE_URL}/api/deepseek/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: 'Crea un sistema completo con múltiples archivos',
        repository: 'test/repo',
        context: {},
      }),
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let chunkCount = 0;
    let lastChunkTime = Date.now();
    const chunkTimes: number[] = [];

    try {
      while (true) {
        const chunkStart = Date.now();
        const { done, value } = await reader.read();
        const chunkTime = Date.now() - chunkStart;

        if (done) break;

        chunkCount++;
        chunkTimes.push(chunkTime);

        const chunk = decoder.decode(value, { stream: true });
        console.log(`📊 Chunk #${chunkCount}: ${chunkTime}ms, tamaño: ${chunk.length}`);

        // Si el stream se detiene abruptamente después de un tiempo específico
        const timeSinceStart = Date.now() - lastChunkTime;
        if (timeSinceStart > 120000 && chunkCount === 1) {
          console.error('❌ PROBLEMA: El stream se detuvo después de ~2 minutos');
          console.error('❌ Esto podría indicar un timeout configurado');
          throw new Error('Stream se detuvo por timeout');
        }

        lastChunkTime = Date.now();
      }
    } finally {
      reader.releaseLock();
    }

    console.log(`📊 Total chunks: ${chunkCount}`);
    console.log(`📊 Tiempos promedio: ${chunkTimes.reduce((a, b) => a + b, 0) / chunkTimes.length}ms`);

    expect(chunkCount).toBeGreaterThan(1);
  }, 180000);
});

