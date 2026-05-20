/**
 * Tests de integración para el endpoint /api/deepseek/stream
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';
const DEEPSEEK_STREAM_ENDPOINT = `${BASE_URL}/api/deepseek/stream`;

describe('DeepSeek Stream Integration Tests', () => {
  it('debería devolver un stream válido con contenido', async () => {
    const testInstruction = 'Crea un archivo test.txt con "Hello World"';
    
    const response = await fetch(DEEPSEEK_STREAM_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: testInstruction,
        repository: 'test/repo',
        context: {},
      }),
    });

    expect(response.ok).toBe(true);
    expect(response.status).toBe(200);
    expect(response.headers.get('content-type')).toContain('text/event-stream');
    expect(response.body).toBeTruthy();

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let chunkCount = 0;
    let totalContent = 0;
    let firstChunk: string | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunkCount++;
        const chunk = decoder.decode(value, { stream: true });
        
        if (chunkCount === 1) {
          firstChunk = chunk;
        }

        // Buscar contenido en el chunk
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            if (parsed.content) {
              totalContent += parsed.content.length;
            }
          } catch (e) {
            // Ignorar líneas que no son JSON
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    console.log(`📊 Test resultado: ${chunkCount} chunks, ${totalContent} caracteres`);
    console.log(`📊 Primer chunk: ${firstChunk?.substring(0, 100)}`);

    expect(chunkCount).toBeGreaterThan(1);
    expect(totalContent).toBeGreaterThan(0);
    expect(firstChunk).not.toBe('{}');
  }, 30000);

  it('debería manejar errores correctamente cuando falta la instrucción', async () => {
    const response = await fetch(DEEPSEEK_STREAM_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        repository: 'test/repo',
      }),
    });

    expect(response.status).toBe(400);
    const error = await response.json();
    expect(error.error).toContain('instrucción');
  });

  it('debería devolver contenido suficiente (más de 500 caracteres)', async () => {
    const testInstruction = 'Crea un sistema completo de gestión de tareas con múltiples archivos';
    
    const response = await fetch(DEEPSEEK_STREAM_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: testInstruction,
        repository: 'test/repo',
        context: {},
      }),
    });

    expect(response.ok).toBe(true);

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let totalContent = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            if (parsed.content) {
              totalContent += parsed.content.length;
            }
          } catch (e) {
            // Ignorar
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    expect(totalContent).toBeGreaterThan(500);
  }, 60000);
});

