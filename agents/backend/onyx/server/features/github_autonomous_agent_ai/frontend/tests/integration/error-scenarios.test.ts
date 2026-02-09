/**
 * Tests de escenarios de error para identificar problemas
 */
import { describe, it, expect } from 'vitest';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

describe('Error Scenarios', () => {
  it('debería manejar correctamente cuando el stream termina sin contenido', async () => {
    // Este test simula el problema actual
    const response = await fetch(`${BASE_URL}/api/deepseek/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: 'Test',
        repository: 'test/repo',
        context: {},
      }),
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let chunkCount = 0;
    let totalContent = 0;
    let firstChunk = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunkCount++;
        const chunk = decoder.decode(value, { stream: true });
        
        if (chunkCount === 1) {
          firstChunk = chunk;
        }

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

    console.log(`📊 Resultado: ${chunkCount} chunks, ${totalContent} caracteres`);
    console.log(`📊 Primer chunk: "${firstChunk.substring(0, 100)}"`);

    if (chunkCount === 1 && totalContent === 0 && firstChunk.includes('{}')) {
      console.error('❌ PROBLEMA CONFIRMADO: Stream solo devuelve {}');
      throw new Error('Stream está devolviendo solo {} y cerrando inmediatamente');
    }

    expect(totalContent).toBeGreaterThan(0);
  }, 30000);

  it('debería verificar si hay un problema con el AbortController', async () => {
    const abortController = new AbortController();
    
    // Crear tarea
    const createResponse = await fetch(`${BASE_URL}/api/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: 'Test',
        repository: 'test/repo',
        status: 'pending',
        repoInfo: { name: 'test', full_name: 'test/repo', default_branch: 'main' },
        model: 'deepseek-chat',
      }),
    });

    const task = await createResponse.json();
    const taskId = task.id;

    // Iniciar procesamiento
    await fetch(`${BASE_URL}/api/tasks/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId }),
    });

    // Esperar un poco
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Verificar si el AbortController se canceló
    const getResponse = await fetch(`${BASE_URL}/api/tasks`);
    const tasks = await getResponse.json();
    const currentTask = tasks.find((t: any) => t.id === taskId);

    if (currentTask) {
      console.log(`📊 Estado: ${currentTask.status}`);
      console.log(`📊 Contenido: ${currentTask.streamingContent?.length || 0}`);
      
      if (currentTask.status === 'stopped') {
        console.error('❌ PROBLEMA: La tarea se detuvo automáticamente');
        console.error('❌ Esto podría indicar que el AbortController se está cancelando');
      }
    }
  }, 30000);

  it('debería verificar si hay un problema con el puerto/URL', async () => {
    const urls = [
      'http://localhost:3000',
      'http://localhost:3001',
      process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001',
    ];

    for (const url of urls) {
      try {
        console.log(`🧪 Probando URL: ${url}`);
        const response = await fetch(`${url}/api/deepseek/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            instruction: 'Test',
            repository: 'test/repo',
            context: {},
          }),
          signal: AbortSignal.timeout(5000),
        });

        if (response.ok) {
          const reader = response.body!.getReader();
          const decoder = new TextDecoder();
          let text = '';
          
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              text += decoder.decode(value, { stream: true });
            }
          } finally {
            reader.releaseLock();
          }
          
          console.log(`✅ ${url} funciona: ${text.length} caracteres`);
          
          if (text.length > 2 && text !== '{}') {
            console.log(`✅ URL correcta encontrada: ${url}`);
            return;
          }
        }
      } catch (error: any) {
        console.log(`❌ ${url} falló: ${error.message}`);
      }
    }

    throw new Error('Ninguna URL funciona correctamente');
  }, 30000);
});

