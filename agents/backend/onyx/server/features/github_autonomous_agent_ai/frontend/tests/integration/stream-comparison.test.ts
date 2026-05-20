/**
 * Test de comparación: endpoint directo vs backend worker
 * Para identificar diferencias en el comportamiento
 */
import { describe, it, expect } from 'vitest';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

describe('Stream Comparison Test', () => {
  it('debería comparar respuesta directa vs respuesta del worker', async () => {
    const testInstruction = 'Crea un archivo test.txt';
    const testRepository = 'test/repo';

    console.log('🧪 Comparando respuesta directa vs worker...');

    // Test 1: Llamada directa al endpoint
    console.log('\n📤 Test 1: Llamada directa a /api/deepseek/stream');
    const directResponse = await fetch(`${BASE_URL}/api/deepseek/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: testInstruction,
        repository: testRepository,
        context: {},
      }),
    });

    const directReader = directResponse.body!.getReader();
    const directDecoder = new TextDecoder();
    let directChunks = 0;
    let directContent = 0;
    let directFirstChunk = '';

    try {
      while (true) {
        const { done, value } = await directReader.read();
        if (done) break;

        directChunks++;
        const chunk = directDecoder.decode(value, { stream: true });
        
        if (directChunks === 1) {
          directFirstChunk = chunk;
        }

        const lines = chunk.split('\n');
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            if (parsed.content) {
              directContent += parsed.content.length;
            }
          } catch (e) {
            // Ignorar
          }
        }
      }
    } finally {
      directReader.releaseLock();
    }

    console.log(`✅ Directo: ${directChunks} chunks, ${directContent} caracteres`);
    console.log(`📄 Primer chunk directo: ${directFirstChunk.substring(0, 100)}`);

    // Test 2: Crear tarea y procesar (simulando worker)
    console.log('\n📤 Test 2: Crear tarea y procesar (worker)');
    
    const createResponse = await fetch(`${BASE_URL}/api/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: testInstruction,
        repository: testRepository,
        status: 'pending',
        repoInfo: { name: 'test', full_name: testRepository, default_branch: 'main' },
        model: 'deepseek-chat',
      }),
    });

    const task = await createResponse.json();
    const taskId = task.id;

    await fetch(`${BASE_URL}/api/tasks/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId }),
    });

    // Esperar y verificar
    let workerContent = 0;
    let workerChunks = 0;
    let workerFirstChunk = '';

    for (let i = 0; i < 10; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const getResponse = await fetch(`${BASE_URL}/api/tasks`);
      const tasks = await getResponse.json();
      const currentTask = tasks.find((t: any) => t.id === taskId);

      if (currentTask) {
        const content = currentTask.streamingContent || '';
        if (content.length > workerContent) {
          workerContent = content.length;
          if (workerChunks === 0) {
            workerFirstChunk = content.substring(0, 100);
          }
          workerChunks++;
        }

        if (currentTask.status === 'completed' || currentTask.status === 'pending_approval') {
          break;
        }
      }
    }

    console.log(`✅ Worker: ${workerChunks} actualizaciones, ${workerContent} caracteres`);
    console.log(`📄 Primer chunk worker: ${workerFirstChunk.substring(0, 100)}`);

    // Comparación
    console.log('\n📊 COMPARACIÓN:');
    console.log(`   Directo: ${directChunks} chunks, ${directContent} caracteres`);
    console.log(`   Worker: ${workerChunks} actualizaciones, ${workerContent} caracteres`);
    console.log(`   Diferencia: ${directContent - workerContent} caracteres`);

    if (directContent > 0 && workerContent === 0) {
      console.error('\n❌ PROBLEMA IDENTIFICADO:');
      console.error('   - El endpoint funciona directamente');
      console.error('   - El worker NO está recibiendo el contenido');
      console.error('   - Esto indica un problema en el procesamiento del stream en el worker');
      throw new Error('El worker no está procesando el stream correctamente');
    }

    if (directFirstChunk === '{}' || directFirstChunk.includes('{}')) {
      console.error('\n❌ PROBLEMA IDENTIFICADO:');
      console.error('   - El endpoint está devolviendo {}');
      console.error('   - Esto indica un problema en el endpoint /api/deepseek/stream');
      throw new Error('El endpoint está devolviendo solo {}');
    }

    // Verificar que ambos tienen contenido
    expect(directContent).toBeGreaterThan(0);
    expect(workerContent).toBeGreaterThan(0);
  }, 60000);
});

