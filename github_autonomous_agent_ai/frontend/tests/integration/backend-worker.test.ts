/**
 * Tests de integración para el backend worker
 * Simula exactamente lo que hace el backend worker
 */
import { describe, it, expect } from 'vitest';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

describe('Backend Worker Integration', () => {
  it('debería procesar una tarea completa end-to-end', async () => {
    // 1. Crear tarea
    const createResponse = await fetch(`${BASE_URL}/api/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: 'Crea un archivo README.md con "Hello World"',
        repository: 'test/repo',
        status: 'pending',
        repoInfo: {
          name: 'test',
          full_name: 'test/repo',
          default_branch: 'main',
        },
        model: 'deepseek-chat',
      }),
    });

    expect(createResponse.ok).toBe(true);
    const task = await createResponse.json();
    const taskId = task.id;

    // 2. Iniciar procesamiento (simulando backend worker)
    const processResponse = await fetch(`${BASE_URL}/api/tasks/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId }),
    });

    expect(processResponse.ok).toBe(true);

    // 3. Esperar y verificar progreso
    let finalTask: any = null;
    for (let i = 0; i < 60; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const getResponse = await fetch(`${BASE_URL}/api/tasks`);
      const tasks = await getResponse.json();
      finalTask = tasks.find((t: any) => t.id === taskId);

      if (finalTask) {
        const status = finalTask.status;
        const contentLength = finalTask.streamingContent?.length || 0;

        console.log(`📊 Segundo ${i + 1}: Estado=${status}, Contenido=${contentLength}`);

        if (status === 'completed' || status === 'pending_approval') {
          break;
        }

        if (status === 'failed') {
          throw new Error(`Tarea falló: ${finalTask.error}`);
        }

        // Si después de 10 segundos no hay contenido, hay un problema
        if (i >= 10 && contentLength === 0) {
          console.error(`❌ PROBLEMA: Después de ${i + 1} segundos, contenido = 0`);
          console.error(`❌ Estado: ${status}`);
          console.error(`❌ Error: ${finalTask.error || 'ninguno'}`);
          
          // Verificar directamente el endpoint
          const directResponse = await fetch(`${BASE_URL}/api/deepseek/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              instruction: finalTask.instruction,
              repository: finalTask.repository,
              context: {},
            }),
          });

          const directText = await directResponse.text();
          console.error(`❌ Respuesta directa: ${directText.substring(0, 500)}`);
          
          throw new Error(`Stream no está devolviendo contenido`);
        }
      }
    }

    // 4. Verificar resultado final
    expect(finalTask).toBeTruthy();
    expect(finalTask.status).toMatch(/completed|pending_approval|failed/);
    
    if (finalTask.status !== 'failed') {
      expect(finalTask.streamingContent?.length || 0).toBeGreaterThan(0);
    }
  }, 120000);

  it('debería detectar el problema del stream que solo devuelve {}', async () => {
    const createResponse = await fetch(`${BASE_URL}/api/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: 'Test simple',
        repository: 'test/repo',
        status: 'pending',
        repoInfo: { name: 'test', full_name: 'test/repo', default_branch: 'main' },
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

    // Esperar 5 segundos
    await new Promise(resolve => setTimeout(resolve, 5000));

    const getResponse = await fetch(`${BASE_URL}/api/tasks`);
    const tasks = await getResponse.json();
    const currentTask = tasks.find((t: any) => t.id === taskId);

    if (currentTask) {
      const contentLength = currentTask.streamingContent?.length || 0;
      const status = currentTask.status;

      console.log(`📊 Estado: ${status}, Contenido: ${contentLength}`);

      if (contentLength === 0 && status === 'processing') {
        console.error(`❌ PROBLEMA DETECTADO: Stream devolvió 0 caracteres`);
        console.error(`❌ Esto indica que el stream se está cerrando prematuramente`);
        
        // Verificar el endpoint directamente
        const directResponse = await fetch(`${BASE_URL}/api/deepseek/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            instruction: currentTask.instruction,
            repository: currentTask.repository,
            context: {},
          }),
        });

        const directText = await directResponse.text();
        console.error(`❌ Respuesta directa: ${directText.substring(0, 500)}`);
        
        // Si el endpoint directo funciona pero el worker no, el problema está en el worker
        if (directText.length > 2 && directText !== '{}') {
          throw new Error('El endpoint funciona directamente pero el worker no lo procesa correctamente');
        }
      }
    }
  }, 30000);
});

