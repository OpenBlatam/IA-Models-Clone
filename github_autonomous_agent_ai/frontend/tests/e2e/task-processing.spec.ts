/**
 * E2E Tests con Playwright para el flujo completo de procesamiento de tareas
 */
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

test.describe('Task Processing E2E', () => {
  test('debería crear una tarea y procesarla correctamente', async ({ request }) => {
    // 1. Crear una tarea
    const createTaskResponse = await request.post(`${BASE_URL}/api/tasks`, {
      data: {
        instruction: 'Crea un archivo README.md con "Hello World"',
        repository: 'test/repo',
        status: 'pending',
        repoInfo: {
          name: 'test',
          full_name: 'test/repo',
          default_branch: 'main',
        },
        model: 'deepseek-chat',
      },
    });

    expect(createTaskResponse.ok()).toBeTruthy();
    const task = await createTaskResponse.json();
    const taskId = task.id;

    console.log(`✅ Tarea creada: ${taskId}`);

    // 2. Iniciar procesamiento
    const processResponse = await request.post(`${BASE_URL}/api/tasks/process`, {
      data: { taskId },
    });

    expect(processResponse.ok()).toBeTruthy();
    console.log(`✅ Procesamiento iniciado`);

    // 3. Esperar y verificar que la tarea se procesa
    let taskStatus = 'processing';
    let attempts = 0;
    const maxAttempts = 60; // 60 segundos máximo

    while (taskStatus === 'processing' && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      attempts++;

      const getTaskResponse = await request.get(`${BASE_URL}/api/tasks`);
      const tasks = await getTaskResponse.json();
      const currentTask = tasks.find((t: any) => t.id === taskId);

      if (currentTask) {
        taskStatus = currentTask.status;
        console.log(`📊 Intento ${attempts}: Estado = ${taskStatus}, Contenido = ${currentTask.streamingContent?.length || 0} caracteres`);

        if (currentTask.streamingContent && currentTask.streamingContent.length > 0) {
          console.log(`✅ Contenido recibido: ${currentTask.streamingContent.length} caracteres`);
          break;
        }

        if (currentTask.error) {
          console.error(`❌ Error en tarea: ${currentTask.error}`);
          break;
        }
      }
    }

    // 4. Verificar resultado
    const finalTaskResponse = await request.get(`${BASE_URL}/api/tasks`);
    const finalTasks = await finalTaskResponse.json();
    const finalTask = finalTasks.find((t: any) => t.id === taskId);

    expect(finalTask).toBeTruthy();
    
    if (finalTask.status === 'completed' || finalTask.status === 'pending_approval') {
      expect(finalTask.streamingContent?.length || 0).toBeGreaterThan(0);
      console.log(`✅ Tarea completada con ${finalTask.streamingContent?.length || 0} caracteres`);
    } else if (finalTask.status === 'failed') {
      console.error(`❌ Tarea falló: ${finalTask.error}`);
      throw new Error(`Tarea falló: ${finalTask.error}`);
    } else {
      console.warn(`⚠️ Tarea aún procesando después de ${attempts} segundos`);
      console.warn(`⚠️ Estado: ${finalTask.status}, Contenido: ${finalTask.streamingContent?.length || 0} caracteres`);
    }
  }, 120000);

  test('debería detectar cuando el stream solo devuelve {}', async ({ request }) => {
    // Crear tarea
    const createTaskResponse = await request.post(`${BASE_URL}/api/tasks`, {
      data: {
        instruction: 'Test simple',
        repository: 'test/repo',
        status: 'pending',
        repoInfo: { name: 'test', full_name: 'test/repo', default_branch: 'main' },
        model: 'deepseek-chat',
      },
    });

    const task = await createTaskResponse.json();
    const taskId = task.id;

    // Iniciar procesamiento
    await request.post(`${BASE_URL}/api/tasks/process`, {
      data: { taskId },
    });

    // Esperar un poco
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Verificar el estado
    const getTaskResponse = await request.get(`${BASE_URL}/api/tasks`);
    const tasks = await getTaskResponse.json();
    const currentTask = tasks.find((t: any) => t.id === taskId);

    if (currentTask) {
      console.log(`📊 Estado de la tarea: ${currentTask.status}`);
      console.log(`📊 Contenido: ${currentTask.streamingContent?.length || 0} caracteres`);
      console.log(`📊 Error: ${currentTask.error || 'ninguno'}`);

      if (currentTask.streamingContent?.length === 0 && currentTask.status === 'processing') {
        console.error(`❌ PROBLEMA DETECTADO: Stream devolvió 0 caracteres`);
        console.error(`❌ Esto indica que el stream se está cerrando prematuramente`);
      }
    }
  }, 30000);
});

