/**
 * E2E Tests específicos para debuguear el problema del stream
 */
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

test.describe('Stream Debug E2E', () => {
  test('debería verificar que el endpoint /api/deepseek/stream funciona', async ({ request }) => {
    console.log('🧪 Test: Verificando endpoint /api/deepseek/stream');

    const response = await request.post(`${BASE_URL}/api/deepseek/stream`, {
      data: {
        instruction: 'Crea un archivo test.txt',
        repository: 'test/repo',
        context: {},
      },
      timeout: 60000,
    });

    expect(response.ok()).toBeTruthy();
    expect(response.status()).toBe(200);

    const contentType = response.headers()['content-type'];
    console.log(`📥 Content-Type: ${contentType}`);
    expect(contentType).toContain('text/event-stream');

    // Leer el stream
    const body = await response.body();
    const text = new TextDecoder().decode(body);
    
    console.log(`📊 Tamaño de la respuesta: ${body.length} bytes`);
    console.log(`📊 Primeros 500 caracteres: ${text.substring(0, 500)}`);

    expect(body.length).toBeGreaterThan(2); // Más que solo {}
    expect(text).not.toBe('{}');
    
    // Verificar que hay contenido JSON válido
    const lines = text.split('\n').filter(line => line.trim());
    let hasContent = false;
    
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        if (parsed.content) {
          hasContent = true;
          console.log(`✅ Contenido encontrado: ${parsed.content.length} caracteres`);
          break;
        }
      } catch (e) {
        // Continuar
      }
    }

    expect(hasContent).toBe(true);
  }, 60000);

  test('debería verificar el flujo completo: crear tarea -> procesar -> verificar contenido', async ({ request }) => {
    console.log('🧪 Test: Flujo completo de procesamiento');

    // Paso 1: Crear tarea
    const createResponse = await request.post(`${BASE_URL}/api/tasks`, {
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

    expect(createResponse.ok()).toBeTruthy();
    const task = await createResponse.json();
    const taskId = task.id;
    console.log(`✅ Paso 1: Tarea creada - ${taskId}`);

    // Paso 2: Iniciar procesamiento
    const processResponse = await request.post(`${BASE_URL}/api/tasks/process`, {
      data: { taskId },
    });

    expect(processResponse.ok()).toBeTruthy();
    console.log(`✅ Paso 2: Procesamiento iniciado`);

    // Paso 3: Monitorear el progreso
    let lastContentLength = 0;
    let stableCount = 0;
    const maxStable = 5; // Si el contenido no cambia en 5 segundos, hay problema

    for (let i = 0; i < 30; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const getResponse = await request.get(`${BASE_URL}/api/tasks`);
      const tasks = await getResponse.json();
      const currentTask = tasks.find((t: any) => t.id === taskId);

      if (currentTask) {
        const contentLength = currentTask.streamingContent?.length || 0;
        const status = currentTask.status;

        console.log(`📊 Segundo ${i + 1}: Estado=${status}, Contenido=${contentLength} caracteres`);

        if (contentLength > lastContentLength) {
          lastContentLength = contentLength;
          stableCount = 0;
          console.log(`✅ Contenido aumentó a ${contentLength} caracteres`);
        } else if (contentLength === lastContentLength && contentLength > 0) {
          stableCount++;
        }

        if (status === 'completed' || status === 'pending_approval') {
          console.log(`✅ Tarea completada: ${status}`);
          expect(contentLength).toBeGreaterThan(0);
          return;
        }

        if (status === 'failed') {
          console.error(`❌ Tarea falló: ${currentTask.error}`);
          throw new Error(`Tarea falló: ${currentTask.error}`);
        }

        // Si el contenido no cambia y es 0, hay un problema
        if (contentLength === 0 && i > 5) {
          console.error(`❌ PROBLEMA: Después de ${i + 1} segundos, el contenido sigue siendo 0`);
          console.error(`❌ Estado: ${status}`);
          console.error(`❌ Error: ${currentTask.error || 'ninguno'}`);
          
          // Verificar directamente el endpoint
          const directResponse = await request.post(`${BASE_URL}/api/deepseek/stream`, {
            data: {
              instruction: currentTask.instruction,
              repository: currentTask.repository,
              context: {},
            },
          });
          
          const directBody = await directResponse.body();
          const directText = new TextDecoder().decode(directBody);
          console.error(`❌ Respuesta directa del endpoint: ${directText.substring(0, 200)}`);
          
          throw new Error(`Stream no está devolviendo contenido después de ${i + 1} segundos`);
        }
      }
    }

    // Si llegamos aquí, la tarea no se completó en 30 segundos
    const finalResponse = await request.get(`${BASE_URL}/api/tasks`);
    const finalTasks = await finalResponse.json();
    const finalTask = finalTasks.find((t: any) => t.id === taskId);

    if (finalTask) {
      console.error(`❌ Tarea no completada después de 30 segundos`);
      console.error(`❌ Estado final: ${finalTask.status}`);
      console.error(`❌ Contenido final: ${finalTask.streamingContent?.length || 0} caracteres`);
      console.error(`❌ Error: ${finalTask.error || 'ninguno'}`);
    }

    // No fallar el test, solo reportar
    expect(finalTask?.streamingContent?.length || 0).toBeGreaterThan(0);
  }, 120000);
});

