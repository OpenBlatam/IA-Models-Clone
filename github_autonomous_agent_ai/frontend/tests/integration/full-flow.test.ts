/**
 * Test de integración completo del flujo end-to-end
 */
import { describe, it, expect } from 'vitest';

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

describe('Full Flow Integration Test', () => {
  it('debería completar el flujo completo: crear -> procesar -> aprobar -> ejecutar', async () => {
    console.log('🧪 Iniciando test de flujo completo...');

    // Paso 1: Crear tarea
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
    console.log(`✅ Paso 1: Tarea creada - ${taskId}`);

    // Paso 2: Iniciar procesamiento
    const processResponse = await fetch(`${BASE_URL}/api/tasks/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId }),
    });

    expect(processResponse.ok).toBe(true);
    console.log(`✅ Paso 2: Procesamiento iniciado`);

    // Paso 3: Monitorear hasta que termine
    let finalStatus = 'processing';
    let maxWait = 60;
    let contentReceived = false;

    for (let i = 0; i < maxWait; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const getResponse = await fetch(`${BASE_URL}/api/tasks`);
      const tasks = await getResponse.json();
      const currentTask = tasks.find((t: any) => t.id === taskId);

      if (currentTask) {
        const status = currentTask.status;
        const contentLength = currentTask.streamingContent?.length || 0;

        if (i % 5 === 0) {
          console.log(`📊 Segundo ${i + 1}: Estado=${status}, Contenido=${contentLength}`);
        }

        if (contentLength > 0) {
          contentReceived = true;
          console.log(`✅ Contenido recibido: ${contentLength} caracteres`);
        }

        if (status === 'completed' || status === 'pending_approval' || status === 'failed') {
          finalStatus = status;
          console.log(`✅ Procesamiento terminado: ${status}`);
          break;
        }

        // Detectar problema temprano
        if (i >= 10 && contentLength === 0 && status === 'processing') {
          console.error(`❌ PROBLEMA DETECTADO en segundo ${i + 1}:`);
          console.error(`   - Estado: ${status}`);
          console.error(`   - Contenido: ${contentLength}`);
          console.error(`   - Error: ${currentTask.error || 'ninguno'}`);
          
          // Diagnosticar
          await diagnoseProblem(currentTask, BASE_URL);
          
          throw new Error(`Stream no está devolviendo contenido después de ${i + 1} segundos`);
        }
      }
    }

    // Verificar resultado
    const finalResponse = await fetch(`${BASE_URL}/api/tasks`);
    const finalTasks = await finalResponse.json();
    const finalTask = finalTasks.find((t: any) => t.id === taskId);

    expect(finalTask).toBeTruthy();
    
    if (finalTask.status === 'failed') {
      console.error(`❌ Tarea falló: ${finalTask.error}`);
      throw new Error(`Tarea falló: ${finalTask.error}`);
    }

    if (!contentReceived) {
      console.error(`❌ PROBLEMA: No se recibió contenido durante el procesamiento`);
      console.error(`❌ Estado final: ${finalTask.status}`);
      console.error(`❌ Contenido final: ${finalTask.streamingContent?.length || 0}`);
      throw new Error('No se recibió contenido durante el procesamiento');
    }

    expect(finalTask.streamingContent?.length || 0).toBeGreaterThan(0);
    console.log(`✅ Test completado exitosamente`);
  }, 120000);
});

async function diagnoseProblem(task: any, baseUrl: string) {
  console.log('\n🔍 DIAGNÓSTICO:');
  
  // Test 1: Verificar endpoint directamente
  console.log('📤 Test 1: Llamando directamente a /api/deepseek/stream...');
  try {
    const directResponse = await fetch(`${baseUrl}/api/deepseek/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: task.instruction,
        repository: task.repository,
        context: {},
      }),
    });

    const directText = await directResponse.text();
    console.log(`📥 Respuesta directa: ${directText.length} caracteres`);
    console.log(`📥 Preview: ${directText.substring(0, 200)}`);
    
    if (directText === '{}' || directText.length < 10) {
      console.error(`❌ El endpoint está devolviendo solo {} o muy poco contenido`);
    } else {
      console.log(`✅ El endpoint funciona correctamente`);
    }
  } catch (error: any) {
    console.error(`❌ Error llamando al endpoint: ${error.message}`);
  }

  // Test 2: Verificar logs del servidor (si están disponibles)
  console.log('\n📋 Test 2: Revisa los logs del servidor para ver:');
  console.log('   - [DEEPSEEK] logs del endpoint');
  console.log('   - [BACKEND] logs del worker');
  console.log('   - [STREAM] logs del procesador');
}

