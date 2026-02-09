/**
 * Script de prueba para diagnosticar el problema del stream que se cierra prematuramente
 * 
 * Uso: npx tsx test-stream-debug.ts
 * O: node --loader ts-node/esm test-stream-debug.ts
 */

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

interface TestResult {
  name: string;
  success: boolean;
  error?: string;
  details?: any;
  contentLength?: number;
  chunksReceived?: number;
}

async function testStreamEndpoint(): Promise<TestResult> {
  const testName = 'Test: Endpoint de Stream de DeepSeek';
  console.log(`\n🧪 ${testName}`);
  console.log('='.repeat(60));

  try {
    const testInstruction = 'Crea un archivo README.md con "Hello World"';
    const testRepository = 'OpenBlatam/test-repo';

    console.log(`📤 Enviando solicitud a ${BASE_URL}/api/deepseek/stream`);
    console.log(`📝 Instrucción: ${testInstruction}`);
    console.log(`📦 Repositorio: ${testRepository}`);

    const response = await fetch(`${BASE_URL}/api/deepseek/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        instruction: testInstruction,
        repository: testRepository,
        context: {},
      }),
    });

    console.log(`📥 Respuesta recibida: ${response.status} ${response.statusText}`);
    console.log(`📋 Headers:`, {
      contentType: response.headers.get('content-type'),
      hasBody: !!response.body,
    });

    if (!response.ok) {
      const errorText = await response.text();
      return {
        name: testName,
        success: false,
        error: `Error ${response.status}: ${errorText}`,
        details: { status: response.status, statusText: response.statusText },
      };
    }

    if (!response.body) {
      return {
        name: testName,
        success: false,
        error: 'La respuesta no tiene body',
      };
    }

    // Leer el stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedContent = '';
    let chunkCount = 0;
    let totalBytes = 0;
    let jsonLines = 0;
    let contentChunks = 0;
    let errorChunks = 0;
    let doneChunks = 0;

    console.log(`\n📥 Iniciando lectura del stream...`);

    const timeout = setTimeout(() => {
      console.log(`⏱️ Timeout después de 30 segundos`);
      reader.cancel();
    }, 30000);

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          console.log(`\n✅ Stream terminado`);
          clearTimeout(timeout);
          break;
        }

        chunkCount++;
        totalBytes += value.length;

        const chunk = decoder.decode(value, { stream: true });
        accumulatedContent += chunk;

        // Parsear líneas JSON
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            jsonLines++;

            if (parsed.content) {
              contentChunks++;
              if (contentChunks <= 3) {
                console.log(`  ✅ Chunk con contenido #${contentChunks}: ${parsed.content.length} caracteres`);
                console.log(`     Preview: ${parsed.content.substring(0, 50)}...`);
              }
            }

            if (parsed.done) {
              doneChunks++;
              console.log(`  🏁 Señal de done recibida`);
            }

            if (parsed.error) {
              errorChunks++;
              console.log(`  ❌ Error en chunk: ${parsed.error}`);
            }
          } catch (parseError) {
            // No es JSON válido, podría ser contenido directo
            if (chunkCount <= 3) {
              console.log(`  ⚠️ Línea no JSON (chunk #${chunkCount}): ${line.substring(0, 100)}`);
            }
          }
        }

        // Log cada 10 chunks
        if (chunkCount % 10 === 0 || chunkCount <= 5) {
          console.log(`📊 Chunk #${chunkCount}: ${value.length} bytes, Total acumulado: ${accumulatedContent.length} caracteres`);
        }
      }
    } catch (readError: any) {
      clearTimeout(timeout);
      console.error(`❌ Error leyendo stream:`, readError);
      return {
        name: testName,
        success: false,
        error: readError.message,
        details: {
          chunksReceived: chunkCount,
          contentLength: accumulatedContent.length,
        },
      };
    }

    // Extraer solo el contenido real (no los JSON wrappers)
    let actualContent = '';
    const contentLines = accumulatedContent.split('\n').filter(line => line.trim());
    for (const line of contentLines) {
      try {
        const parsed = JSON.parse(line);
        if (parsed.content) {
          actualContent += parsed.content;
        }
      } catch {
        // Ignorar líneas que no son JSON
      }
    }

    console.log(`\n📊 Resumen del Stream:`);
    console.log(`   - Chunks recibidos: ${chunkCount}`);
    console.log(`   - Bytes totales: ${totalBytes}`);
    console.log(`   - Líneas JSON: ${jsonLines}`);
    console.log(`   - Chunks con contenido: ${contentChunks}`);
    console.log(`   - Chunks con error: ${errorChunks}`);
    console.log(`   - Señales de done: ${doneChunks}`);
    console.log(`   - Contenido acumulado (raw): ${accumulatedContent.length} caracteres`);
    console.log(`   - Contenido real extraído: ${actualContent.length} caracteres`);

    if (actualContent.length === 0) {
      return {
        name: testName,
        success: false,
        error: 'Stream terminó sin contenido',
        details: {
          chunksReceived: chunkCount,
          jsonLines,
          contentChunks,
          errorChunks,
          rawContentLength: accumulatedContent.length,
        },
        contentLength: 0,
        chunksReceived: chunkCount,
      };
    }

    return {
      name: testName,
      success: true,
      contentLength: actualContent.length,
      chunksReceived: chunkCount,
      details: {
        jsonLines,
        contentChunks,
        errorChunks,
        totalBytes,
      },
    };
  } catch (error: any) {
    return {
      name: testName,
      success: false,
      error: error.message,
      details: error,
    };
  }
}

async function testTaskProcessing(): Promise<TestResult> {
  const testName = 'Test: Procesamiento de Tarea en Backend';
  console.log(`\n🧪 ${testName}`);
  console.log('='.repeat(60));

  try {
    // Primero crear una tarea de prueba
    const testTask = {
      repository: 'OpenBlatam/test-repo',
      instruction: 'Crea un archivo README.md con "Hello World"',
      status: 'pending' as const,
      repoInfo: {
        name: 'test-repo',
        full_name: 'OpenBlatam/test-repo',
        description: 'Test repository',
        language: 'TypeScript',
        default_branch: 'main',
        html_url: 'https://github.com/OpenBlatam/test-repo',
        id: 12345,
      },
      model: 'deepseek-chat',
    };

    console.log(`📤 Creando tarea de prueba...`);
    const createResponse = await fetch(`${BASE_URL}/api/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task: testTask }),
    });

    if (!createResponse.ok) {
      const errorText = await createResponse.text();
      return {
        name: testName,
        success: false,
        error: `Error creando tarea: ${createResponse.status} - ${errorText}`,
      };
    }

    const { task } = await createResponse.json();
    console.log(`✅ Tarea creada: ${task.id}`);

    // Iniciar procesamiento
    console.log(`📤 Iniciando procesamiento...`);
    const processResponse = await fetch(`${BASE_URL}/api/tasks/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId: task.id }),
    });

    if (!processResponse.ok) {
      const errorText = await processResponse.text();
      return {
        name: testName,
        success: false,
        error: `Error iniciando procesamiento: ${processResponse.status} - ${errorText}`,
      };
    }

    console.log(`✅ Procesamiento iniciado`);

    // Esperar y verificar el estado
    console.log(`⏳ Esperando 15 segundos para que el stream procese...`);
    await new Promise(resolve => setTimeout(resolve, 15000));

    // Verificar el estado de la tarea
    const tasksResponse = await fetch(`${BASE_URL}/api/tasks`);
    const { tasks } = await tasksResponse.json();
    const updatedTask = tasks.find((t: any) => t.id === task.id);

    if (!updatedTask) {
      return {
        name: testName,
        success: false,
        error: 'Tarea no encontrada después del procesamiento',
      };
    }

    console.log(`📊 Estado de la tarea:`);
    console.log(`   - Status: ${updatedTask.status}`);
    console.log(`   - Streaming Content Length: ${updatedTask.streamingContent?.length || 0} caracteres`);
    console.log(`   - Error: ${updatedTask.error || 'Ninguno'}`);

    if (updatedTask.error?.includes('0 caracteres')) {
      return {
        name: testName,
        success: false,
        error: updatedTask.error,
        details: {
          status: updatedTask.status,
          streamingContentLength: updatedTask.streamingContent?.length || 0,
        },
      };
    }

    return {
      name: testName,
      success: updatedTask.status !== 'failed' && (updatedTask.streamingContent?.length || 0) > 0,
      contentLength: updatedTask.streamingContent?.length || 0,
      details: {
        status: updatedTask.status,
        hasStreamingContent: !!updatedTask.streamingContent,
      },
    };
  } catch (error: any) {
    return {
      name: testName,
      success: false,
      error: error.message,
      details: error,
    };
  }
}

async function testDirectDeepSeekAPI(): Promise<TestResult> {
  const testName = 'Test: API Directa de DeepSeek';
  console.log(`\n🧪 ${testName}`);
  console.log('='.repeat(60));

  try {
    const DEEPSEEK_API_KEY = process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY || 'sk-ae1c47feaa3e483b85a936430d1f494a';
    const DEEPSEEK_API_BASE_URL = process.env.NEXT_PUBLIC_DEEPSEEK_API_BASE_URL || 'https://api.deepseek.com';
    const DEEPSEEK_MODEL = process.env.NEXT_PUBLIC_DEEPSEEK_MODEL || 'deepseek-chat';

    console.log(`📤 Llamando directamente a la API de DeepSeek...`);
    console.log(`🔗 URL: ${DEEPSEEK_API_BASE_URL}/v1/chat/completions`);
    console.log(`🔑 API Key: ${DEEPSEEK_API_KEY.substring(0, 10)}...`);

    const response = await fetch(`${DEEPSEEK_API_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        messages: [
          {
            role: 'user',
            content: 'Responde solo con "Hello World"',
          },
        ],
        stream: true,
        max_tokens: 100,
      }),
    });

    console.log(`📥 Respuesta: ${response.status} ${response.statusText}`);

    if (!response.ok) {
      const errorText = await response.text();
      return {
        name: testName,
        success: false,
        error: `Error ${response.status}: ${errorText}`,
      };
    }

    if (!response.body) {
      return {
        name: testName,
        success: false,
        error: 'La respuesta no tiene body',
      };
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let chunkCount = 0;
    let contentLength = 0;

    console.log(`📥 Leyendo stream...`);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunkCount++;
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n').filter(line => line.trim() && line.startsWith('data: '));

      for (const line of lines) {
        const data = line.slice(6);
        if (data === '[DONE]') continue;

        try {
          const json = JSON.parse(data);
          const content = json.choices?.[0]?.delta?.content || '';
          if (content) {
            contentLength += content.length;
            if (chunkCount <= 3) {
              console.log(`  ✅ Contenido recibido: ${content}`);
            }
          }
        } catch (e) {
          // Ignorar errores de parseo
        }
      }

      if (chunkCount <= 5 || chunkCount % 10 === 0) {
        console.log(`📊 Chunk #${chunkCount}: ${contentLength} caracteres acumulados`);
      }
    }

    console.log(`\n📊 Resumen:`);
    console.log(`   - Chunks: ${chunkCount}`);
    console.log(`   - Contenido: ${contentLength} caracteres`);

    return {
      name: testName,
      success: contentLength > 0,
      contentLength,
      chunksReceived: chunkCount,
      details: {
        apiKeyValid: !!DEEPSEEK_API_KEY,
        apiUrl: DEEPSEEK_API_BASE_URL,
      },
    };
  } catch (error: any) {
    return {
      name: testName,
      success: false,
      error: error.message,
      details: error,
    };
  }
}

async function runAllTests() {
  console.log('🚀 Iniciando tests de diagnóstico del stream');
  console.log('='.repeat(60));
  console.log(`🌐 Base URL: ${BASE_URL}`);
  console.log(`⏰ Fecha: ${new Date().toISOString()}`);

  const results: TestResult[] = [];

  // Test 1: API directa de DeepSeek
  results.push(await testDirectDeepSeekAPI());

  // Test 2: Endpoint de stream
  results.push(await testStreamEndpoint());

  // Test 3: Procesamiento completo de tarea
  results.push(await testTaskProcessing());

  // Resumen
  console.log('\n' + '='.repeat(60));
  console.log('📊 RESUMEN DE TESTS');
  console.log('='.repeat(60));

  results.forEach((result, index) => {
    const icon = result.success ? '✅' : '❌';
    console.log(`\n${icon} Test ${index + 1}: ${result.name}`);
    if (!result.success) {
      console.log(`   Error: ${result.error}`);
    }
    if (result.contentLength !== undefined) {
      console.log(`   Contenido: ${result.contentLength} caracteres`);
    }
    if (result.chunksReceived !== undefined) {
      console.log(`   Chunks: ${result.chunksReceived}`);
    }
    if (result.details) {
      console.log(`   Detalles:`, JSON.stringify(result.details, null, 2));
    }
  });

  const successCount = results.filter(r => r.success).length;
  const totalCount = results.length;

  console.log(`\n📈 Resultados: ${successCount}/${totalCount} tests pasaron`);

  if (successCount < totalCount) {
    console.log('\n❌ Algunos tests fallaron. Revisa los detalles arriba.');
    process.exit(1);
  } else {
    console.log('\n✅ Todos los tests pasaron!');
    process.exit(0);
  }
}

// Ejecutar tests
runAllTests().catch((error) => {
  console.error('❌ Error ejecutando tests:', error);
  process.exit(1);
});

