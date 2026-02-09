/**
 * Script de debug para diagnosticar el problema del stream de DeepSeek
 * que solo devuelve {} y termina inmediatamente
 */

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';
const DEEPSEEK_STREAM_ENDPOINT = `${BASE_URL}/api/deepseek/stream`;

interface TestResult {
  success: boolean;
  error?: string;
  chunksReceived: number;
  totalContentLength: number;
  firstChunk?: string;
  lastChunk?: string;
  chunks: Array<{ number: number; content: string; length: number }>;
  responseStatus?: number;
  responseHeaders?: Record<string, string>;
}

async function testDeepSeekStream(): Promise<TestResult> {
  const result: TestResult = {
    success: false,
    chunksReceived: 0,
    totalContentLength: 0,
    chunks: [],
  };

  console.log('🧪 [TEST] Iniciando test del stream de DeepSeek...');
  console.log('🧪 [TEST] Endpoint:', DEEPSEEK_STREAM_ENDPOINT);
  console.log('🧪 [TEST] Base URL:', BASE_URL);

  const testInstruction = 'Crea un archivo README.md con el contenido "Hello World"';
  const testRepository = 'OpenBlatam/test-repo';

  try {
    console.log('\n📤 [TEST] Enviando solicitud al endpoint...');
    console.log('📤 [TEST] Payload:', {
      instruction: testInstruction.substring(0, 50) + '...',
      repository: testRepository,
    });

    const startTime = Date.now();
    const response = await fetch(DEEPSEEK_STREAM_ENDPOINT, {
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

    const responseTime = Date.now() - startTime;
    console.log(`\n📥 [TEST] Respuesta recibida en ${responseTime}ms`);
    console.log('📥 [TEST] Status:', response.status, response.statusText);
    console.log('📥 [TEST] Headers:', Object.fromEntries(response.headers.entries()));

    result.responseStatus = response.status;
    result.responseHeaders = Object.fromEntries(response.headers.entries());

    if (!response.ok) {
      const errorText = await response.text();
      result.error = `HTTP ${response.status}: ${errorText}`;
      console.error('❌ [TEST] Error en respuesta:', result.error);
      return result;
    }

    if (!response.body) {
      result.error = 'La respuesta no tiene body';
      console.error('❌ [TEST] La respuesta no tiene body');
      return result;
    }

    console.log('\n📥 [TEST] Iniciando lectura del stream...');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let chunkNumber = 0;
    let totalLength = 0;

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        console.log(`\n✅ [TEST] Stream terminado después de ${chunkNumber} chunks`);
        console.log(`📊 [TEST] Total de contenido: ${totalLength} caracteres`);
        console.log(`📊 [TEST] Buffer restante: ${buffer.length} caracteres`);
        
        if (buffer.trim()) {
          console.log(`📥 [TEST] Procesando buffer restante: "${buffer}"`);
          result.chunks.push({
            number: chunkNumber + 1,
            content: buffer,
            length: buffer.length,
          });
          totalLength += buffer.length;
        }
        break;
      }

      chunkNumber++;
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;

      // Procesar líneas completas
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      console.log(`\n📥 [TEST] Chunk #${chunkNumber} recibido:`);
      console.log(`   - Longitud: ${chunk.length} caracteres`);
      console.log(`   - Preview: ${chunk.substring(0, 200)}`);
      console.log(`   - Líneas completas: ${lines.length}`);
      console.log(`   - Buffer restante: ${buffer.length} caracteres`);

      if (chunkNumber === 1) {
        result.firstChunk = chunk;
      }
      result.lastChunk = chunk;

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (!line.trim()) continue;

        console.log(`   📄 Línea ${i + 1}: ${line.substring(0, 100)}...`);

        try {
          const parsed = JSON.parse(line);
          console.log(`   ✅ JSON válido:`, {
            hasContent: !!parsed.content,
            contentLength: parsed.content?.length || 0,
            hasDone: 'done' in parsed,
            done: parsed.done,
            hasError: !!parsed.error,
            keys: Object.keys(parsed),
          });

          if (parsed.content) {
            totalLength += parsed.content.length;
            result.chunks.push({
              number: chunkNumber,
              content: parsed.content,
              length: parsed.content.length,
            });
          }

          if (parsed.error) {
            console.error(`   ❌ Error en stream:`, parsed.error);
            result.error = parsed.error;
          }
        } catch (parseError: any) {
          console.warn(`   ⚠️ Error parseando línea como JSON:`, parseError.message);
          console.warn(`   ⚠️ Línea completa:`, line);
        }
      }
    }

    result.chunksReceived = chunkNumber;
    result.totalContentLength = totalLength;
    result.success = totalLength > 0;

    console.log('\n📊 [TEST] Resumen final:');
    console.log(`   - Chunks recibidos: ${result.chunksReceived}`);
    console.log(`   - Contenido total: ${result.totalContentLength} caracteres`);
    console.log(`   - Chunks con contenido: ${result.chunks.length}`);
    console.log(`   - Éxito: ${result.success ? '✅' : '❌'}`);

    if (!result.success && result.chunksReceived > 0) {
      console.error('\n❌ [TEST] PROBLEMA DETECTADO:');
      console.error(`   - Se recibieron ${result.chunksReceived} chunks pero 0 caracteres de contenido`);
      console.error(`   - Primer chunk: "${result.firstChunk}"`);
      console.error(`   - Último chunk: "${result.lastChunk}"`);
      console.error(`   - Esto indica que el stream se está cerrando prematuramente o no está enviando contenido`);
    }

  } catch (error: any) {
    result.error = error.message;
    console.error('\n❌ [TEST] Error durante el test:', error);
    console.error('Stack:', error.stack);
  }

  return result;
}

async function testDirectDeepSeekAPI(): Promise<void> {
  console.log('\n🧪 [TEST] Probando llamada directa a DeepSeek API...');
  
  const DEEPSEEK_API_KEY = process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY || 'sk-ae1c47feaa3e483b85a936430d1f494a';
  const DEEPSEEK_API_BASE_URL = process.env.NEXT_PUBLIC_DEEPSEEK_API_BASE_URL || 'https://api.deepseek.com';
  const DEEPSEEK_MODEL = process.env.NEXT_PUBLIC_DEEPSEEK_MODEL || 'deepseek-chat';

  try {
    const testInstruction = 'Crea un archivo README.md con el contenido "Hello World"';
    
    console.log('📤 [TEST] Llamando directamente a DeepSeek API...');
    console.log('📤 [TEST] URL:', `${DEEPSEEK_API_BASE_URL}/v1/chat/completions`);
    console.log('📤 [TEST] Model:', DEEPSEEK_MODEL);
    console.log('📤 [TEST] API Key:', DEEPSEEK_API_KEY ? `${DEEPSEEK_API_KEY.substring(0, 10)}...` : 'NO KEY');

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
            role: 'system',
            content: 'Eres un asistente experto en desarrollo de software.',
          },
          {
            role: 'user',
            content: testInstruction,
          },
        ],
        temperature: 0.7,
        max_tokens: 100,
        stream: true,
      }),
    });

    console.log('📥 [TEST] Respuesta de DeepSeek:', {
      status: response.status,
      statusText: response.statusText,
      ok: response.ok,
      hasBody: !!response.body,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ [TEST] Error en respuesta de DeepSeek:', errorText);
      return;
    }

    if (!response.body) {
      console.error('❌ [TEST] La respuesta de DeepSeek no tiene body');
      return;
    }

    console.log('📥 [TEST] Leyendo stream de DeepSeek...');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let chunkCount = 0;
    let totalContent = 0;

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        console.log(`\n✅ [TEST] Stream de DeepSeek terminado: ${chunkCount} chunks, ${totalContent} caracteres`);
        break;
      }

      chunkCount++;
      const chunk = decoder.decode(value, { stream: true });
      
      if (chunkCount <= 5) {
        console.log(`📥 [TEST] Chunk #${chunkCount} de DeepSeek:`, {
          length: chunk.length,
          preview: chunk.substring(0, 200),
        });
      }

      // Buscar contenido en el chunk
      if (chunk.includes('"content"')) {
        const contentMatch = chunk.match(/"content":"([^"]*)"/);
        if (contentMatch) {
          totalContent += contentMatch[1].length;
          if (chunkCount <= 5) {
            console.log(`   ✅ Contenido encontrado: ${contentMatch[1].length} caracteres`);
          }
        }
      }
    }

    if (totalContent === 0 && chunkCount > 0) {
      console.error('\n❌ [TEST] PROBLEMA: DeepSeek está enviando chunks pero sin contenido');
    } else if (totalContent > 0) {
      console.log(`\n✅ [TEST] DeepSeek está funcionando correctamente: ${totalContent} caracteres recibidos`);
    }

  } catch (error: any) {
    console.error('❌ [TEST] Error llamando a DeepSeek directamente:', error);
  }
}

async function main() {
  console.log('🚀 Iniciando tests de debug del stream de DeepSeek...\n');
  console.log('='.repeat(80));

  // Test 1: Llamada directa a DeepSeek API
  console.log('\n📋 TEST 1: Llamada directa a DeepSeek API');
  console.log('-'.repeat(80));
  await testDirectDeepSeekAPI();

  // Test 2: Llamada al endpoint local
  console.log('\n\n📋 TEST 2: Llamada al endpoint local /api/deepseek/stream');
  console.log('-'.repeat(80));
  const result = await testDeepSeekStream();

  // Resumen final
  console.log('\n\n' + '='.repeat(80));
  console.log('📊 RESUMEN FINAL');
  console.log('='.repeat(80));
  console.log('Test 1 (DeepSeek directo):', result.chunksReceived > 0 ? '✅ Chunks recibidos' : '❌ Sin chunks');
  console.log('Test 2 (Endpoint local):', result.success ? '✅ Éxito' : '❌ Falló');
  console.log(`   - Chunks: ${result.chunksReceived}`);
  console.log(`   - Contenido: ${result.totalContentLength} caracteres`);
  if (result.error) {
    console.log(`   - Error: ${result.error}`);
  }

  if (!result.success && result.chunksReceived === 1 && result.totalContentLength === 0) {
    console.log('\n🔍 DIAGNÓSTICO:');
    console.log('   El endpoint está devolviendo solo {} y cerrando el stream inmediatamente.');
    console.log('   Esto sugiere que:');
    console.log('   1. El ReadableStream se está cerrando antes de que DeepSeek responda');
    console.log('   2. Hay un error que se está capturando silenciosamente');
    console.log('   3. El método start() del ReadableStream no se está ejecutando correctamente');
    console.log('\n   Revisa los logs del servidor para ver los logs [DEEPSEEK] que agregamos.');
  }

  process.exit(result.success ? 0 : 1);
}

main().catch((error) => {
  console.error('❌ Error fatal:', error);
  process.exit(1);
});

