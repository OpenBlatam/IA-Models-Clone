/**
 * Test que simula exactamente lo que hace el backend worker
 * para encontrar el problema del stream que solo devuelve {}
 */

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';
const DEEPSEEK_STREAM_ENDPOINT = `${BASE_URL}/api/deepseek/stream`;

async function testBackendWorkerStream() {
  console.log('🧪 [TEST] Simulando backend worker...');
  console.log('🧪 [TEST] Endpoint:', DEEPSEEK_STREAM_ENDPOINT);
  console.log('🧪 [TEST] Base URL:', BASE_URL);

  const testInstruction = 'Crea un archivo README.md con el contenido "Hello World"';
  const testRepository = 'OpenBlatam/test-repo';

  // Simular exactamente lo que hace el backend worker
  const abortController = new AbortController();
  
  // Verificar periódicamente (simulando el checkStopInterval)
  const checkStopInterval = setInterval(() => {
    // En el test, nunca detenemos
  }, 2000);

  try {
    console.log('\n📤 [TEST] Realizando fetch (simulando backend worker)...');
    const fetchStartTime = Date.now();
    
    const response = await fetch(DEEPSEEK_STREAM_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: testInstruction,
        repository: testRepository,
        context: {},
        model: 'deepseek-chat',
      }),
      signal: abortController.signal,
    }).finally(() => {
      clearInterval(checkStopInterval);
    });

    const fetchTime = Date.now() - fetchStartTime;
    console.log(`📥 [TEST] Respuesta recibida en ${fetchTime}ms: ${response.status} ${response.statusText}`);
    console.log('📥 [TEST] Headers:', {
      contentType: response.headers.get('content-type'),
      hasBody: !!response.body,
      allHeaders: Object.fromEntries(response.headers.entries()),
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'No se pudo leer el error');
      console.error('❌ [TEST] Error en respuesta:', errorText);
      return;
    }

    if (!response.body) {
      console.error('❌ [TEST] La respuesta no tiene body');
      return;
    }

    // Procesar el stream exactamente como lo hace el backend worker
    console.log('\n📥 [TEST] Iniciando lectura del stream (simulando stream-processor)...');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedContent = '';
    let chunkCount = 0;
    let buffer = '';

    while (true) {
      let readResult;
      try {
        readResult = await reader.read();
      } catch (readError: any) {
        console.error(`❌ [TEST] Error leyendo del reader:`, {
          error: readError.message,
          name: readError.name,
        });
        if (readError.name === 'AbortError') {
          console.log(`🛑 [TEST] Stream cancelado por AbortController`);
          break;
        }
        throw readError;
      }

      const { done, value } = readResult;

      if (done) {
        console.log(`\n✅ [TEST] Stream marcado como done después de ${chunkCount} chunks`);
        console.log(`📊 [TEST] Estadísticas: ${chunkCount} chunks recibidos, ${accumulatedContent.length} caracteres acumulados, buffer: ${buffer.length} caracteres`);
        
        // Procesar buffer restante
        if (buffer.trim()) {
          console.log(`📥 [TEST] Procesando buffer restante (${buffer.length} caracteres): "${buffer}"`);
          try {
            const parsed = JSON.parse(buffer);
            if (parsed.content) {
              accumulatedContent += parsed.content;
              console.log(`✅ [TEST] Contenido del buffer agregado. Total: ${accumulatedContent.length} caracteres`);
            }
          } catch (parseError) {
            console.warn(`⚠️ [TEST] Error parseando buffer restante, agregando como texto directo`);
            accumulatedContent += buffer;
          }
        }
        break;
      }

      if (!value || value.length === 0) {
        continue;
      }

      chunkCount++;
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;

      // Procesar líneas completas
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      if (chunkCount <= 5) {
        console.log(`\n📥 [TEST] Chunk #${chunkCount} recibido:`);
        console.log(`   - Longitud: ${chunk.length} caracteres`);
        console.log(`   - Preview: ${chunk.substring(0, 200)}`);
        console.log(`   - Líneas completas: ${lines.length}`);
        console.log(`   - Buffer restante: ${buffer.length} caracteres`);
      }

      for (const line of lines) {
        if (!line.trim()) continue;

        if (chunkCount <= 5) {
          console.log(`   📄 Línea: ${line.substring(0, 100)}...`);
        }

        try {
          const parsed = JSON.parse(line);

          if (parsed.content) {
            accumulatedContent += parsed.content;
            if (chunkCount <= 5) {
              console.log(`   ✅ Contenido extraído: ${parsed.content.length} caracteres, total: ${accumulatedContent.length}`);
            }
          }

          if (parsed.error) {
            console.error(`   ❌ Error en stream:`, parsed.error);
          }
        } catch (parseError) {
          if (chunkCount <= 5) {
            console.warn(`   ⚠️ Error parseando línea:`, parseError);
            console.warn(`   ⚠️ Línea completa:`, line);
          }
        }
      }

      if (chunkCount % 10 === 0) {
        console.log(`📊 [TEST] Progreso: ${chunkCount} chunks, ${accumulatedContent.length} caracteres`);
      }
    }

    console.log('\n📊 [TEST] Resumen final:');
    console.log(`   - Chunks recibidos: ${chunkCount}`);
    console.log(`   - Contenido total: ${accumulatedContent.length} caracteres`);
    console.log(`   - Éxito: ${accumulatedContent.length > 0 ? '✅' : '❌'}`);

    if (chunkCount === 1 && accumulatedContent.length === 0) {
      console.error('\n❌ [TEST] PROBLEMA DETECTADO:');
      console.error(`   - Solo se recibió 1 chunk con contenido vacío`);
      console.error(`   - Esto indica que el stream se está cerrando prematuramente`);
      console.error(`   - Posibles causas:`);
      console.error(`     1. El AbortController se está cancelando inmediatamente`);
      console.error(`     2. El endpoint está devolviendo {} y cerrando el stream`);
      console.error(`     3. Hay un error que se está capturando silenciosamente`);
    }

  } catch (error: any) {
    console.error('\n❌ [TEST] Error durante el test:', error);
    console.error('Stack:', error.stack);
  }
}

testBackendWorkerStream().catch((error) => {
  console.error('❌ Error fatal:', error);
  process.exit(1);
});

