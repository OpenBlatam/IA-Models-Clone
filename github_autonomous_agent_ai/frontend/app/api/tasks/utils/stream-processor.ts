/**
 * Utilidades para procesar streams de respuesta
 */

// Tipo para el reader del stream (compatible con Node.js y Web APIs)
export type StreamReader = {
  read(): Promise<{ done: boolean; value?: Uint8Array }>;
  cancel(): Promise<void>;
  releaseLock(): void;
};

export interface StreamProcessorOptions {
  taskId: string;
  reader: StreamReader;
  onContent: (content: string) => void;
  onError?: (error: any) => void;
  checkStopped?: () => Promise<boolean>;
  minContentLength?: number;
}

export interface StreamProcessorResult {
  success: boolean;
  contentLength: number;
  chunksReceived: number;
  error?: string;
}

/**
 * Procesa un stream de respuesta JSON línea por línea
 */
export async function processStream(
  options: StreamProcessorOptions
): Promise<StreamProcessorResult> {
  const {
    taskId,
    reader,
    onContent,
    onError,
    checkStopped,
    minContentLength = 500,
  } = options;

  const decoder = new TextDecoder();
  let accumulatedContent = '';
  let chunkCount = 0;
  let buffer = '';

  try {
    let lastStopCheck = Date.now();
    const STOP_CHECK_INTERVAL = 500; // Verificar cada 500ms
    
    while (true) {
      // Verificar si el reader está todavía disponible
      if (!reader) {
        console.error(`❌ [STREAM] Reader no disponible para tarea ${taskId}`);
        break;
      }
      
      // Verificar si la tarea fue detenida - verificar más frecuentemente
      const now = Date.now();
      if (checkStopped && (now - lastStopCheck >= STOP_CHECK_INTERVAL)) {
        const isStopped = await checkStopped();
        if (isStopped) {
          console.log(`🛑 [STREAM] Tarea ${taskId} fue detenida, deteniendo stream inmediatamente`);
          // Intentar cancelar el reader
          try {
            await reader.cancel();
          } catch (cancelError) {
            console.warn(`⚠️ [STREAM] Error cancelando reader:`, cancelError);
          }
          break;
        }
        lastStopCheck = now;
      }

      let readResult;
      try {
        readResult = await reader.read();
      } catch (readError: any) {
        console.error(`❌ [STREAM] Error leyendo del reader:`, {
          error: readError.message,
          name: readError.name,
          taskId,
        });
        if (readError.name === 'AbortError') {
          console.log(`🛑 [STREAM] Stream cancelado por AbortController para tarea ${taskId}`);
          break;
        }
        throw readError;
      }

      const { done, value } = readResult;

      if (done) {
        console.log(`✅ [STREAM] Stream marcado como done para tarea ${taskId}`);
        console.log(`📊 [STREAM] Estadísticas: ${chunkCount} chunks recibidos, ${accumulatedContent.length} caracteres acumulados, buffer: ${buffer.length} caracteres`);
        
        // Procesar buffer restante
        if (buffer.trim()) {
          console.log(`📥 [STREAM] Procesando buffer restante (${buffer.length} caracteres)`);
          try {
            const parsed = JSON.parse(buffer);
            if (parsed.content) {
              accumulatedContent += parsed.content;
              onContent(parsed.content);
              console.log(`✅ [STREAM] Contenido del buffer agregado. Total: ${accumulatedContent.length} caracteres`);
            }
          } catch (parseError) {
            console.warn(`⚠️ [STREAM] Error parseando buffer restante, agregando como texto directo`);
            accumulatedContent += buffer;
          }
        }

        // Validar contenido mínimo
        if (chunkCount === 0) {
          console.error(`❌ [STREAM] Stream terminó sin recibir ningún chunk`);
          // NO lanzar error - retornar resultado con error para que el procesador lo maneje
          return {
            success: false,
            contentLength: 0,
            chunksReceived: 0,
            error: 'Stream terminó sin recibir ningún chunk',
          };
        }

        if (accumulatedContent.length < minContentLength) {
          console.warn(
            `⚠️ [STREAM] Tarea ${taskId} terminó con poco contenido ` +
            `(${accumulatedContent.length} caracteres, mínimo: ${minContentLength})`
          );
        } else {
          console.log(`✅ [STREAM] Tarea ${taskId} tiene contenido suficiente (${accumulatedContent.length} caracteres)`);
          console.log(`✅ [STREAM] Stream completado - el contenido será procesado para generar el plan`);
        }

        // IMPORTANTE: No cambiar el status aquí - el status se mantiene en 'processing'
        // hasta que el plan esté completamente procesado en processStreamResult
        break;
      }

      if (!value || value.length === 0) {
        continue;
      }

      chunkCount++;
      const chunk = decoder.decode(value, { stream: true });
      
      // Log detallado de los primeros chunks
      if (chunkCount <= 10) {
        console.log(`📥 [STREAM] Chunk #${chunkCount} recibido:`, {
          chunkLength: chunk.length,
          chunkPreview: chunk.substring(0, 100),
          hasNewlines: chunk.includes('\n'),
        });
      }
      
      buffer += chunk;

      // Procesar líneas completas
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      if (chunkCount <= 10) {
        console.log(`📥 [STREAM] Chunk #${chunkCount} dividido en ${lines.length} líneas completas, buffer: ${buffer.length} caracteres`);
      }

      for (const line of lines) {
        if (!line.trim()) continue;

        try {
          // Intentar parsear como JSON
          const parsed = JSON.parse(line);

          if (parsed.content) {
            accumulatedContent += parsed.content;
            onContent(parsed.content);
            if (chunkCount <= 10) {
              console.log(`📥 [STREAM] Contenido extraído del chunk #${chunkCount}: ${parsed.content.length} caracteres, total acumulado: ${accumulatedContent.length}`);
            }
          }

          if (parsed.error && onError) {
            console.error(`❌ [STREAM] Error en stream:`, parsed.error);
            onError(parsed.error);
          }

          // Si tiene done: true, el stream terminó pero continuamos hasta que reader.read() devuelva done
          if (parsed.done === true) {
            console.log(`✅ [STREAM] Señal de done recibida en chunk #${chunkCount}, pero continuando lectura...`);
          }
        } catch (parseError) {
          // Si no es JSON válido, podría ser contenido directo o una línea incompleta
          if (chunkCount <= 10) {
            console.warn(`⚠️ [STREAM] Error parseando línea del chunk #${chunkCount}:`, parseError);
            console.warn(`⚠️ [STREAM] Línea que falló:`, line.substring(0, 200));
          }
          // Solo agregar si parece ser contenido real (no solo caracteres de control)
          if (line.length > 2 && !line.startsWith('data:') && !line.startsWith('{') && !line.startsWith('[')) {
            accumulatedContent += line;
            onContent(line);
          }
        }
      }
      
      // Log periódico del progreso
      if (chunkCount % 50 === 0 || (chunkCount <= 10 && chunkCount % 5 === 0)) {
        console.log(`📊 [STREAM] Progreso: ${chunkCount} chunks, ${accumulatedContent.length} caracteres acumulados`);
      }
    }

    const result = {
      success: accumulatedContent.length >= minContentLength,
      contentLength: accumulatedContent.length,
      chunksReceived: chunkCount,
    };

    console.log(`✅ [STREAM] Stream procesado exitosamente:`, {
      contentLength: result.contentLength,
      chunksReceived: result.chunksReceived,
      minContentLength,
      success: result.success,
    });

    return result;
  } catch (error: any) {
    console.error(`❌ [STREAM] Error procesando stream:`, error);
    
    // Asegurar que el reader se libere en caso de error
    try {
      if (reader) {
        reader.releaseLock();
      }
    } catch (releaseError) {
      // El reader ya podría estar liberado o el stream cerrado
      console.warn(`⚠️ [STREAM] No se pudo liberar el reader después del error:`, releaseError);
    }
    
    return {
      success: false,
      contentLength: accumulatedContent.length,
      chunksReceived: chunkCount,
      error: error.message,
    };
  } finally {
    // Asegurar que el reader se libere siempre
    try {
      if (reader) {
        reader.releaseLock();
      }
    } catch (releaseError) {
      // El reader ya podría estar liberado
      // No hacer nada, esto es normal si el stream ya terminó
    }
  }
}

/**
 * Actualiza el progreso de una tarea periódicamente
 */
export async function updateTaskProgress(
  taskId: string,
  content: string,
  updateTask: (taskId: string, updates: any) => Promise<void>,
  intervalMs: number = 2000
): Promise<() => void> {
  let lastUpdate = Date.now();
  let updateInterval: NodeJS.Timeout | null = null;

  const update = async () => {
    const now = Date.now();
    if (now - lastUpdate >= intervalMs) {
      try {
        await updateTask(taskId, {
          streamingContent: content,
          status: 'processing',
        });
        lastUpdate = now;
      } catch (error) {
        console.error(`Error actualizando progreso de tarea ${taskId}:`, error);
      }
    }
  };

  updateInterval = setInterval(update, intervalMs);

  // Retornar función para limpiar el intervalo
  return () => {
    if (updateInterval) {
      clearInterval(updateInterval);
    }
  };
}

