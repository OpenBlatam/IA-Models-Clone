import { NextRequest, NextResponse } from 'next/server';
import { extractPartialJSON } from '../../../utils/json-helpers';
import { githubExecutor } from '../../../lib/github-executor';
import * as taskStorage from '../utils/task-storage';
import { processStream, StreamProcessorResult } from '../utils/stream-processor';
import { buildTaskContext, getApiEndpoint } from '../utils/task-context';

export const maxDuration = 600;
export const dynamic = 'force-dynamic';

// Set global para evitar procesamiento duplicado
const processingTasks = new Set<string>();

/**
 * POST: Iniciar procesamiento de una tarea en background
 */
export async function POST(request: NextRequest) {
  try {
    const { taskId } = await validateRequest(request);
    const task = await findAndValidateTask(taskId);
    const processing = await loadProcessingSet();

    if (await shouldSkipProcessing(taskId, task, processing)) {
      return NextResponse.json(
        { message: 'La tarea ya está siendo procesada' },
        { status: 200 }
      );
    }

    await prepareTaskForProcessing(taskId, task, processing);
    const requestOrigin = extractRequestOrigin(request);

    startBackgroundProcessing(taskId, task, requestOrigin);

    return NextResponse.json({
      success: true,
      message: 'Procesamiento iniciado en background'
    });
  } catch (error: any) {
    return handleProcessingError(error);
  }
}

/**
 * Valida la solicitud y extrae el taskId
 */
async function validateRequest(request: NextRequest): Promise<{ taskId: string }> {
  const body = await request.json();
  const { taskId } = body;

  if (!taskId) {
    throw new Error('Se requiere taskId');
  }

  return { taskId };
}

/**
 * Busca y valida que la tarea exista
 */
async function findAndValidateTask(taskId: string): Promise<any> {
  const task = await taskStorage.findTask(taskId);

  if (!task) {
    throw new Error('Tarea no encontrada');
  }

  return task;
}

/**
 * Carga el conjunto de procesamiento, manejando errores
 */
async function loadProcessingSet(): Promise<Set<string>> {
  try {
    return await taskStorage.loadProcessing();
  } catch (error: any) {
    console.error('❌ [API] Error cargando processing:', error);
    return new Set<string>();
  }
}

/**
 * Determina si se debe omitir el procesamiento
 */
async function shouldSkipProcessing(
  taskId: string,
  task: any,
  processing: Set<string>
): Promise<boolean> {
  const isCurrentlyProcessing = processing.has(taskId) || processingTasks.has(taskId);
  const taskStatus = task.status;
  const hasPlan = task.pendingApproval?.actions?.length > 0;

  // Si está procesando Y tiene plan pendiente, no reiniciar
  if (isCurrentlyProcessing && hasPlan) {
    return true;
  }

  // Si la tarea está en 'processing' sin plan, permitir reinicio
  if (taskStatus === 'processing' && !hasPlan && isCurrentlyProcessing) {
    await clearProcessingForRestart(taskId, processing);
    return false;
  }

  // Si ya está procesando, omitir
  if (isCurrentlyProcessing) {
    return true;
  }

  return false;
}

/**
 * Limpia el procesamiento para permitir reinicio
 */
async function clearProcessingForRestart(
  taskId: string,
  processing: Set<string>
): Promise<void> {
  processing.delete(taskId);
  processingTasks.delete(taskId);
  await taskStorage.saveProcessing(processing);
  console.log(`🔄 [API] Limpiado procesamiento anterior para permitir reinicio continuo`);
}

/**
 * Prepara la tarea para procesamiento
 */
async function prepareTaskForProcessing(
  taskId: string,
  task: any,
  processing: Set<string>
): Promise<void> {
  await markTaskAsProcessing(taskId, processing);
  await updateTaskStatus(taskId);
}

/**
 * Marca la tarea como procesando
 */
async function markTaskAsProcessing(
  taskId: string,
  processing: Set<string>
): Promise<void> {
  try {
    processing.add(taskId);
    processingTasks.add(taskId);
    await taskStorage.saveProcessing(processing);
  } catch (error: any) {
    console.error('❌ [API] Error guardando processing:', error);
  }
}

/**
 * Actualiza el estado de la tarea a processing
 */
async function updateTaskStatus(taskId: string): Promise<void> {
  try {
    await taskStorage.updateTask(taskId, {
      status: 'processing',
      processingStartedAt: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('❌ [API] Error actualizando tarea:', error);
  }
}

/**
 * Extrae el origin del request
 */
function extractRequestOrigin(request: NextRequest): string | null {
  return request.url ? new URL(request.url).origin : null;
}

/**
 * Inicia el procesamiento en background
 */
function startBackgroundProcessing(
  taskId: string,
  task: any,
  requestOrigin: string | null
): void {
  processTaskInBackground(taskId, task, requestOrigin).catch((error) => {
    console.error(`❌ [API] Error procesando tarea ${taskId}:`, error);
    processingTasks.delete(taskId);
  });
}

/**
 * Maneja errores del procesamiento
 */
function handleProcessingError(error: any): NextResponse {
  console.error('❌ [API] Error starting task processing:', error);

  return NextResponse.json(
    {
      error: 'Error al iniciar procesamiento',
      details: error.message,
      name: error.name,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    },
    {
      status: error.message === 'Se requiere taskId' ? 400 :
        error.message === 'Tarea no encontrada' ? 404 : 500
    }
  );
}

/**
 * Intentar delegar el procesamiento al backend unificado.
 * Retorna true si el backend manejó la tarea, false para fallback local.
 */
async function tryBackendProcessing(taskId: string, task: any): Promise<boolean> {
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8030';

  try {
    console.log(`🔄 [BACKEND-UNIFIED] Intentando delegar tarea ${taskId} al backend unificado (${backendUrl})...`);

    // Verificar que el backend esté disponible
    const healthResponse = await fetch(`${backendUrl}/health`, {
      signal: AbortSignal.timeout(5000),
    });

    if (!healthResponse.ok) {
      console.warn(`⚠️ [BACKEND-UNIFIED] Backend no saludable (${healthResponse.status}), usando fallback local`);
      return false;
    }

    // Iniciar el agente si no está corriendo
    await fetch(`${backendUrl}/api/v1/agent/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(5000),
    });

    // Crear o actualizar la tarea en el backend
    const [owner, repo] = (task.repository || '').split('/');
    if (owner && repo) {
      try {
        await fetch(`${backendUrl}/api/v1/tasks/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            repository_owner: owner,
            repository_name: repo,
            instruction: task.instruction,
            metadata: {
              frontendTaskId: taskId,
              model: task.model || 'deepseek-chat',
            },
          }),
          signal: AbortSignal.timeout(10000),
        });
        console.log(`✅ [BACKEND-UNIFIED] Tarea creada/delegada al backend unificado`);
      } catch (createError: any) {
        console.warn(`⚠️ [BACKEND-UNIFIED] Error creando tarea en backend (no-crítico):`, createError.message);
      }
    }

    console.log(`✅ [BACKEND-UNIFIED] Backend disponible y agente iniciado`);
    // El backend manejará el procesamiento, pero seguimos con el fallback local 
    // para mantener la funcionalidad de streaming del frontend
    return false; // Retornar false para usar también el procesamiento local de streaming

  } catch (error: any) {
    console.warn(`⚠️ [BACKEND-UNIFIED] Backend no disponible (${error.message}), usando procesamiento local`);
    return false;
  }
}

/**
 * Procesar tarea en background
 */
async function processTaskInBackground(taskId: string, task: any, requestOrigin?: string | null): Promise<void> {
  try {
    logProcessingStart(taskId, task);

    // Intentar delegar al backend unificado primero
    await tryBackendProcessing(taskId, task);

    const baseUrl = resolveBaseUrl(requestOrigin);
    const model = task.model || 'deepseek-chat';
    const context = buildTaskContext(task);
    const apiEndpoint = getApiEndpoint(model, baseUrl);
    const checkIfStopped = createStopChecker(taskId);

    await processStreamWithMonitoring(taskId, task, apiEndpoint, context, model, checkIfStopped, requestOrigin);
  } catch (error: any) {
    await handleProcessingError(taskId, error);
  }
}

/**
 * Registra el inicio del procesamiento
 */
function logProcessingStart(taskId: string, task: any): void {
  console.log(`🔄 [BACKEND] Iniciando procesamiento en background para tarea ${taskId}`);
  console.log(`📋 [BACKEND] Instrucción: ${task.instruction.substring(0, 100)}...`);
  console.log(`📦 [BACKEND] Modelo: ${task.model || 'deepseek-chat'}`);
}

/**
 * Resuelve la URL base para las peticiones
 */
function resolveBaseUrl(requestOrigin?: string | null): string {
  const baseUrl = requestOrigin || process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

  if (baseUrl.includes(':3000') && !baseUrl.includes('3001')) {
    console.warn(`⚠️ [BACKEND] La URL base usa el puerto 3000, pero debería usar 3001 para el frontend`);
  }

  return baseUrl;
}

/**
 * Crea una función para verificar si la tarea fue detenida
 */
function createStopChecker(taskId: string): () => Promise<boolean> {
  return async (): Promise<boolean> => {
    try {
      const currentTask = await taskStorage.findTask(taskId);
      return currentTask?.status === 'stopped';
    } catch (error) {
      console.warn(`⚠️ [BACKEND] Error verificando estado de tarea ${taskId}:`, error);
      return false;
    }
  };
}

/**
 * Procesa el stream con monitoreo y actualización de progreso
 */
async function processStreamWithMonitoring(
  taskId: string,
  task: any,
  apiEndpoint: string,
  context: any,
  model: string,
  checkIfStopped: () => Promise<boolean>,
  requestOrigin?: string | null
): Promise<void> {
  const abortController = new AbortController();
  const checkStopInterval = setupStopMonitoring(taskId, abortController, checkIfStopped);

  try {
    const response = await fetchStreamResponse(apiEndpoint, task, context, model, abortController);
    await validateStreamResponse(response, apiEndpoint);

    const reader = await getStreamReader(response);
    const { accumulatedContent, streamResult } = await processStreamContent(
      taskId,
      reader,
      checkIfStopped,
      abortController
    );

    await handleStreamResult(taskId, task, accumulatedContent, streamResult, requestOrigin);
  } finally {
    clearInterval(checkStopInterval);
  }
}

/**
 * Configura el monitoreo de detención de tarea
 */
function setupStopMonitoring(
  taskId: string,
  abortController: AbortController,
  checkIfStopped: () => Promise<boolean>
): NodeJS.Timeout {
  return setInterval(async () => {
    const isStopped = await checkIfStopped();
    if (isStopped) {
      console.log(`🛑 [BACKEND] Tarea ${taskId} fue detenida, cancelando fetch inmediatamente`);
      abortController.abort();
      await cleanupProcessingOnStop(taskId);
    }
  }, 500);
}

/**
 * Limpia el procesamiento cuando la tarea es detenida
 */
async function cleanupProcessingOnStop(taskId: string): Promise<void> {
  processingTasks.delete(taskId);
  const processing = await taskStorage.loadProcessing();
  processing.delete(taskId);
  await taskStorage.saveProcessing(processing);
}

/**
 * Realiza el fetch del stream
 */
async function fetchStreamResponse(
  apiEndpoint: string,
  task: any,
  context: any,
  model: string,
  abortController: AbortController
): Promise<Response> {
  console.log(`📤 [BACKEND] Llamando a ${apiEndpoint}...`);

  const fetchStartTime = Date.now();
  const response = await fetch(apiEndpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      instruction: task.instruction,
      repository: task.repository,
      context: context,
      model: model,
    }),
    signal: abortController.signal,
  });

  const fetchTime = Date.now() - fetchStartTime;
  console.log(`📥 [BACKEND] Respuesta recibida en ${fetchTime}ms: ${response.status} ${response.statusText}`);

  return response;
}

/**
 * Valida que la respuesta sea un stream válido
 */
async function validateStreamResponse(
  response: Response,
  apiEndpoint: string
): Promise<void> {
  const contentType = response.headers.get('content-type');
  const isStream = contentType?.includes('text/event-stream') || contentType?.includes('stream');
  const isJson = contentType?.includes('application/json');

  if (!response.ok) {
    const errorText = await extractErrorFromResponse(response, isJson);
    throw new Error(`Error en API de DeepSeek: ${response.statusText} - ${errorText}`);
  }

  if (isJson && !isStream) {
    const errorMessage = await extractJsonError(response, apiEndpoint);
    throw new Error(`Error: ${errorMessage}. El endpoint debería devolver un stream, no JSON. Verifica que la URL ${apiEndpoint} sea correcta y que el servidor esté corriendo.`);
  }

  if (!response.body) {
    throw new Error('La respuesta del stream no tiene body');
  }

  if (!isStream) {
    console.warn(`⚠️ [BACKEND] Content-Type inesperado: ${contentType}`);
  }
}

/**
 * Extrae el error de una respuesta
 */
async function extractErrorFromResponse(response: Response, isJson: boolean): Promise<string> {
  try {
    return isJson ? await response.text() : await response.clone().text();
  } catch (error: any) {
    console.warn(`⚠️ [BACKEND] No se pudo leer el error de la respuesta:`, error.message);
    return `Error ${response.status}: ${response.statusText}`;
  }
}

/**
 * Extrae el error de una respuesta JSON
 */
async function extractJsonError(response: Response, apiEndpoint: string): Promise<string> {
  try {
    const jsonContent = await response.json();
    if (jsonContent.error) {
      return jsonContent.error;
    } else if (typeof jsonContent === 'object' && Object.keys(jsonContent).length === 0) {
      return 'El endpoint devolvió un objeto vacío {} - posible error en el endpoint o la URL está incorrecta';
    } else {
      return JSON.stringify(jsonContent);
    }
  } catch (error) {
    try {
      return await response.text() || 'El endpoint devolvió JSON en lugar de un stream';
    } catch {
      return 'El endpoint devolvió JSON en lugar de un stream';
    }
  }
}

/**
 * Obtiene el reader del stream
 */
async function getStreamReader(response: Response): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  try {
    return response.body!.getReader();
  } catch (readerError: any) {
    if (readerError.message?.includes('locked')) {
      throw new Error('El stream está bloqueado. No se puede obtener el reader.');
    }
    throw readerError;
  }
}

/**
 * Procesa el contenido del stream
 */
async function processStreamContent(
  taskId: string,
  reader: ReadableStreamDefaultReader<Uint8Array>,
  checkIfStopped: () => Promise<boolean>,
  abortController: AbortController
): Promise<{ accumulatedContent: string; streamResult: StreamProcessorResult }> {
  let accumulatedContent = '';
  const contentRef = { value: '' };
  const progressInterval = setupProgressMonitoring(taskId, () => accumulatedContent, abortController);

  try {
    const streamResult = await processStream({
      taskId,
      reader,
      onContent: (content: string) => {
        contentRef.value += content;
        accumulatedContent += content;
      },
      onError: (error: any) => {
        console.error(`❌ [BACKEND] Error en stream para tarea ${taskId}:`, error);
      },
      checkStopped: checkIfStopped,
      minContentLength: 500,
    });

    if (contentRef.value.length > accumulatedContent.length) {
      accumulatedContent = contentRef.value;
    }

    return { accumulatedContent, streamResult };
  } finally {
    clearInterval(progressInterval);
  }
}

/**
 * Configura el monitoreo de progreso
 */
function setupProgressMonitoring(
  taskId: string,
  getContent: () => string,
  abortController: AbortController
): NodeJS.Timeout {
  let lastUpdate = Date.now();

  return setInterval(async () => {
    const now = Date.now();
    const accumulatedContent = getContent();

    if (now - lastUpdate > 2000 && accumulatedContent.length > 0) {
      try {
        const currentTask = await taskStorage.findTask(taskId);
        if (!currentTask) {
          return;
        }

        if (currentTask.status === 'stopped') {
          console.log(`🛑 [BACKEND] Tarea ${taskId} fue detenida durante actualización de progreso`);
          abortController.abort();
          return;
        }

        await taskStorage.updateTask(taskId, {
          streamingContent: accumulatedContent,
          status: 'processing',
          error: undefined,
        });
        lastUpdate = now;
        console.log(`💾 [BACKEND] Progreso guardado para tarea ${taskId} (${accumulatedContent.length} caracteres)`);
      } catch (error) {
        console.error(`Error actualizando progreso:`, error);
      }
    }
  }, 2000);
}

/**
 * Maneja el resultado del stream procesado
 */
async function handleStreamResult(
  taskId: string,
  task: any,
  accumulatedContent: string,
  streamResult: StreamProcessorResult,
  requestOrigin?: string | null
): Promise<void> {
  console.log(`📊 [BACKEND] Stream procesado: ${accumulatedContent.length} caracteres, ${streamResult.chunksReceived} chunks, éxito: ${streamResult.success}`);

  if (!streamResult.success) {
    await handleUnsuccessfulStream(taskId, accumulatedContent, streamResult);
    return;
  }

  const finalContentLength = Math.max(accumulatedContent.length, streamResult.contentLength);

  if (finalContentLength < 500) {
    await handleInsufficientContent(taskId, task, accumulatedContent);
    return;
  }

  await saveFinalProgress(taskId, accumulatedContent);
  await processStreamResult(taskId, task, accumulatedContent);

  const finalTaskState = await taskStorage.findTask(taskId);
  await handleTaskProcessingCompletion(taskId, finalTaskState, requestOrigin);
}

/**
 * Maneja un stream no exitoso
 */
async function handleUnsuccessfulStream(
  taskId: string,
  accumulatedContent: string,
  streamResult: StreamProcessorResult
): Promise<void> {
  console.error(`❌ [BACKEND] Stream no fue exitoso para tarea ${taskId}`);
  console.error(`❌ [BACKEND] Razón: ${streamResult.error || 'Contenido insuficiente'}`);

  const currentTask = await taskStorage.findTask(taskId);
  if (currentTask?.status === 'stopped') {
    await cleanupProcessingOnStop(taskId);
    return;
  }

  if (streamResult.error && accumulatedContent.length < 500) {
    await taskStorage.updateTask(taskId, {
      status: 'failed',
      error: streamResult.error,
      streamingContent: accumulatedContent,
    });
    await cleanupProcessing(taskId, 'Error crítico en stream');
  }
}

/**
 * Guarda el progreso final
 */
async function saveFinalProgress(taskId: string, accumulatedContent: string): Promise<void> {
  if (accumulatedContent.length === 0) {
    return;
  }

  try {
    const currentTask = await taskStorage.findTask(taskId);
    if (currentTask && currentTask.status !== 'stopped') {
      await taskStorage.updateTask(taskId, {
        streamingContent: accumulatedContent,
        status: 'processing',
      });
      console.log(`💾 [BACKEND] Progreso guardado: ${accumulatedContent.length} caracteres, status: processing`);
    }
  } catch (error) {
    console.error(`Error actualizando progreso final:`, error);
  }
}

/**
 * Maneja errores del procesamiento en background
 */
async function handleProcessingError(taskId: string, error: any): Promise<void> {
  console.error(`❌ [BACKEND] Error procesando tarea ${taskId}:`, error);
  console.error(`❌ [BACKEND] Stack trace:`, error.stack);

  const currentTask = await taskStorage.findTask(taskId);
  if (currentTask?.status === 'stopped') {
    await cleanupProcessingOnStop(taskId);
    return;
  }

  const isCriticalError = isCriticalProcessingError(error);

  if (isCriticalError) {
    await taskStorage.updateTask(taskId, {
      status: 'failed',
      error: error.message || 'Error desconocido',
    });
  } else {
    await taskStorage.updateTask(taskId, {
      status: 'processing',
      error: `Error temporal: ${error.message || 'Error desconocido'}. El procesamiento continuará.`,
    });
  }

  await cleanupProcessing(taskId, 'Error en procesamiento');
}

/**
 * Determina si un error es crítico
 */
function isCriticalProcessingError(error: any): boolean {
  return error.name === 'TypeError' ||
    error.message?.includes('fetch failed') ||
    error.message?.includes('network') ||
    error.message?.includes('timeout');
}

/**
 * Maneja la finalización del procesamiento de una tarea
 * Decide si limpiar el procesamiento o mantenerlo activo según el estado
 */
async function handleTaskProcessingCompletion(
  taskId: string,
  finalTaskState: any,
  requestOrigin?: string | null
): Promise<void> {
  if (!finalTaskState) {
    console.warn(`⚠️ [BACKEND] No se encontró estado final para tarea ${taskId}`);
    return;
  }

  const status = finalTaskState.status;
  const hasPlan = finalTaskState.pendingApproval?.actions?.length > 0;

  switch (status) {
    case 'stopped':
      await cleanupProcessing(taskId, 'Tarea detenida por el usuario');
      break;

    case 'pending_approval':
      await cleanupProcessing(taskId, 'Plan pendiente de aprobación');
      console.log(`✅ [BACKEND] La tarea ${taskId} seguirá activa esperando aprobación del usuario`);
      break;

    case 'processing':
      await handleProcessingContinuation(taskId, finalTaskState, requestOrigin, hasPlan);
      break;

    default:
      console.log(`🔄 [BACKEND] Tarea ${taskId} en estado '${status}' - manteniendo procesamiento activo`);
  }

  console.log(`✅ [BACKEND] Resultado procesado completamente para tarea ${taskId}`);
  console.log(`✅ [BACKEND] Estado final: ${status}`);
}

/**
 * Limpia el procesamiento de una tarea
 */
async function cleanupProcessing(taskId: string, reason: string): Promise<void> {
  console.log(`🔄 [BACKEND] Limpiando procesamiento para tarea ${taskId}: ${reason}`);
  const processing = await taskStorage.loadProcessing();
  processing.delete(taskId);
  await taskStorage.saveProcessing(processing);
  processingTasks.delete(taskId);
}

/**
 * Maneja la continuación del procesamiento para tareas en estado 'processing'
 */
async function handleProcessingContinuation(
  taskId: string,
  taskState: any,
  requestOrigin: string | null | undefined,
  hasPlan: boolean
): Promise<void> {
  console.log(`🔄 [BACKEND] Tarea ${taskId} sigue en 'processing' - NO limpiando procesamiento`);
  console.log(`🔄 [BACKEND] La tarea continuará procesando hasta que el usuario presione pausa`);
  console.log(`🔄 [BACKEND] El procesamiento se mantiene activo en el Set para continuidad`);

  // Si no tiene plan, reiniciar automáticamente el procesamiento para que continúe indefinidamente
  if (!hasPlan) {
    await scheduleProcessingRestart(taskId, requestOrigin);
  }
}

/**
 * Programa el reinicio automático del procesamiento para tareas sin plan
 */
async function scheduleProcessingRestart(
  taskId: string,
  requestOrigin: string | null | undefined
): Promise<void> {
  console.log(`🔄 [BACKEND] Tarea ${taskId} en 'processing' sin plan - programando reinicio automático`);
  console.log(`🔄 [BACKEND] Esto asegura que el agente continúe trabajando indefinidamente`);

  // Usar setTimeout con delay mínimo para el siguiente tick del event loop
  // Esto evita loops infinitos inmediatos y permite que el sistema respire
  setTimeout(async () => {
    try {
      const checkTask = await taskStorage.findTask(taskId);

      if (!shouldRestartProcessing(checkTask)) {
        return;
      }

      console.log(`🔄 [BACKEND] Reiniciando procesamiento para tarea ${taskId}...`);

      // Reiniciar procesamiento llamando a processTaskInBackground nuevamente
      await processTaskInBackground(taskId, checkTask, requestOrigin);
    } catch (error: any) {
      console.error(`❌ [BACKEND] Error reiniciando procesamiento para tarea ${taskId}:`, error);
    }
  });
}

/**
 * Verifica si una tarea debe reiniciar su procesamiento
 */
function shouldRestartProcessing(task: any): boolean {
  if (!task) {
    return false;
  }

  const isProcessing = task.status === 'processing';
  const hasNoPlan = !task.pendingApproval?.actions?.length;

  return isProcessing && hasNoPlan;
}

/**
 * Maneja el caso de contenido insuficiente
 */
async function handleInsufficientContent(
  taskId: string,
  task: any,
  accumulatedContent: string
): Promise<void> {
  console.log(`🔄 [BACKEND] Reintentando procesamiento de tarea ${taskId}...`);

  await taskStorage.updateTask(taskId, {
    status: 'processing',
    streamingContent: accumulatedContent,
    error: `Stream se cerró prematuramente con muy poco contenido (${accumulatedContent.length} caracteres). Reintentando...`,
  });

  const processing = await taskStorage.loadProcessing();
  processing.delete(taskId);
  await taskStorage.saveProcessing(processing);
  processingTasks.delete(taskId);

  // Reintentar después de 5 segundos
  setTimeout(async () => {
    try {
      console.log(`🔄 [BACKEND] Reintentando procesamiento de tarea ${taskId}...`);
      const processing = await taskStorage.loadProcessing();
      if (!processing.has(taskId) && !processingTasks.has(taskId)) {
        processing.add(taskId);
        processingTasks.add(taskId);
        await taskStorage.saveProcessing(processing);
        processTaskInBackground(taskId, task).catch((error) => {
          console.error(`❌ [BACKEND] Error en reintento de tarea ${taskId}:`, error);
          processingTasks.delete(taskId);
        });
      }
    } catch (error) {
      console.error(`❌ [BACKEND] Error al reintentar tarea ${taskId}:`, error);
      processingTasks.delete(taskId);
    }
  }, 5000);
}

/**
 * Procesa el resultado del stream
 */
async function processStreamResult(
  taskId: string,
  task: any,
  accumulatedContent: string
): Promise<void> {
  console.log(`📝 [BACKEND] Parseando resultado para tarea ${taskId}...`);
  console.log(`📊 [BACKEND] Contenido acumulado: ${accumulatedContent.length} caracteres`);

  // Parsear el resultado
  let plan: any = null;
  let explanation = '';
  let code = null;
  let commitMessage = '';

  try {
    const cleanContent = accumulatedContent.trim();
    console.log(`📝 [BACKEND] Contenido limpio: ${cleanContent.length} caracteres`);

    // Intentar extraer JSON del contenido
    const jsonMatch = extractPartialJSON(cleanContent);
    if (jsonMatch) {
      console.log(`📝 [BACKEND] JSON encontrado en contenido (${jsonMatch.length} caracteres)`);
      try {
        const parsed = JSON.parse(jsonMatch);
        plan = parsed.plan || parsed;
        explanation = parsed.explanation || parsed.content || '';
        code = parsed.code || null;
        commitMessage = parsed.commit_message || parsed.commitMessage || '';
        console.log(`✅ [BACKEND] JSON parseado exitosamente`);
      } catch (e) {
        console.warn('⚠️ [BACKEND] Error parseando JSON extraído:', e);
      }
    } else {
      console.warn(`⚠️ [BACKEND] No se encontró JSON válido en el contenido`);
    }

    // Si no hay plan, intentar crear uno básico
    if (!plan && cleanContent.length > 0) {
      console.log(`ℹ️ [BACKEND] Creando plan básico desde contenido`);
      plan = {
        steps: [cleanContent.substring(0, 1000)],
        files_to_create: [],
        files_to_modify: [],
      };
      explanation = cleanContent;
    }
  } catch (e) {
    console.warn('⚠️ [BACKEND] Error procesando resultado:', e);
    explanation = accumulatedContent;
  }

  // IMPORTANTE: Asegurar que el plan esté completamente procesado antes de cambiar el status
  // El status se mantiene en 'processing' hasta que TODO esté listo

  // Verificar si la tarea fue detenida antes de procesar el plan
  const taskBeforePlan = await taskStorage.findTask(taskId);
  if (taskBeforePlan?.status === 'stopped') {
    console.log(`🛑 [BACKEND] Tarea ${taskId} fue detenida antes de procesar el plan, preservando contenido`);
    // Preservar el contenido y plan parcial si existe
    await taskStorage.updateTask(taskId, {
      streamingContent: accumulatedContent,
      status: 'stopped',
      result: {
        content: explanation,
        plan: plan || { steps: [], files_to_create: [], files_to_modify: [] },
        code: code,
      },
    });
    return;
  }

  // Preparar acciones si hay plan
  let actions: any[] = [];
  if (plan && (plan.files_to_create?.length > 0 || plan.files_to_modify?.length > 0)) {
    console.log(`📋 [BACKEND] Plan encontrado para tarea ${taskId}:`, {
      filesToCreate: plan.files_to_create?.length || 0,
      filesToModify: plan.files_to_modify?.length || 0,
    });

    try {
      console.log(`🔄 [BACKEND] Preparando acciones del plan...`);
      actions = githubExecutor.prepareActionsFromPlan(plan);
      const finalCommitMessage = commitMessage || githubExecutor.generateCommitMessage(plan, task.instruction);

      console.log(`✅ [BACKEND] Plan completamente procesado: ${actions.length} acciones preparadas`);
      console.log(`✅ [BACKEND] Commit message: ${finalCommitMessage.substring(0, 100)}...`);

      // Verificar nuevamente si fue detenida antes de cambiar el status
      const taskBeforeStatusChange = await taskStorage.findTask(taskId);
      if (taskBeforeStatusChange?.status === 'stopped') {
        console.log(`🛑 [BACKEND] Tarea ${taskId} fue detenida durante procesamiento del plan, preservando plan parcial`);
        await taskStorage.updateTask(taskId, {
          streamingContent: accumulatedContent,
          status: 'stopped',
          result: {
            content: explanation,
            plan: plan,
            code: code,
          },
          pendingApproval: {
            plan,
            commitMessage: finalCommitMessage,
            actions: actions,
          },
        });
        return;
      }

      console.log(`🔄 [BACKEND] Cambiando status a 'pending_approval' - el plan está listo para aprobación`);

      // SOLO AHORA cambiar el status a pending_approval - cuando TODO está listo
      await taskStorage.updateTask(taskId, {
        status: 'pending_approval',
        streamingContent: accumulatedContent,
        result: {
          content: explanation,
          plan: plan,
          code: code,
        },
        pendingApproval: {
          plan,
          commitMessage: finalCommitMessage,
          actions: actions,
        },
      });

      console.log(`✅ [BACKEND] Tarea ${taskId} lista para aprobación (${actions.length} acciones)`);
      console.log(`✅ [BACKEND] El usuario ahora puede ver el plan y decidir si aprobar o rechazar`);
    } catch (planError: any) {
      console.error(`❌ [BACKEND] Error procesando plan para tarea ${taskId}:`, planError);
      // Si hay error procesando el plan, guardar el contenido y plan parcial
      await taskStorage.updateTask(taskId, {
        status: 'pending_approval',
        streamingContent: accumulatedContent,
        result: {
          content: explanation,
          plan: plan,
          code: code,
        },
        pendingApproval: {
          plan,
          commitMessage: commitMessage || `feat: ${task.instruction.substring(0, 50)}...`,
          actions: [],
        },
        error: `Error procesando plan: ${planError.message}`,
      });
    }
  } else {
    console.log(`ℹ️ [BACKEND] Tarea ${taskId} completada sin plan de archivos`);
    console.log(`🔄 [BACKEND] Manteniendo status en 'processing' - NO se detiene automáticamente`);
    console.log(`🔄 [BACKEND] La tarea continuará procesando hasta que el usuario presione pausa`);

    // NO cambiar a 'completed' - mantener en 'processing' para que continúe
    // Solo actualizar el contenido y resultado, pero mantener el status activo
    await taskStorage.updateTask(taskId, {
      status: 'processing', // Mantener en processing - NO detener automáticamente
      streamingContent: accumulatedContent,
      result: {
        content: explanation,
        plan: plan || { steps: [], files_to_create: [], files_to_modify: [] },
        code: code,
      },
    });

    console.log(`✅ [BACKEND] Tarea ${taskId} actualizada con contenido - continúa en 'processing'`);
    console.log(`✅ [BACKEND] El usuario puede presionar pausa cuando desee detenerla`);
  }

  console.log(`✅ [BACKEND] Procesamiento de resultado completado para tarea ${taskId}`);
}
