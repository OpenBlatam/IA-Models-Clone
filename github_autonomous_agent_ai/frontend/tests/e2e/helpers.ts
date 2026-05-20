/**
 * Utilidades compartidas para tests E2E
 * 
 * Contiene helpers reutilizables para todos los tests de Playwright
 */
import { Page, Locator } from '@playwright/test';

// ============================================================================
// Constants
// ============================================================================

export const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';
export const AGENT_CONTROL_PATH = '/agent-control';

// Timeouts (en milisegundos)
export const TIMEOUTS = {
  PAGE_LOAD: 30000,
  TASK_CREATION: 10000,
  POLL_INTERVAL: 1000,
  MAX_POLL_ATTEMPTS: 60,
  CONTENT_TIMEOUT_THRESHOLD: 10,
  TEST_TIMEOUT: 120000,
  LOG_CHECK_TIMEOUT: 30000,
  API_REQUEST: 30000,
} as const;

// Selectors
export const SELECTORS = {
  instructionTextarea: 'textarea[name="instruction"]',
  createButton: /crear|procesar/i,
  taskCard: '[data-testid="task-card"], .task-card, [class*="task"]',
} as const;

// Task statuses
export const TASK_STATUS = {
  COMPLETED: 'completed',
  PENDING_APPROVAL: 'pending_approval',
  FAILED: 'failed',
  PROCESSING: 'processing',
  PENDING: 'pending',
} as const;

// Problematic log patterns
export const PROBLEMATIC_LOG_PATTERNS = [
  'Stream se cerró prematuramente',
  '0 caracteres',
  '{}',
  'Stream se cerró',
] as const;

// ============================================================================
// Types
// ============================================================================

export interface TaskInfo {
  text: string | null;
  status: string;
  element: Locator | null;
}

export interface TaskMonitorResult {
  completed: boolean;
  contentReceived: boolean;
  finalStatus?: string;
  error?: string;
}

export interface ApiTask {
  id: string;
  status: string;
  instruction: string;
  streamingContent?: string;
  [key: string]: any;
}

// ============================================================================
// Navigation Helpers
// ============================================================================

/**
 * Navega a la página de control del agente y espera a que cargue
 */
export async function navigateToAgentControl(page: Page): Promise<void> {
  await page.goto(`${BASE_URL}${AGENT_CONTROL_PATH}`);
  await page.waitForLoadState('networkidle', { timeout: TIMEOUTS.PAGE_LOAD });
}

/**
 * Navega a una URL específica y espera a que cargue
 */
export async function navigateTo(page: Page, path: string): Promise<void> {
  await page.goto(`${BASE_URL}${path}`);
  await page.waitForLoadState('networkidle', { timeout: TIMEOUTS.PAGE_LOAD });
}

// ============================================================================
// Task Helpers
// ============================================================================

/**
 * Crea una nueva tarea desde el formulario
 */
export async function createTask(page: Page, instruction: string): Promise<void> {
  await page.fill(SELECTORS.instructionTextarea, instruction);
  const createButton = page.getByRole('button', { name: SELECTORS.createButton });
  await createButton.click();
}

/**
 * Espera a que aparezca al menos una tarea en la página
 */
export async function waitForTaskToAppear(page: Page, timeout?: number): Promise<void> {
  await page.waitForSelector(SELECTORS.taskCard, {
    timeout: timeout || TIMEOUTS.TASK_CREATION,
  });
}

/**
 * Obtiene información de todas las tareas visibles
 */
export async function getTaskInfo(page: Page): Promise<TaskInfo[]> {
  const taskElements = await page.$$(SELECTORS.taskCard);
  const tasks: TaskInfo[] = [];

  for (const element of taskElements) {
    const text = await element.textContent();
    const status = (await element.getAttribute('data-status')) || '';
    tasks.push({ text, status, element: element as any });
  }

  return tasks;
}

/**
 * Obtiene la última tarea (la más reciente)
 */
export async function getLastTask(page: Page): Promise<TaskInfo | null> {
  const tasks = await getTaskInfo(page);
  return tasks.length > 0 ? tasks[tasks.length - 1] : null;
}

/**
 * Verifica si una tarea está completada
 */
export function isTaskCompleted(task: TaskInfo | null): boolean {
  if (!task) return false;
  
  return (
    task.status === TASK_STATUS.COMPLETED ||
    task.status === TASK_STATUS.PENDING_APPROVAL ||
    task.text?.toLowerCase().includes('completada') === true
  );
}

/**
 * Verifica si una tarea falló
 */
export function isTaskFailed(task: TaskInfo | null): boolean {
  if (!task) return false;
  
  return (
    task.status === TASK_STATUS.FAILED ||
    task.text?.toLowerCase().includes('error') === true ||
    task.text?.toLowerCase().includes('falló') === true
  );
}

/**
 * Verifica si una tarea tiene contenido suficiente
 */
export function hasContent(task: TaskInfo | null, minLength: number = 50): boolean {
  return (task?.text?.length ?? 0) >= minLength;
}

// ============================================================================
// Monitoring Helpers
// ============================================================================

/**
 * Monitorea el progreso de una tarea hasta que se complete o falle
 */
export async function monitorTaskProgress(
  page: Page,
  onProgress?: (attempt: number, task: TaskInfo | null) => void
): Promise<TaskMonitorResult> {
  let taskCompleted = false;
  let contentReceived = false;
  let finalStatus: string | undefined;
  let error: string | undefined;

  for (let attempt = 0; attempt < TIMEOUTS.MAX_POLL_ATTEMPTS; attempt++) {
    await page.waitForTimeout(TIMEOUTS.POLL_INTERVAL);

    const task = await getLastTask(page);

    // Callback para logging o debugging
    if (onProgress && task) {
      onProgress(attempt + 1, task);
    }

    if (task) {
      // Verificar contenido
      if (hasContent(task)) {
        contentReceived = true;
      }

      // Verificar si está completada
      if (isTaskCompleted(task)) {
        taskCompleted = true;
        finalStatus = task.status;
        break;
      }

      // Verificar si falló
      if (isTaskFailed(task)) {
        error = `Tarea falló: ${task.text?.substring(0, 200)}`;
        break;
      }
    }

    // Verificar timeout de contenido
    if (attempt >= TIMEOUTS.CONTENT_TIMEOUT_THRESHOLD && !contentReceived) {
      await handleContentTimeout(page, attempt + 1);
      error = `No se recibió contenido después de ${attempt + 1} segundos`;
      break;
    }
  }

  return {
    completed: taskCompleted,
    contentReceived,
    finalStatus,
    error,
  };
}

/**
 * Maneja el caso cuando no se recibe contenido después del timeout
 */
export async function handleContentTimeout(
  page: Page,
  elapsedSeconds: number
): Promise<void> {
  // Tomar screenshot para debugging
  const screenshotPath = `test-results/content-timeout-${Date.now()}.png`;
  await page.screenshot({ path: screenshotPath });
  console.error(`📸 Screenshot guardado: ${screenshotPath}`);

  // Verificar estado en la API
  try {
    const apiResponse = await page.request.get(`${BASE_URL}/api/tasks`);
    const tasks: ApiTask[] = await apiResponse.json();
    const lastTask = tasks[tasks.length - 1];

    if (lastTask) {
      console.error(`❌ Estado API: ${lastTask.status}`);
      console.error(
        `❌ Contenido API: ${lastTask.streamingContent?.length || 0} caracteres`
      );
    }
  } catch (apiError) {
    console.error(`❌ Error al consultar API: ${apiError}`);
  }
}

// ============================================================================
// API Helpers
// ============================================================================

/**
 * Obtiene todas las tareas desde la API
 */
export async function getTasksFromApi(page: Page): Promise<ApiTask[]> {
  const response = await page.request.get(`${BASE_URL}/api/tasks`, {
    timeout: TIMEOUTS.API_REQUEST,
  });
  return await response.json();
}

/**
 * Obtiene una tarea específica desde la API
 */
export async function getTaskFromApi(page: Page, taskId: string): Promise<ApiTask | null> {
  try {
    const response = await page.request.get(`${BASE_URL}/api/tasks/${taskId}`, {
      timeout: TIMEOUTS.API_REQUEST,
    });
    return await response.json();
  } catch (error) {
    console.error(`Error al obtener tarea ${taskId}:`, error);
    return null;
  }
}

/**
 * Crea una tarea mediante la API
 */
export async function createTaskViaApi(
  page: Page,
  instruction: string,
  repository: string = 'test/repo',
  model: string = 'deepseek-chat'
): Promise<ApiTask> {
  const response = await page.request.post(`${BASE_URL}/api/tasks`, {
    data: {
      instruction,
      repository,
      status: 'pending',
      repoInfo: {
        name: repository.split('/')[0],
        full_name: repository,
        default_branch: 'main',
      },
      model,
    },
    timeout: TIMEOUTS.API_REQUEST,
  });

  if (!response.ok()) {
    throw new Error(`Error al crear tarea: ${response.status()}`);
  }

  return await response.json();
}

/**
 * Inicia el procesamiento de una tarea mediante la API
 */
export async function processTaskViaApi(
  page: Page,
  taskId: string
): Promise<boolean> {
  const response = await page.request.post(`${BASE_URL}/api/tasks/process`, {
    data: { taskId },
    timeout: TIMEOUTS.API_REQUEST,
  });

  return response.ok();
}

// ============================================================================
// Logging Helpers
// ============================================================================

/**
 * Configura un listener para capturar logs de la consola
 */
export function setupConsoleLogCapture(page: Page): string[] {
  const logs: string[] = [];

  page.on('console', (msg) => {
    const text = msg.text();
    logs.push(text);

    // Detectar problemas específicos
    if (PROBLEMATIC_LOG_PATTERNS.some((pattern) => text.includes(pattern))) {
      console.error(`❌ Log problemático detectado: ${text}`);
    }
  });

  return logs;
}

/**
 * Filtra logs problemáticos
 */
export function filterProblematicLogs(logs: string[]): string[] {
  return logs.filter((log) =>
    PROBLEMATIC_LOG_PATTERNS.some((pattern) => log.includes(pattern))
  );
}

// ============================================================================
// Utility Helpers
// ============================================================================

/**
 * Espera un tiempo específico
 */
export async function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Toma un screenshot con nombre descriptivo
 */
export async function takeScreenshot(
  page: Page,
  name: string
): Promise<string> {
  const path = `test-results/${name}-${Date.now()}.png`;
  await page.screenshot({ path });
  return path;
}

/**
 * Verifica que una respuesta de API sea exitosa
 */
export function expectApiSuccess(response: { ok: () => boolean; status: () => number }): void {
  if (!response.ok()) {
    throw new Error(`API request failed with status ${response.status()}`);
  }
}



