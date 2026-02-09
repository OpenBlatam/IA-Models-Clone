/**
 * Pasos reutilizables para tests E2E
 * 
 * Encapsula flujos comunes de testing
 */
import { Page } from '@playwright/test';
import { AgentControlPage } from '../page-objects/agent-control-page';
import {
  navigateToAgentControl,
  createTask,
  waitForTaskToAppear,
  monitorTaskProgress,
  setupConsoleLogCapture,
  filterProblematicLogs,
  type TaskInfo,
  TIMEOUTS,
} from '../helpers';
import {
  waitForCondition,
  interceptNetworkRequests,
} from '../test-utils';
import {
  createMetricsTracker,
  collectPerformanceMetrics,
  type TestMetrics,
} from './metrics';
import {
  generateReport,
  generateTextReport,
  saveReport,
} from './reporting';
import { expectTaskSuccess } from './assertions';

/**
 * Flujo completo: crear tarea y monitorear progreso
 */
export async function completeTaskFlow(
  page: Page,
  instruction: string,
  onProgress?: (attempt: number, task: TaskInfo | null) => void
): Promise<void> {
  const agentPage = new AgentControlPage(page);

  // 1. Navegar a la página
  await agentPage.goto();
  console.log('✅ Página cargada');

  // 2. Crear tarea
  await agentPage.createTask(instruction);
  console.log('✅ Instrucción ingresada y botón clickeado');

  // 3. Esperar a que aparezca la tarea
  await agentPage.waitForTask();
  console.log('✅ Tarea creada');

  // 4. Monitorear progreso
  const result = await monitorTaskProgress(page, onProgress);

  // 5. Verificar resultado
  expectTaskSuccess(result);

  if (result.completed) {
    console.log(`✅ Tarea completada con estado: ${result.finalStatus}`);
  } else if (result.contentReceived) {
    console.log('✅ Contenido recibido (tarea en progreso)');
  }
}

/**
 * Flujo para verificar logs de consola
 */
export async function verifyConsoleLogs(
  page: Page,
  testInstruction: string = 'Test'
): Promise<void> {
  // 1. Configurar captura de logs
  const logs = setupConsoleLogCapture(page);

  // 2. Navegar y crear tarea
  const agentPage = new AgentControlPage(page);
  await agentPage.goto();
  await agentPage.createTask(testInstruction);
  await page.waitForTimeout(5000);

  // 3. Analizar logs
  const problematicLogs = filterProblematicLogs(logs);

  if (problematicLogs.length > 0) {
    console.error('❌ Logs problemáticos encontrados:');
    problematicLogs.forEach((log) => console.error(`   - ${log}`));
    throw new Error('Se detectaron logs que indican problemas con el stream');
  }

  console.log(
    `✅ No se encontraron logs problemáticos (${logs.length} logs totales)`
  );
}

/**
 * Ejecuta el flujo completo de creación y monitoreo de tarea con métricas
 */
export async function executeCompleteFlowWithMetrics(
  page: Page,
  instruction: string,
  testName: string = 'Complete Flow E2E'
): Promise<{
  metrics: TestMetrics;
  result: Awaited<ReturnType<typeof monitorTaskProgress>> | null;
  error?: unknown;
}> {
  const tracker = createMetricsTracker(testName);
  tracker.start();

  try {
    // 1. Navegar a la página
    tracker.startStep('Navegar a página');
    await navigateToAgentControl(page);
    tracker.endStep(true);

    // 2. Crear tarea
    tracker.startStep('Crear tarea');
    await createTask(page, instruction);
    tracker.endStep(true);

    // 3. Esperar a que aparezca la tarea
    tracker.startStep('Esperar tarea');
    await waitForTaskToAppear(page);
    tracker.endStep(true);

    // 4. Monitorear progreso
    tracker.startStep('Monitorear progreso');
    const result = await monitorTaskProgress(
      page,
      (attempt: number, task: TaskInfo | null) => {
        if (task) {
          const preview = task.text?.substring(0, 100) || '';
          console.log(`📊 Segundo ${attempt}: ${preview}`);
        }
      }
    );
    tracker.endStep(!result.error);

    // 5. Recolectar métricas de performance
    const performance = await collectPerformanceMetrics(page);
    const metrics = tracker.finish(!result.error);
    metrics.performance = performance;

    // 6. Verificar resultado
    if (result.error) {
      tracker.addError(result.error);
      throw new Error(result.error);
    }

    return { metrics, result };
  } catch (error) {
    // Capturar screenshot en caso de error
    const screenshotPath = `test-results/error-${Date.now()}.png`;
    await page.screenshot({ path: screenshotPath });
    tracker.addScreenshot(screenshotPath);

    const errorMessage = error instanceof Error ? error.message : String(error);
    tracker.addError(errorMessage);

    const metrics = tracker.finish(false);
    return { metrics, result: null, error };
  }
}

/**
 * Verifica logs de consola con análisis de red
 */
export async function verifyConsoleLogsWithNetworkAnalysis(
  page: Page,
  instruction: string
): Promise<{
  logs: string[];
  networkRequests: Array<{ url: string; method: string; status: number; timestamp: number }>;
  failedRequests: Array<{ url: string; method: string; status: number; timestamp: number }>;
}> {
  // 1. Configurar captura de logs
  const logs = setupConsoleLogCapture(page);

  // 2. Interceptar requests de red para análisis
  const networkRequests = interceptNetworkRequests(page);

  // 3. Navegar y crear tarea
  await navigateToAgentControl(page);
  await createTask(page, instruction);

  // 4. Esperar con condición en lugar de timeout fijo
  await waitForCondition(
    async () => {
      const problematicLogs = filterProblematicLogs(logs);
      return problematicLogs.length === 0;
    },
    {
      timeout: TIMEOUTS.LOG_CHECK_TIMEOUT,
      interval: 1000,
      timeoutMessage: 'Logs problemáticos detectados durante el timeout',
    }
  );

  // 5. Analizar logs
  const problematicLogs = filterProblematicLogs(logs);

  if (problematicLogs.length > 0) {
    console.error('❌ Logs problemáticos encontrados:');
    problematicLogs.forEach((log) => console.error(`   - ${log}`));
    throw new Error('Se detectaron logs que indican problemas con el stream');
  }

  // 6. Analizar requests de red
  const failedRequests = networkRequests.filter((r) => r.status >= 400);
  if (failedRequests.length > 0) {
    console.warn(`⚠️ ${failedRequests.length} requests fallaron`);
  }

  return { logs, networkRequests, failedRequests };
}

/**
 * Crea múltiples tareas en secuencia
 */
export async function createMultipleTasks(
  page: Page,
  instructions: string[],
  delayBetweenTasks: number = 500
): Promise<void> {
  await navigateToAgentControl(page);

  for (const instruction of instructions) {
    await createTask(page, instruction);
    await page.waitForTimeout(delayBetweenTasks);
  }

  await waitForTaskToAppear(page);
}

/**
 * Valida performance de operaciones críticas
 */
export async function validatePerformanceMetrics(
  page: Page,
  operations: Array<{ name: string; action: () => Promise<void> }>
): Promise<{
  metrics: ReturnType<typeof createMetricsTracker> extends { finish: (passed: boolean) => infer R } ? R : never;
  report: ReturnType<typeof generateReport>;
}> {
  const tracker = createMetricsTracker('Performance Validation');
  tracker.start();

  for (const operation of operations) {
    tracker.startStep(operation.name);
    await operation.action();
    tracker.endStep(true);
  }

  const performance = await collectPerformanceMetrics(page);
  const metrics = tracker.finish(true);
  metrics.performance = performance;

  const report = generateReport(metrics);
  return { metrics, report };
}

