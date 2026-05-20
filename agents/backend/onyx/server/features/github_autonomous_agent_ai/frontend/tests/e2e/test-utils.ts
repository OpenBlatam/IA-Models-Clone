/**
 * Test Utilities
 * 
 * Utilidades adicionales para tests E2E que no están en helpers.ts
 */
import { Page, APIRequestContext } from '@playwright/test';
import { BASE_URL, TIMEOUTS } from './helpers';

// ============================================================================
// Assertion Helpers
// ============================================================================

/**
 * Verifica que una respuesta de API tenga el formato esperado
 */
export function assertValidApiResponse(response: any): asserts response is {
  id: string;
  status: string;
  [key: string]: any;
} {
  if (!response || typeof response !== 'object') {
    throw new Error('API response is not an object');
  }
  if (!response.id || typeof response.id !== 'string') {
    throw new Error('API response missing valid id');
  }
  if (!response.status || typeof response.status !== 'string') {
    throw new Error('API response missing valid status');
  }
}

/**
 * Verifica que un stream tenga contenido válido
 */
export function assertValidStreamContent(content: string): void {
  if (!content || content.trim().length === 0) {
    throw new Error('Stream content is empty');
  }
  if (content === '{}') {
    throw new Error('Stream content is empty object');
  }
  if (content.length < 10) {
    throw new Error(`Stream content too short: ${content.length} characters`);
  }
}

// ============================================================================
// Retry Helpers
// ============================================================================

/**
 * Ejecuta una función con retry automático
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: {
    maxAttempts?: number;
    delay?: number;
    onRetry?: (attempt: number, error: Error) => void;
  } = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    delay = 1000,
    onRetry,
  } = options;

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      if (attempt < maxAttempts) {
        if (onRetry) {
          onRetry(attempt, lastError);
        }
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError || new Error('Retry failed');
}

/**
 * Espera hasta que una condición sea verdadera
 */
export async function waitForCondition(
  condition: () => Promise<boolean> | boolean,
  options: {
    timeout?: number;
    interval?: number;
    timeoutMessage?: string;
  } = {}
): Promise<void> {
  const {
    timeout = 30000,
    interval = 500,
    timeoutMessage = 'Condition not met within timeout',
  } = options;

  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }

  throw new Error(timeoutMessage);
}

// ============================================================================
// Performance Helpers
// ============================================================================

/**
 * Mide el tiempo de ejecución de una función
 */
export async function measureTime<T>(
  fn: () => Promise<T>
): Promise<{ result: T; duration: number }> {
  const startTime = Date.now();
  const result = await fn();
  const duration = Date.now() - startTime;
  return { result, duration };
}

/**
 * Verifica que una operación complete dentro de un tiempo límite
 */
export async function assertTiming<T>(
  fn: () => Promise<T>,
  maxDuration: number
): Promise<T> {
  const { result, duration } = await measureTime(fn);
  
  if (duration > maxDuration) {
    throw new Error(
      `Operation took ${duration}ms, expected less than ${maxDuration}ms`
    );
  }

  return result;
}

// ============================================================================
// Network Helpers
// ============================================================================

/**
 * Espera a que todas las requests de red se completen
 */
export async function waitForNetworkIdle(
  page: Page,
  timeout: number = TIMEOUTS.PAGE_LOAD
): Promise<void> {
  await page.waitForLoadState('networkidle', { timeout });
}

/**
 * Intercepta y registra todas las requests de red
 */
export function interceptNetworkRequests(page: Page): Array<{
  url: string;
  method: string;
  status: number;
  timestamp: number;
}> {
  const requests: Array<{
    url: string;
    method: string;
    status: number;
    timestamp: number;
  }> = [];

  page.on('request', (request) => {
    requests.push({
      url: request.url(),
      method: request.method(),
      status: 0, // Se actualizará en la respuesta
      timestamp: Date.now(),
    });
  });

  page.on('response', (response) => {
    const request = requests.find(
      (r) => r.url === response.url() && r.status === 0
    );
    if (request) {
      request.status = response.status();
    }
  });

  return requests;
}

// ============================================================================
// Data Helpers
// ============================================================================

/**
 * Genera datos de prueba aleatorios
 */
export function generateTestData() {
  return {
    instruction: `Test instruction ${Date.now()}`,
    repository: `test/repo-${Math.random().toString(36).substring(7)}`,
    timestamp: Date.now(),
  };
}

/**
 * Limpia datos de prueba (placeholder para implementación futura)
 */
export async function cleanupTestData(taskIds: string[]): Promise<void> {
  // Implementar lógica de limpieza si es necesario
  console.log(`Would cleanup ${taskIds.length} test tasks`);
}

// ============================================================================
// Reporting Helpers
// ============================================================================

/**
 * Formatea un resultado de test para reporting
 */
export function formatTestResult(result: {
  passed: boolean;
  duration: number;
  error?: string;
  metadata?: Record<string, any>;
}): string {
  const status = result.passed ? '✅ PASSED' : '❌ FAILED';
  const duration = `${result.duration}ms`;
  const error = result.error ? `\nError: ${result.error}` : '';
  const metadata = result.metadata
    ? `\nMetadata: ${JSON.stringify(result.metadata, null, 2)}`
    : '';

  return `${status} (${duration})${error}${metadata}`;
}

/**
 * Crea un reporte de test con información detallada
 */
export function createTestReport(testName: string, results: {
  passed: boolean;
  duration: number;
  steps: Array<{ name: string; duration: number; passed: boolean }>;
  screenshots?: string[];
  errors?: string[];
}): string {
  const report = [
    `\n${'='.repeat(60)}`,
    `Test: ${testName}`,
    `${'='.repeat(60)}`,
    `Status: ${results.passed ? '✅ PASSED' : '❌ FAILED'}`,
    `Duration: ${results.duration}ms`,
    `\nSteps:`,
    ...results.steps.map(
      (step) =>
        `  ${step.passed ? '✅' : '❌'} ${step.name} (${step.duration}ms)`
    ),
  ];

  if (results.screenshots && results.screenshots.length > 0) {
    report.push(`\nScreenshots:`, ...results.screenshots.map((s) => `  - ${s}`));
  }

  if (results.errors && results.errors.length > 0) {
    report.push(`\nErrors:`, ...results.errors.map((e) => `  - ${e}`));
  }

  report.push(`${'='.repeat(60)}\n`);

  return report.join('\n');
}



