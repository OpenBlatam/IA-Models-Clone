/**
 * Helpers para Escenarios de Error
 * 
 * Proporciona utilidades para probar diferentes tipos de errores
 * y validar el manejo robusto de la aplicación
 */
import { Page } from '@playwright/test';
import { navigateToAgentControl, createTask } from '../helpers';
import { TEST_INSTRUCTIONS } from '../constants';
import { SELECTORS, findElement, getElementText } from './selectors';

// ============================================================================
// Error Scenario Helpers
// ============================================================================

/**
 * Simula diferentes tipos de errores de red
 */
export async function simulateNetworkErrors(
  page: Page,
  errorType: 'timeout' | 'failed' | 'aborted' | 'blocked' = 'failed'
): Promise<void> {
  await page.route('**/api/**', (route) => {
    switch (errorType) {
      case 'timeout':
        // Simular timeout (no hacer nada, dejar que expire)
        break;
      case 'failed':
        route.abort('failed');
        break;
      case 'aborted':
        route.abort('aborted');
        break;
      case 'blocked':
        route.abort('blocked');
        break;
    }
  });
}

/**
 * Valida que la aplicación maneja errores de red gracefully
 */
export async function testNetworkErrorHandling(
  page: Page,
  errorType: 'timeout' | 'failed' | 'aborted' | 'blocked' = 'failed'
): Promise<{
  errorDisplayed: boolean;
  errorMessage?: string;
  recoveryPossible: boolean;
}> {
  await simulateNetworkErrors(page, errorType);
  await navigateToAgentControl(page);
  await createTask(page, TEST_INSTRUCTIONS.SIMPLE);

  // Buscar mensajes de error
  const errorElement = await findElement(page, SELECTORS.error);
  const errorDisplayed = errorElement !== null;
  const errorMessage = errorDisplayed ? await getElementText(page, SELECTORS.error) : undefined;

  // Verificar si hay opción de reintentar
  const retryButton = await findElement(page, SELECTORS.retryButton);
  const recoveryPossible = retryButton !== null;

  return {
    errorDisplayed,
    errorMessage,
    recoveryPossible,
  };
}

/**
 * Valida comportamiento con datos inválidos
 */
export async function testInvalidDataHandling(
  page: Page
): Promise<{
  validationShown: boolean;
  submissionBlocked: boolean;
}> {
  await navigateToAgentControl(page);

  // Intentar enviar con diferentes tipos de datos inválidos
  const invalidInputs = [
    '', // Vacío
    ' '.repeat(100), // Solo espacios
    '\n\n\n', // Solo saltos de línea
  ];

  let validationShown = false;
  let submissionBlocked = false;

  for (const invalidInput of invalidInputs) {
    const textarea = page.locator(SELECTORS.taskInput);
    await textarea.fill(invalidInput);

    const createButton = page.getByRole('button', { name: /crear|procesar/i });
    const isDisabled = await createButton.isDisabled().catch(() => false);
    
    const validationElement = await findElement(page, SELECTORS.validation);
    const hasValidation = validationElement !== null;

    if (isDisabled || hasValidation) {
      validationShown = true;
      submissionBlocked = true;
      break;
    }
  }

  return {
    validationShown,
    submissionBlocked,
  };
}

/**
 * Valida comportamiento con rate limiting
 */
export async function testRateLimiting(
  page: Page,
  rapidRequests: number = 10
): Promise<{
  rateLimited: boolean;
  rateLimitMessage?: string;
}> {
  await navigateToAgentControl(page);

  // Hacer múltiples requests rápidos
  for (let i = 0; i < rapidRequests; i++) {
    await createTask(page, `Test rate limit ${i}`);
    await page.waitForTimeout(100); // Muy rápido
  }

  // Buscar mensaje de rate limit
  const rateLimitElement = await findElement(page, SELECTORS.rateLimit);
  const rateLimited = rateLimitElement !== null;
  const rateLimitMessage = rateLimited ? await getElementText(page, SELECTORS.rateLimit) : undefined;

  return {
    rateLimited,
    rateLimitMessage,
  };
}

/**
 * Valida comportamiento con servidor lento
 */
export async function testSlowServerResponse(
  page: Page,
  delay: number = 5000
): Promise<{
  loadingIndicatorShown: boolean;
  timeoutHandled: boolean;
}> {
  // Interceptar requests y agregar delay
  await page.route('**/api/**', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, delay));
    await route.continue();
  });

  await navigateToAgentControl(page);
  await createTask(page, TEST_INSTRUCTIONS.SIMPLE);

  // Buscar indicadores de carga
  const loadingElement = await findElement(page, [SELECTORS.loading, SELECTORS.loadingText]);
  const loadingIndicatorShown = loadingElement !== null;

  // Verificar si hay manejo de timeout
  const timeoutElement = await findElement(page, 'text=/timeout/i, text=/tiempo agotado/i');
  const timeoutHandled = timeoutElement !== null;

  return {
    loadingIndicatorShown,
    timeoutHandled,
  };
}


