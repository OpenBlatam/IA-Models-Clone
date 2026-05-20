/**
 * Estrategias de Retry Avanzadas para Tests E2E
 * 
 * Proporciona diferentes estrategias de retry para operaciones inestables
 */
import { Page, Locator } from '@playwright/test';

// ============================================================================
// Types
// ============================================================================

export interface RetryOptions {
  maxAttempts?: number;
  delay?: number;
  backoff?: 'linear' | 'exponential' | 'fixed';
  backoffMultiplier?: number;
  onRetry?: (attempt: number, error: Error) => void;
  shouldRetry?: (error: Error) => boolean;
}

// ============================================================================
// Retry Strategies
// ============================================================================

/**
 * Ejecuta una función con retry usando estrategia configurable
 */
export async function retryWithStrategy<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    delay = 1000,
    backoff = 'exponential',
    backoffMultiplier = 2,
    onRetry,
    shouldRetry = () => true,
  } = options;

  let lastError: Error | null = null;
  let currentDelay = delay;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Verificar si se debe reintentar
      if (!shouldRetry(lastError) || attempt >= maxAttempts) {
        throw lastError;
      }

      // Callback de retry
      if (onRetry) {
        onRetry(attempt, lastError);
      }

      // Calcular delay según estrategia
      switch (backoff) {
        case 'exponential':
          currentDelay = delay * Math.pow(backoffMultiplier, attempt - 1);
          break;
        case 'linear':
          currentDelay = delay * attempt;
          break;
        case 'fixed':
        default:
          currentDelay = delay;
      }

      await new Promise((resolve) => setTimeout(resolve, currentDelay));
    }
  }

  throw lastError || new Error('Retry failed');
}

/**
 * Hace click con retry inteligente
 */
export async function clickWithRetryStrategy(
  locator: Locator,
  options: RetryOptions = {}
): Promise<void> {
  await retryWithStrategy(
    async () => {
      await locator.click({ timeout: 5000 });
    },
    {
      maxAttempts: 3,
      delay: 1000,
      ...options,
      shouldRetry: (error) => {
        const errorMessage = error.message.toLowerCase();
        return (
          errorMessage.includes('timeout') ||
          errorMessage.includes('detached') ||
          errorMessage.includes('not visible')
        );
      },
    }
  );
}

/**
 * Espera a que un elemento sea visible con retry
 */
export async function waitForVisibleWithRetry(
  locator: Locator,
  options: RetryOptions = {}
): Promise<void> {
  await retryWithStrategy(
    async () => {
      await locator.waitFor({ state: 'visible', timeout: 5000 });
    },
    {
      maxAttempts: 5,
      delay: 500,
      ...options,
    }
  );
}

/**
 * Llena un campo con retry y validación
 */
export async function fillWithRetryStrategy(
  locator: Locator,
  value: string,
  options: RetryOptions = {}
): Promise<void> {
  await retryWithStrategy(
    async () => {
      await locator.fill(value);
      // Validar que el valor se guardó
      const actualValue = await locator.inputValue();
      if (actualValue !== value) {
        throw new Error(`Value mismatch: expected "${value}", got "${actualValue}"`);
      }
    },
    {
      maxAttempts: 3,
      delay: 500,
      ...options,
    }
  );
}

/**
 * Navega a una URL con retry
 */
export async function navigateWithRetry(
  page: Page,
  url: string,
  options: RetryOptions = {}
): Promise<void> {
  await retryWithStrategy(
    async () => {
      const response = await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
      if (!response || !response.ok()) {
        throw new Error(`Navigation failed: ${response?.status() || 'no response'}`);
      }
    },
    {
      maxAttempts: 3,
      delay: 2000,
      backoff: 'exponential',
      ...options,
    }
  );
}



