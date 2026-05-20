/**
 * Test Helpers Especializados
 * 
 * Helpers adicionales específicos para simplificar tests comunes
 */
import { Page, Locator, expect as playwrightExpect } from '@playwright/test';
import { TIMEOUTS, BASE_URL } from './helpers';
import { measureTime, waitForCondition } from './test-utils';

// ============================================================================
// UI Helpers
// ============================================================================

/**
 * Espera a que un elemento sea visible y clickeable
 */
export async function waitForElementReady(
  page: Page,
  selector: string | Locator,
  options: { timeout?: number } = {}
): Promise<Locator> {
  const locator = typeof selector === 'string' ? page.locator(selector) : selector;
  await locator.waitFor({ 
    state: 'visible',
    timeout: options.timeout || TIMEOUTS.PAGE_LOAD 
  });
  await locator.waitFor({ state: 'attached' });
  return locator;
}

/**
 * Hace clic en un elemento con retry automático
 */
export async function clickWithRetry(
  page: Page,
  selector: string | Locator,
  options: { maxAttempts?: number; delay?: number } = {}
): Promise<void> {
  const { maxAttempts = 3, delay = 500 } = options;
  const locator = typeof selector === 'string' ? page.locator(selector) : selector;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await locator.click({ timeout: 5000 });
      return;
    } catch (error) {
      if (attempt === maxAttempts) throw error;
      await page.waitForTimeout(delay);
    }
  }
}

/**
 * Rellena un campo de formulario con validación
 */
export async function fillFieldWithValidation(
  page: Page,
  selector: string,
  value: string,
  options: { validate?: boolean } = {}
): Promise<void> {
  const field = await waitForElementReady(page, selector);
  await field.fill(value);
  
  if (options.validate !== false) {
    const actualValue = await field.inputValue();
    if (actualValue !== value) {
      throw new Error(
        `Field value mismatch. Expected: "${value}", Got: "${actualValue}"`
      );
    }
  }
}

/**
 * Verifica que un elemento tenga el texto esperado
 */
export async function expectElementText(
  locator: Locator,
  expectedText: string | RegExp,
  options: { timeout?: number } = {}
): Promise<void> {
  await playwrightExpect(locator).toHaveText(expectedText, {
    timeout: options.timeout || 5000,
  });
}

/**
 * Verifica que un elemento sea visible
 */
export async function expectElementVisible(
  locator: Locator,
  options: { timeout?: number } = {}
): Promise<void> {
  await playwrightExpect(locator).toBeVisible({
    timeout: options.timeout || 5000,
  });
}

// ============================================================================
// Navigation Helpers
// ============================================================================

/**
 * Navega y espera a que la página esté completamente cargada
 */
export async function navigateAndWait(
  page: Page,
  url: string,
  options: { waitUntil?: 'load' | 'domcontentloaded' | 'networkidle' } = {}
): Promise<void> {
  await page.goto(url, {
    waitUntil: options.waitUntil || 'networkidle',
    timeout: TIMEOUTS.PAGE_LOAD,
  });
  await page.waitForLoadState('networkidle');
}

/**
 * Verifica que la URL actual coincida con la esperada
 */
export async function expectCurrentUrl(
  page: Page,
  expectedUrl: string | RegExp
): Promise<void> {
  const currentUrl = page.url();
  if (expectedUrl instanceof RegExp) {
    playwrightExpect(currentUrl).toMatch(expectedUrl);
  } else {
    playwrightExpect(currentUrl).toBe(expectedUrl);
  }
}

// ============================================================================
// Form Helpers
// ============================================================================

/**
 * Rellena un formulario completo
 */
export async function fillForm(
  page: Page,
  fields: Record<string, string>
): Promise<void> {
  for (const [selector, value] of Object.entries(fields)) {
    await fillFieldWithValidation(page, selector, value);
  }
}

/**
 * Envía un formulario y espera la respuesta
 */
export async function submitForm(
  page: Page,
  submitSelector: string,
  options: { waitForNavigation?: boolean } = {}
): Promise<void> {
  const submitButton = await waitForElementReady(page, submitSelector);
  
  if (options.waitForNavigation) {
    await Promise.all([
      page.waitForNavigation({ timeout: TIMEOUTS.PAGE_LOAD }),
      submitButton.click(),
    ]);
  } else {
    await submitButton.click();
  }
}

// ============================================================================
// Assertion Helpers
// ============================================================================

/**
 * Verifica que una respuesta de API sea exitosa
 */
export async function expectApiSuccess(
  response: { ok: () => boolean; status: () => number }
): Promise<void> {
  playwrightExpect(response.ok()).toBeTruthy();
  playwrightExpect(response.status()).toBeLessThan(400);
}

/**
 * Verifica que una respuesta de API tenga el formato esperado
 */
export async function expectApiResponseFormat(
  response: any,
  schema: {
    requiredFields?: string[];
    type?: 'object' | 'array';
  } = {}
): Promise<void> {
  const { requiredFields = ['id'], type = 'object' } = schema;

  if (type === 'array') {
    playwrightExpect(Array.isArray(response)).toBeTruthy();
    if (response.length > 0) {
      for (const field of requiredFields) {
        playwrightExpect(response[0]).toHaveProperty(field);
      }
    }
  } else {
    playwrightExpect(typeof response).toBe('object');
    for (const field of requiredFields) {
      playwrightExpect(response).toHaveProperty(field);
    }
  }
}

// ============================================================================
// Screenshot Helpers
// ============================================================================

/**
 * Toma un screenshot con nombre descriptivo y timestamp
 */
export async function takeNamedScreenshot(
  page: Page,
  name: string,
  options: { fullPage?: boolean } = {}
): Promise<string> {
  const timestamp = Date.now();
  const filename = `test-results/${name}-${timestamp}.png`;
  await page.screenshot({
    path: filename,
    fullPage: options.fullPage || false,
  });
  return filename;
}

/**
 * Toma un screenshot de un elemento específico
 */
export async function takeElementScreenshot(
  locator: Locator,
  name: string
): Promise<string> {
  const timestamp = Date.now();
  const filename = `test-results/${name}-${timestamp}.png`;
  await locator.screenshot({ path: filename });
  return filename;
}

// ============================================================================
// Performance Helpers
// ============================================================================

/**
 * Mide el tiempo de carga de una página
 */
export async function measurePageLoadTime(
  page: Page,
  url: string
): Promise<number> {
  const { duration } = await measureTime(async () => {
    await navigateAndWait(page, url);
  });
  return duration;
}

/**
 * Verifica que una operación complete dentro del tiempo esperado
 */
export async function expectTiming(
  operation: () => Promise<void>,
  maxDuration: number
): Promise<void> {
  const { duration } = await measureTime(operation);
  playwrightExpect(duration).toBeLessThan(maxDuration);
}

// ============================================================================
// Network Helpers
// ============================================================================

/**
 * Espera a que una request específica se complete
 */
export async function waitForRequest(
  page: Page,
  urlPattern: string | RegExp,
  options: { timeout?: number } = {}
): Promise<void> {
  const pattern = typeof urlPattern === 'string' 
    ? new RegExp(urlPattern) 
    : urlPattern;

  await page.waitForResponse(
    (response) => pattern.test(response.url()),
    { timeout: options.timeout || TIMEOUTS.API_REQUEST }
  );
}

/**
 * Intercepta y valida una request específica
 */
export async function interceptAndValidateRequest(
  page: Page,
  urlPattern: string | RegExp,
  validator: (response: any) => boolean | Promise<boolean>
): Promise<boolean> {
  return new Promise((resolve, reject) => {
    const pattern = typeof urlPattern === 'string' 
      ? new RegExp(urlPattern) 
      : urlPattern;

    const timeout = setTimeout(() => {
      reject(new Error('Request timeout'));
    }, TIMEOUTS.API_REQUEST);

    page.on('response', async (response) => {
      if (pattern.test(response.url())) {
        clearTimeout(timeout);
        try {
          const data = await response.json();
          const isValid = await validator(data);
          resolve(isValid);
        } catch (error) {
          reject(error);
        }
      }
    });
  });
}

// ============================================================================
// Utility Helpers
// ============================================================================

/**
 * Espera un tiempo específico (wrapper mejorado)
 */
export async function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Genera un ID único para tests
 */
export function generateTestId(prefix: string = 'test'): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).substring(7)}`;
}

/**
 * Limpia texto para comparaciones
 */
export function normalizeText(text: string): string {
  return text
    .trim()
    .replace(/\s+/g, ' ')
    .toLowerCase();
}

/**
 * Compara dos textos ignorando espacios y mayúsculas
 */
export function textMatches(text1: string, text2: string): boolean {
  return normalizeText(text1) === normalizeText(text2);
}



