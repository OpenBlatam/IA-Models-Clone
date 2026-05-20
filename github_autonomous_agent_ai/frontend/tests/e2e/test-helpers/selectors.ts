/**
 * Selectores y Patrones Comunes para E2E Tests
 * 
 * Centraliza todos los selectores CSS y patrones de búsqueda
 * para facilitar mantenimiento y consistencia
 */
import { Page, Locator } from '@playwright/test';

// ============================================================================
// Selector Constants
// ============================================================================

export const SELECTORS = {
  // Task-related
  taskCard: '[data-testid="task-card"], .task-card, [class*="task"]',
  taskInput: 'textarea[name="instruction"]',
  createButton: 'button:has-text("crear"), button:has-text("procesar"), button[type="submit"]',
  
  // Error-related
  error: '[role="alert"], .error, [class*="error"], [class*="Error"], [data-error]',
  validation: '[role="alert"], .error, [class*="invalid"], [class*="validation"]',
  
  // Loading-related
  loading: '[class*="loading"], [class*="spinner"], [aria-busy="true"]',
  loadingText: 'text=/cargando/i, text=/loading/i',
  
  // Rate limiting
  rateLimit: 'text=/rate limit/i, text=/too many requests/i, text=/demasiadas solicitudes/i, [data-rate-limit]',
  
  // Retry/recovery
  retryButton: 'button:has-text("reintentar"), button:has-text("retry"), button:has-text("Reintentar")',
  
  // Main content
  mainContent: 'main, [role="main"], .main, #main',
  
  // Form elements
  form: 'form, [role="form"]',
  buttons: 'button',
  inputs: 'input, textarea',
  links: 'a[href]',
  
  // Headings
  h1: 'h1',
} as const;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Busca un elemento usando múltiples selectores alternativos
 */
export async function findElement(
  page: Page,
  selectors: string | string[]
): Promise<Locator | null> {
  const selectorArray = Array.isArray(selectors) ? selectors : [selectors];
  
  for (const selector of selectorArray) {
    const element = page.locator(selector).first();
    const count = await element.count();
    if (count > 0) {
      return element;
    }
  }
  
  return null;
}

/**
 * Verifica si un elemento existe usando múltiples selectores
 */
export async function elementExists(
  page: Page,
  selectors: string | string[]
): Promise<boolean> {
  const element = await findElement(page, selectors);
  return element !== null;
}

/**
 * Obtiene el texto de un elemento si existe
 */
export async function getElementText(
  page: Page,
  selectors: string | string[]
): Promise<string | null> {
  const element = await findElement(page, selectors);
  if (!element) return null;
  
  return element.textContent().catch(() => null);
}

/**
 * Cuenta elementos que coinciden con múltiples selectores
 */
export async function countElements(
  page: Page,
  selectors: string | string[]
): Promise<number> {
  const selectorArray = Array.isArray(selectors) ? selectors : [selectors];
  let total = 0;
  
  for (const selector of selectorArray) {
    const count = await page.locator(selector).count();
    total += count;
  }
  
  return total;
}


