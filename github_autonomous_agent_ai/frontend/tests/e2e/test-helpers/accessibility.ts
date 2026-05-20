/**
 * Utilidades para Testing de Accesibilidad
 * 
 * Proporciona funciones para validar accesibilidad en tests E2E
 */
import { Page, Locator } from '@playwright/test';
import * as axe from 'axe-core';

// ============================================================================
// Types
// ============================================================================

export interface AccessibilityViolation {
  id: string;
  impact: 'minor' | 'moderate' | 'serious' | 'critical';
  description: string;
  help: string;
  helpUrl: string;
  nodes: Array<{
    html: string;
    target: string[];
    failureSummary: string;
  }>;
}

export interface AccessibilityResult {
  passed: boolean;
  violations: AccessibilityViolation[];
  incomplete: any[];
  inapplicable: any[];
  timestamp: string;
}

// ============================================================================
// Accessibility Testing
// ============================================================================

/**
 * Ejecuta análisis de accesibilidad usando axe-core
 */
export async function runAccessibilityCheck(
  page: Page,
  options: {
    include?: string[];
    exclude?: string[];
    rules?: Record<string, { enabled: boolean }>;
  } = {}
): Promise<AccessibilityResult> {
  // Inyectar axe-core si no está disponible
  await page.addInitScript(() => {
    // Cargar axe-core desde CDN si no está disponible
    if (typeof (window as any).axe === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://unpkg.com/axe-core@4.8.0/axe.min.js';
      document.head.appendChild(script);
    }
  });

  // Esperar a que axe esté disponible
  await page.waitForFunction(() => typeof (window as any).axe !== 'undefined');

  const result = await page.evaluate(
    async (opts) => {
      const axe = (window as any).axe;
      return await axe.run(document, opts);
    },
    options
  );

  return {
    passed: result.violations.length === 0,
    violations: result.violations.map((v: any) => ({
      id: v.id,
      impact: v.impact,
      description: v.description,
      help: v.help,
      helpUrl: v.helpUrl,
      nodes: v.nodes.map((n: any) => ({
        html: n.html,
        target: n.target,
        failureSummary: n.failureSummary,
      })),
    })),
    incomplete: result.incomplete || [],
    inapplicable: result.inapplicable || [],
    timestamp: new Date().toISOString(),
  };
}

/**
 * Valida que todos los elementos interactivos sean accesibles por teclado
 */
export async function validateKeyboardAccessibility(
  page: Page
): Promise<{
  passed: boolean;
  issues: Array<{ element: string; issue: string }>;
}> {
  const issues: Array<{ element: string; issue: string }> = [];

  // Verificar que todos los botones sean accesibles por teclado
  const buttons = await page.locator('button, [role="button"]').all();
  for (const button of buttons) {
    const tabIndex = await button.getAttribute('tabindex');
    const disabled = await button.getAttribute('disabled');
    
    if (disabled === null && (tabIndex === '-1' || tabIndex === null)) {
      const text = await button.textContent();
      issues.push({
        element: text || 'Button',
        issue: 'Button may not be keyboard accessible',
      });
    }
  }

  // Verificar que todos los links sean accesibles por teclado
  const links = await page.locator('a[href]').all();
  for (const link of links) {
    const tabIndex = await link.getAttribute('tabindex');
    if (tabIndex === '-1') {
      const text = await link.textContent();
      issues.push({
        element: text || 'Link',
        issue: 'Link has tabindex="-1" and may not be keyboard accessible',
      });
    }
  }

  // Verificar que todos los inputs tengan labels
  const inputs = await page.locator('input, textarea, select').all();
  for (const input of inputs) {
    const id = await input.getAttribute('id');
    const ariaLabel = await input.getAttribute('aria-label');
    const ariaLabelledBy = await input.getAttribute('aria-labelledby');
    const placeholder = await input.getAttribute('placeholder');
    const type = await input.getAttribute('type');

    // Skip hidden inputs
    if (type === 'hidden') continue;

    if (!id && !ariaLabel && !ariaLabelledBy && !placeholder) {
      issues.push({
        element: await input.getAttribute('name') || 'Input',
        issue: 'Input missing label or aria-label',
      });
    }
  }

  return {
    passed: issues.length === 0,
    issues,
  };
}

/**
 * Valida contraste de colores (básico)
 */
export async function validateColorContrast(
  page: Page
): Promise<{
  passed: boolean;
  issues: Array<{ element: string; issue: string }>;
}> {
  const issues: Array<{ element: string; issue: string }> = [];

  // Obtener todos los elementos con texto
  const textElements = await page.locator('p, span, div, h1, h2, h3, h4, h5, h6, a, button, label').all();

  for (const element of textElements.slice(0, 50)) { // Limitar a 50 para performance
    const text = await element.textContent();
    if (!text || text.trim().length === 0) continue;

    const styles = await element.evaluate((el) => {
      const computed = window.getComputedStyle(el);
      return {
        color: computed.color,
        backgroundColor: computed.backgroundColor,
        fontSize: computed.fontSize,
      };
    });

    // Validación básica - verificar que hay contraste
    if (styles.color === styles.backgroundColor) {
      const tagName = await element.evaluate((el) => el.tagName);
      issues.push({
        element: `${tagName}: ${text.substring(0, 30)}`,
        issue: 'Text and background colors are the same',
      });
    }
  }

  return {
    passed: issues.length === 0,
    issues,
  };
}

/**
 * Valida que todos los elementos tengan roles ARIA apropiados
 */
export async function validateARIA(
  page: Page
): Promise<{
  passed: boolean;
  issues: Array<{ element: string; issue: string }>;
}> {
  const issues: Array<{ element: string; issue: string }> = [];

  // Verificar elementos interactivos sin roles
  const interactiveElements = await page
    .locator('button, a, input, select, textarea, [onclick], [role]')
    .all();

  for (const element of interactiveElements) {
    const role = await element.getAttribute('role');
    const tagName = await element.evaluate((el) => el.tagName.toLowerCase());
    const hasOnClick = await element.getAttribute('onclick');

    // Elementos div/span con onclick deberían tener role="button"
    if ((tagName === 'div' || tagName === 'span') && hasOnClick && !role) {
      const text = await element.textContent();
      issues.push({
        element: text || tagName,
        issue: 'Interactive element without ARIA role',
      });
    }
  }

  // Verificar que elementos con aria-label tengan contenido
  const ariaLabeled = await page.locator('[aria-label]').all();
  for (const element of ariaLabeled) {
    const ariaLabel = await element.getAttribute('aria-label');
    if (!ariaLabel || ariaLabel.trim().length === 0) {
      const tagName = await element.evaluate((el) => el.tagName);
      issues.push({
        element: tagName,
        issue: 'Element has empty aria-label',
      });
    }
  }

  return {
    passed: issues.length === 0,
    issues,
  };
}

/**
 * Valida navegación por teclado
 */
export async function validateKeyboardNavigation(
  page: Page
): Promise<{
  passed: boolean;
  issues: Array<{ element: string; issue: string }>;
}> {
  const issues: Array<{ element: string; issue: string }> = [];

  // Verificar orden de tab
  const focusableElements = await page.evaluate(() => {
    const selectors = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
    ].join(', ');

    return Array.from(document.querySelectorAll(selectors)).map((el) => ({
      tagName: el.tagName,
      text: el.textContent?.substring(0, 30) || '',
      tabIndex: (el as HTMLElement).tabIndex,
    }));
  });

  // Verificar que no haya tabindex > 0 (mal práctica)
  for (const element of focusableElements) {
    if (element.tabIndex > 0) {
      issues.push({
        element: `${element.tagName}: ${element.text}`,
        issue: 'Element has tabindex > 0, which disrupts natural tab order',
      });
    }
  }

  return {
    passed: issues.length === 0,
    issues,
  };
}

/**
 * Ejecuta validación completa de accesibilidad
 */
export async function runFullAccessibilityCheck(
  page: Page
): Promise<{
  axe: AccessibilityResult;
  keyboard: Awaited<ReturnType<typeof validateKeyboardAccessibility>>;
  contrast: Awaited<ReturnType<typeof validateColorContrast>>;
  aria: Awaited<ReturnType<typeof validateARIA>>;
  navigation: Awaited<ReturnType<typeof validateKeyboardNavigation>>;
  passed: boolean;
}> {
  const [axeResult, keyboard, contrast, aria, navigation] = await Promise.all([
    runAccessibilityCheck(page),
    validateKeyboardAccessibility(page),
    validateColorContrast(page),
    validateARIA(page),
    validateKeyboardNavigation(page),
  ]);

  const passed =
    axeResult.passed &&
    keyboard.passed &&
    contrast.passed &&
    aria.passed &&
    navigation.passed;

  return {
    axe: axeResult,
    keyboard,
    contrast,
    aria,
    navigation,
    passed,
  };
}


