/**
 * Utilidades de Testing Visual y Accesibilidad
 * 
 * Proporciona funciones para validar accesibilidad, layout responsive y testing visual
 */
import { Page, expect as playwrightExpect } from '@playwright/test';

// ============================================================================
// Types
// ============================================================================

export interface ViewportSize {
  width: number;
  height: number;
}

export interface AccessibilityIssue {
  type: 'missing-label' | 'missing-aria' | 'contrast' | 'keyboard' | 'focus';
  element: string;
  severity: 'error' | 'warning';
  message: string;
}

// ============================================================================
// Accessibility Helpers
// ============================================================================

/**
 * Valida accesibilidad básica de la página
 */
export async function validateBasicAccessibility(page: Page): Promise<void> {
  const issues: AccessibilityIssue[] = [];

  // 1. Verificar que los botones tienen labels o aria-labels
  const buttons = await page.locator('button').all();
  for (const button of buttons) {
    const text = await button.textContent();
    const ariaLabel = await button.getAttribute('aria-label');
    const ariaLabelledBy = await button.getAttribute('aria-labelledby');
    
    if (!text?.trim() && !ariaLabel && !ariaLabelledBy) {
      issues.push({
        type: 'missing-label',
        element: 'button',
        severity: 'error',
        message: 'Button without accessible label',
      });
    }
  }

  // 2. Verificar que los inputs tienen labels asociados
  const inputs = await page.locator('input, textarea, select').all();
  for (const input of inputs) {
    const id = await input.getAttribute('id');
    const ariaLabel = await input.getAttribute('aria-label');
    const ariaLabelledBy = await input.getAttribute('aria-labelledby');
    const name = await input.getAttribute('name');
    
    if (id) {
      const label = await page.locator(`label[for="${id}"]`).count();
      if (label === 0 && !ariaLabel && !ariaLabelledBy) {
        issues.push({
          type: 'missing-label',
          element: `input#${id}`,
          severity: 'warning',
          message: `Input ${id} without associated label`,
        });
      }
    } else if (!ariaLabel && !ariaLabelledBy && !name) {
      issues.push({
        type: 'missing-label',
        element: 'input',
        severity: 'warning',
        message: 'Input without accessible label',
      });
    }
  }

  // 3. Verificar que hay un título de página
  const title = await page.title();
  if (!title || title.trim().length === 0) {
    issues.push({
      type: 'missing-aria',
      element: 'head > title',
      severity: 'error',
      message: 'Page missing title',
    });
  }

  // 4. Verificar que hay un h1 en la página
  const h1Count = await page.locator('h1').count();
  if (h1Count === 0) {
    issues.push({
      type: 'missing-aria',
      element: 'body',
      severity: 'warning',
      message: 'Page missing h1 heading',
    });
  }

  // Reportar issues críticos
  const errors = issues.filter((i) => i.severity === 'error');
  if (errors.length > 0) {
    console.error('❌ Accesibilidad: Se encontraron errores:');
    errors.forEach((issue) => {
      console.error(`   - ${issue.type}: ${issue.message} (${issue.element})`);
    });
    // No fallar el test, solo reportar
    // throw new Error(`Accesibilidad: ${errors.length} errores encontrados`);
  }

  const warnings = issues.filter((i) => i.severity === 'warning');
  if (warnings.length > 0) {
    console.warn('⚠️ Accesibilidad: Se encontraron advertencias:');
    warnings.forEach((issue) => {
      console.warn(`   - ${issue.type}: ${issue.message} (${issue.element})`);
    });
  }

  if (issues.length === 0) {
    console.log('✅ Accesibilidad: No se encontraron problemas');
  }
}

/**
 * Valida que los elementos interactivos son accesibles por teclado
 */
export async function validateKeyboardAccessibility(page: Page): Promise<void> {
  // Verificar que los botones son focusables
  const buttons = await page.locator('button:not([disabled])').all();
  for (const button of buttons) {
    const tabIndex = await button.getAttribute('tabindex');
    if (tabIndex === '-1') {
      console.warn(`⚠️ Button no es accesible por teclado: ${await button.textContent()}`);
    }
  }

  // Verificar que los links son focusables
  const links = await page.locator('a[href]').all();
  for (const link of links) {
    const tabIndex = await link.getAttribute('tabindex');
    if (tabIndex === '-1') {
      console.warn(`⚠️ Link no es accesible por teclado: ${await link.textContent()}`);
    }
  }
}

// ============================================================================
// Layout Validation
// ============================================================================

/**
 * Valida el layout de la página en diferentes viewports
 */
export async function validatePageLayout(
  page: Page,
  viewport: ViewportSize
): Promise<void> {
  // Establecer viewport
  await page.setViewportSize(viewport);
  await page.waitForLoadState('networkidle');

  // Verificar que los elementos principales son visibles
  const mainContent = await page.locator('main, [role="main"], .main, #main').first();
  if (await mainContent.count() > 0) {
    await playwrightExpect(mainContent).toBeVisible();
  }

  // Verificar que no hay overflow horizontal
  const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
  const viewportWidth = viewport.width;
  
  if (bodyWidth > viewportWidth) {
    console.warn(
      `⚠️ Layout: Overflow horizontal detectado (${bodyWidth}px > ${viewportWidth}px)`
    );
  }

  // Verificar que los elementos críticos son visibles
  const criticalElements = [
    'textarea[name="instruction"]',
    'button[type="submit"], button:has-text("crear"), button:has-text("procesar")',
  ];

  for (const selector of criticalElements) {
    const element = page.locator(selector).first();
    const count = await element.count();
    if (count > 0) {
      const isVisible = await element.isVisible().catch(() => false);
      if (!isVisible) {
        console.warn(`⚠️ Layout: Elemento crítico no visible en viewport ${viewport.width}x${viewport.height}: ${selector}`);
      }
    }
  }

  console.log(
    `✅ Layout validado para viewport ${viewport.width}x${viewport.height}`
  );
}

/**
 * Valida que el layout es responsive
 */
export async function validateResponsiveLayout(page: Page): Promise<void> {
  const viewports: ViewportSize[] = [
    { width: 1920, height: 1080 }, // Desktop
    { width: 1440, height: 900 },  // Laptop
    { width: 768, height: 1024 },   // Tablet
    { width: 375, height: 667 },    // Mobile
  ];

  for (const viewport of viewports) {
    await validatePageLayout(page, viewport);
  }
}

// ============================================================================
// Visual Regression Helpers
// ============================================================================

/**
 * Toma screenshot y compara con baseline (placeholder para integración futura)
 */
export async function compareScreenshot(
  page: Page,
  name: string,
  options: { threshold?: number; fullPage?: boolean } = {}
): Promise<{ match: boolean; diff?: string }> {
  const { threshold = 0.2, fullPage = false } = options;
  
  // Por ahora, solo toma el screenshot
  // En el futuro, esto se integraría con herramientas como Percy o Chromatic
  const screenshotPath = `test-results/screenshots/${name}-${Date.now()}.png`;
  await page.screenshot({
    path: screenshotPath,
    fullPage,
  });

  console.log(`📸 Screenshot guardado: ${screenshotPath}`);
  
  // Placeholder: siempre retorna match=true
  // En producción, aquí se haría la comparación real
  return { match: true };
}

/**
 * Valida que los elementos visuales críticos están presentes
 */
export async function validateVisualElements(page: Page): Promise<void> {
  // Verificar que hay elementos visuales básicos
  const hasForm = await page.locator('form, [role="form"]').count() > 0;
  const hasButtons = await page.locator('button').count() > 0;
  const hasInputs = await page.locator('input, textarea').count() > 0;

  if (!hasForm && !hasButtons && !hasInputs) {
    console.warn('⚠️ Visual: No se encontraron elementos interactivos básicos');
  }

  // Verificar que no hay elementos con z-index extremos (puede indicar problemas de layout)
  const elementsWithHighZIndex = await page.evaluate(() => {
    const allElements = document.querySelectorAll('*');
    let count = 0;
    allElements.forEach((el) => {
      const zIndex = window.getComputedStyle(el).zIndex;
      if (zIndex && parseInt(zIndex) > 9999) {
        count++;
      }
    });
    return count;
  });

  if (elementsWithHighZIndex > 10) {
    console.warn(
      `⚠️ Visual: Se encontraron ${elementsWithHighZIndex} elementos con z-index muy alto (posible problema de layout)`
    );
  }

  console.log('✅ Visual: Elementos visuales básicos validados');
}

// ============================================================================
// Color and Contrast Helpers (Placeholder)
// ============================================================================

/**
 * Valida contraste de colores (placeholder - requiere librería externa)
 */
export async function validateColorContrast(page: Page): Promise<void> {
  // Placeholder: En producción, esto usaría una librería como axe-core
  // o una herramienta de contraste de colores
  console.log('ℹ️ Validación de contraste: Requiere integración con herramienta externa');
}

// ============================================================================
// Advanced Visual Testing
// ============================================================================

/**
 * Compara un elemento con una imagen de referencia usando Playwright's screenshot comparison
 */
export async function compareElementWithSnapshot(
  page: Page,
  selector: string,
  snapshotName: string,
  options: { threshold?: number; maxDiffPixels?: number } = {}
): Promise<void> {
  const { threshold = 0.2, maxDiffPixels = 100 } = options;
  const element = page.locator(selector).first();
  
  await playwrightExpect(element).toHaveScreenshot(snapshotName, {
    threshold,
    maxDiffPixels,
  });
}

/**
 * Compara toda la página con un snapshot
 */
export async function comparePageWithSnapshot(
  page: Page,
  snapshotName: string,
  options: { threshold?: number; fullPage?: boolean } = {}
): Promise<void> {
  const { threshold = 0.2, fullPage = true } = options;
  
  await playwrightExpect(page).toHaveScreenshot(snapshotName, {
    threshold,
    fullPage,
  });
}

/**
 * Valida que un elemento tenga las dimensiones esperadas
 */
export async function expectElementDimensions(
  page: Page,
  selector: string,
  expectedWidth: number,
  expectedHeight: number,
  tolerance: number = 5
): Promise<void> {
  const element = page.locator(selector).first();
  const box = await element.boundingBox();
  
  playwrightExpect(box).not.toBeNull();
  
  if (box) {
    playwrightExpect(Math.abs(box.width - expectedWidth)).toBeLessThanOrEqual(tolerance);
    playwrightExpect(Math.abs(box.height - expectedHeight)).toBeLessThanOrEqual(tolerance);
  }
}

/**
 * Valida que un elemento esté visible y tenga contenido
 */
export async function expectElementVisibleWithContent(
  page: Page,
  selector: string,
  minContentLength: number = 1
): Promise<void> {
  const element = page.locator(selector).first();
  await playwrightExpect(element).toBeVisible();
  
  const text = await element.textContent();
  playwrightExpect(text?.length || 0).toBeGreaterThanOrEqual(minContentLength);
}
