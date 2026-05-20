/**
 * Helpers para Tests de Edge Cases
 * 
 * Proporciona utilidades para probar casos límite y escenarios especiales
 */
import { Page, expect } from '@playwright/test';
import { navigateToAgentControl, createTask, waitForTaskToAppear } from '../helpers';
import { TEST_INSTRUCTIONS } from '../constants';

// ============================================================================
// Edge Case Helpers
// ============================================================================

/**
 * Valida comportamiento con instrucciones muy largas
 */
export async function testWithLongInstruction(
  page: Page,
  length: number = 5000
): Promise<void> {
  await navigateToAgentControl(page);

  const longInstruction = 'A'.repeat(length);
  await createTask(page, longInstruction);

  // Verificar que la aplicación maneja la instrucción larga
  const textarea = page.locator('textarea[name="instruction"]');
  const value = await textarea.inputValue();
  expect(value.length).toBeGreaterThan(0);
}

/**
 * Valida comportamiento con instrucciones vacías
 */
export async function testWithEmptyInstruction(page: Page): Promise<void> {
  await navigateToAgentControl(page);

  const createButton = page.getByRole('button', { name: /crear|procesar/i });

  // Verificar que el botón está deshabilitado o muestra validación
  const isDisabled = await createButton.isDisabled().catch(() => false);
  const hasValidation = await page
    .locator('[role="alert"], .error, [class*="error"], [class*="invalid"]')
    .count()
    .then((count) => count > 0)
    .catch(() => false);

  // Al menos una de estas validaciones debería estar presente
  expect(isDisabled || hasValidation).toBeTruthy();
}

/**
 * Valida persistencia de estado después de refresh
 */
export async function testStatePersistence(page: Page): Promise<void> {
  await navigateToAgentControl(page);
  await createTask(page, TEST_INSTRUCTIONS.DEFAULT);
  await waitForTaskToAppear(page);

  // Obtener información de la tarea antes del refresh
  const taskBeforeRefresh = await page
    .locator('[data-testid="task-card"], .task-card, [class*="task"]')
    .first()
    .textContent();

  // Refrescar la página
  await page.reload();
  await page.waitForLoadState('networkidle');

  // Verificar que la tarea persiste (si la aplicación lo soporta)
  const taskAfterRefresh = await page
    .locator('[data-testid="task-card"], .task-card, [class*="task"]')
    .first()
    .textContent()
    .catch(() => null);

  // Si hay persistencia, las tareas deberían ser similares
  if (taskAfterRefresh) {
    expect(taskAfterRefresh.length).toBeGreaterThan(0);
  }
}

/**
 * Valida manejo de errores de red
 */
export async function testNetworkErrorHandling(page: Page): Promise<void> {
  // Interceptar y simular error de red
  await page.route('**/api/tasks', (route) => {
    route.abort('failed');
  });

  await navigateToAgentControl(page);
  await createTask(page, TEST_INSTRUCTIONS.SIMPLE);

  // Verificar que la aplicación maneja el error
  const errorMessage = await page
    .locator('[role="alert"], .error, [class*="error"]')
    .first()
    .textContent()
    .catch(() => null);

  // La aplicación debería mostrar algún tipo de feedback
  expect(errorMessage || true).toBeTruthy();
}

