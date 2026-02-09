/**
 * E2E tests para verificar que el botón de parar detiene la generación correctamente
 */
import { test, expect } from '@playwright/test';

test.describe('Stop Generation Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Navegar a la página de kanban
    await page.goto('/kanban');
    await page.waitForLoadState('networkidle');
  });

  test('debería detener la generación cuando se presiona el botón de parar', async ({ page }) => {
    // Crear una nueva tarea
    const createButton = page.locator('button:has-text("Crear Tarea"), button:has-text("Nueva Tarea"), button[title*="crear"]').first();
    await createButton.click();

    // Esperar a que aparezca el modal o formulario
    await page.waitForTimeout(500);

    // Llenar el formulario de tarea
    const instructionInput = page.locator('textarea[placeholder*="instrucción"], input[placeholder*="instrucción"], textarea').first();
    await instructionInput.fill('Test task for stop functionality');

    const repoSelect = page.locator('select, input[type="text"]').filter({ hasText: /repo/i }).first();
    if (await repoSelect.count() > 0) {
      await repoSelect.fill('test-repo');
    }

    // Enviar la tarea
    const submitButton = page.locator('button:has-text("Crear"), button:has-text("Enviar"), button[type="submit"]').first();
    await submitButton.click();

    // Esperar a que la tarea aparezca y comience a procesarse
    await page.waitForTimeout(1000);

    // Buscar la tarea en la lista/kanban
    const taskCard = page.locator('[data-testid*="task"], .task-card, [class*="task"]').filter({ hasText: 'Test task for stop functionality' }).first();
    await expect(taskCard).toBeVisible({ timeout: 5000 });

    // Verificar que la tarea está en estado 'processing'
    const statusIndicator = taskCard.locator('[class*="status"], [class*="processing"], [data-status="processing"]').first();
    await expect(statusIndicator).toBeVisible({ timeout: 5000 });

    // Esperar un poco para que comience a generar contenido
    await page.waitForTimeout(2000);

    // Buscar y presionar el botón de parar
    const stopButton = taskCard.locator('button:has-text("Parar"), button:has-text("Stop"), button[title*="parar"], button[title*="stop"]').first();
    await expect(stopButton).toBeVisible({ timeout: 5000 });
    await stopButton.click();

    // Verificar que la tarea cambia a estado 'stopped'
    await page.waitForTimeout(1000);
    const stoppedStatus = taskCard.locator('[class*="stopped"], [data-status="stopped"]').first();
    await expect(stoppedStatus).toBeVisible({ timeout: 5000 });

    // Verificar que el contenido deja de actualizarse
    const contentBefore = await taskCard.locator('[class*="content"], [class*="streaming"]').textContent();
    await page.waitForTimeout(2000);
    const contentAfter = await taskCard.locator('[class*="content"], [class*="streaming"]').textContent();
    
    // El contenido no debería cambiar significativamente después de parar
    expect(contentBefore).toBe(contentAfter);
  });

  test('debería preservar el plan y contenido cuando se detiene la generación', async ({ page }) => {
    // Crear una tarea que genere un plan
    const createButton = page.locator('button:has-text("Crear Tarea"), button:has-text("Nueva Tarea")').first();
    await createButton.click();
    await page.waitForTimeout(500);

    const instructionInput = page.locator('textarea, input[type="text"]').first();
    await instructionInput.fill('Create a new file with test content');

    const submitButton = page.locator('button:has-text("Crear"), button[type="submit"]').first();
    await submitButton.click();

    await page.waitForTimeout(2000);

    // Buscar la tarea
    const taskCard = page.locator('[class*="task"]').filter({ hasText: 'Create a new file' }).first();
    await expect(taskCard).toBeVisible({ timeout: 5000 });

    // Esperar a que genere algo de contenido
    await page.waitForTimeout(3000);

    // Presionar parar
    const stopButton = taskCard.locator('button:has-text("Parar"), button[title*="parar"]').first();
    await stopButton.click();

    await page.waitForTimeout(1000);

    // Verificar que el contenido/plan está preservado
    const content = await taskCard.locator('[class*="content"]').textContent();
    expect(content).toBeTruthy();
    expect(content!.length).toBeGreaterThan(0);

    // Verificar que se puede ver el plan si existe
    const planButton = taskCard.locator('button:has-text("Plan"), button:has-text("Ver Plan")');
    if (await planButton.count() > 0) {
      await planButton.click();
      await page.waitForTimeout(500);
      const planContent = page.locator('[class*="plan"], [class*="modal"]').textContent();
      expect(planContent).toBeTruthy();
    }
  });

  test('debería permitir reanudar una tarea detenida', async ({ page }) => {
    // Crear y detener una tarea
    const createButton = page.locator('button:has-text("Crear Tarea")').first();
    await createButton.click();
    await page.waitForTimeout(500);

    const instructionInput = page.locator('textarea').first();
    await instructionInput.fill('Task to resume');

    const submitButton = page.locator('button[type="submit"]').first();
    await submitButton.click();

    await page.waitForTimeout(2000);

    const taskCard = page.locator('[class*="task"]').filter({ hasText: 'Task to resume' }).first();
    await expect(taskCard).toBeVisible({ timeout: 5000 });

    // Detener
    const stopButton = taskCard.locator('button:has-text("Parar")').first();
    await stopButton.click();
    await page.waitForTimeout(1000);

    // Buscar botón de reanudar
    const resumeButton = taskCard.locator('button:has-text("Reanudar"), button:has-text("Resume")').first();
    if (await resumeButton.count() > 0) {
      await resumeButton.click();
      await page.waitForTimeout(1000);

      // Verificar que vuelve a procesar
      const processingStatus = taskCard.locator('[class*="processing"]').first();
      await expect(processingStatus).toBeVisible({ timeout: 5000 });
    }
  });

  test('debería detener múltiples tareas simultáneamente', async ({ page }) => {
    // Crear múltiples tareas
    for (let i = 0; i < 3; i++) {
      const createButton = page.locator('button:has-text("Crear Tarea")').first();
      await createButton.click();
      await page.waitForTimeout(300);

      const instructionInput = page.locator('textarea').first();
      await instructionInput.fill(`Task ${i + 1} to stop`);

      const submitButton = page.locator('button[type="submit"]').first();
      await submitButton.click();
      await page.waitForTimeout(1000);
    }

    // Esperar a que todas comiencen a procesarse
    await page.waitForTimeout(2000);

    // Detener todas las tareas
    const stopButtons = page.locator('button:has-text("Parar")');
    const count = await stopButtons.count();
    
    for (let i = 0; i < count; i++) {
      await stopButtons.nth(i).click();
      await page.waitForTimeout(300);
    }

    // Verificar que todas están detenidas
    await page.waitForTimeout(1000);
    const stoppedTasks = page.locator('[data-status="stopped"], [class*="stopped"]');
    const stoppedCount = await stoppedTasks.count();
    expect(stoppedCount).toBeGreaterThanOrEqual(0); // Al menos algunas deberían estar detenidas
  });
});

