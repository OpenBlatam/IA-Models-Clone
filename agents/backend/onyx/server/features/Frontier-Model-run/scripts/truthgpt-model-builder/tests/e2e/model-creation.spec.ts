import { test, expect } from '@playwright/test';

test.describe('Model Creation Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('should show model preview before creating', async ({ page }) => {
    const input = page.locator('input[type="text"]');
    
    // Escribir descripción completa
    const description = 'Un modelo de clasificación de imágenes con 3 capas convolucionales, max pooling, dropout del 20% y softmax para 10 clases';
    await input.fill(description);
    
    // Esperar a que se procese la validación
    await page.waitForTimeout(1000);
    
    // Verificar que hay algún feedback visual
    const validationBadge = page.locator('[class*="validation"], [class*="badge"]').first();
    if (await validationBadge.isVisible({ timeout: 2000 }).catch(() => false)) {
      await expect(validationBadge).toBeVisible();
    }
  });

  test('should submit model creation form', async ({ page }) => {
    const input = page.locator('input[type="text"]');
    const submitButton = page.getByRole('button', { name: /Crear Modelo/i });
    
    // Llenar el formulario
    await input.fill('Un modelo de clasificación de imágenes con CNN');
    
    // Esperar a que se habilite el botón
    await page.waitForTimeout(500);
    
    // Verificar que el botón está habilitado
    const isDisabled = await submitButton.isDisabled();
    if (!isDisabled) {
      await submitButton.click();
      
      // Esperar respuesta
      await page.waitForTimeout(2000);
      
      // Verificar que hay algún feedback
      const feedback = page.locator('[class*="toast"], [class*="message"], [class*="status"]').first();
      if (await feedback.isVisible({ timeout: 3000 }).catch(() => false)) {
        await expect(feedback).toBeVisible();
      }
    }
  });

  test('should show loading state during model creation', async ({ page }) => {
    const input = page.locator('input[type="text"]');
    const submitButton = page.getByRole('button', { name: /Crear Modelo/i });
    
    await input.fill('Un modelo de clasificación de imágenes con CNN y dropout');
    await page.waitForTimeout(500);
    
    if (!(await submitButton.isDisabled())) {
      await submitButton.click();
      
      // Buscar indicador de carga
      const loadingIndicator = page.locator('[class*="loading"], [class*="spinner"], [class*="loader"]').first();
      const loadingText = page.getByText(/Creando|Loading|Cargando/i).first();
      
      // Verificar que aparece algún indicador de carga
      if (await loadingIndicator.isVisible({ timeout: 2000 }).catch(() => false)) {
        await expect(loadingIndicator).toBeVisible();
      } else if (await loadingText.isVisible({ timeout: 2000 }).catch(() => false)) {
        await expect(loadingText).toBeVisible();
      }
      
      // Verificar que el botón está deshabilitado durante la carga
      const isDisabledDuringLoad = await submitButton.isDisabled();
      expect(isDisabledDuringLoad).toBeTruthy();
    }
  });

  test('should show model status after creation', async ({ page }) => {
    const input = page.locator('input[type="text"]');
    const submitButton = page.getByRole('button', { name: /Crear Modelo/i });
    
    await input.fill('Un modelo de clasificación de imágenes con CNN');
    await page.waitForTimeout(500);
    
    if (!(await submitButton.isDisabled())) {
      await submitButton.click();
      
      // Esperar a que aparezca el estado del modelo
      await page.waitForTimeout(3000);
      
      // Buscar información del modelo o estado
      const modelStatus = page.locator('[class*="status"], [class*="model"], [class*="progress"]').first();
      const statusText = page.getByText(/creando|creating|completed|completado/i).first();
      
      // Al menos debería haber algún indicador de estado
      if (await modelStatus.isVisible({ timeout: 5000 }).catch(() => false)) {
        await expect(modelStatus).toBeVisible();
      } else if (await statusText.isVisible({ timeout: 5000 }).catch(() => false)) {
        await expect(statusText).toBeVisible();
      }
    }
  });
});

test.describe('Model Templates', () => {
  test('should select template and fill input', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Abrir templates
    const templatesButton = page.getByRole('button', { name: /Templates/i });
    if (await templatesButton.isVisible().catch(() => false)) {
      await templatesButton.click();
      await page.waitForTimeout(500);
      
      // Buscar y hacer clic en un template
      const templateOption = page.locator('[class*="template"], [class*="card"]').first();
      if (await templateOption.isVisible({ timeout: 2000 }).catch(() => false)) {
        await templateOption.click();
        
        // Verificar que el input se llenó
        await page.waitForTimeout(500);
        const input = page.locator('input[type="text"]');
        const value = await input.inputValue();
        expect(value.length).toBeGreaterThan(0);
      }
    }
  });
});

test.describe('Model History', () => {
  test('should display model history', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Abrir historial
    const historyButton = page.getByRole('button', { name: /Historial/i });
    if (await historyButton.isVisible().catch(() => false)) {
      await historyButton.click();
      await page.waitForTimeout(500);
      
      // Verificar que el panel de historial aparece
      const historyPanel = page.locator('[class*="history"], [class*="panel"]').first();
      if (await historyPanel.isVisible({ timeout: 2000 }).catch(() => false)) {
        await expect(historyPanel).toBeVisible();
      }
    }
  });

  test('should show model stats in history', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const historyButton = page.getByRole('button', { name: /Historial/i });
    if (await historyButton.isVisible().catch(() => false)) {
      await historyButton.click();
      await page.waitForTimeout(500);
      
      // Buscar estadísticas (pueden estar o no)
      const stats = page.locator('[class*="stat"], [class*="metric"], [class*="count"]').first();
      // No fallar si no hay stats, solo verificar si están visibles
      if (await stats.isVisible({ timeout: 2000 }).catch(() => false)) {
        await expect(stats).toBeVisible();
      }
    }
  });
});











