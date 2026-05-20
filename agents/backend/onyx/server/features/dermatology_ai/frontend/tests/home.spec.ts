import { test, expect } from '@playwright/test';
import { waitForPageLoad, expectToastMessage } from './helpers/test-helpers';

test.describe('Home Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForPageLoad(page);
  });

  test('should display the main heading and description', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /análisis de piel con ia/i })).toBeVisible();
    await expect(
      page.getByText(/sube una imagen o video de tu piel/i)
    ).toBeVisible();
  });

  test('should display image and video upload tabs', async ({ page }) => {
    await expect(page.getByRole('tab', { name: /imagen/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /video/i })).toBeVisible();
  });

  test('should show authentication badge when user is not logged in', async ({ page }) => {
    const badge = page.getByText(/inicia sesión para guardar tu historial/i);
    await expect(badge).toBeVisible();
  });

  test('should display empty state when no file is selected', async ({ page }) => {
    await expect(page.getByText(/comienza tu análisis/i)).toBeVisible();
    await expect(page.getByText(/sube una imagen de tu piel para comenzar/i)).toBeVisible();
  });

  test('should allow file upload via drag and drop', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]').first();
    
    const buffer = Buffer.from('fake image content');
    await fileInput.setInputFiles({
      name: 'test-image.jpg',
      mimeType: 'image/jpeg',
      buffer,
    });
    
    await expect(page.getByAltText('Preview')).toBeVisible({ timeout: 5000 });
  });

  test('should allow file upload via click', async ({ page }) => {
    const dropzone = page.getByText(/arrastra una imagen aquí o haz clic para seleccionar/i);
    await dropzone.click();
    
    const fileInput = page.locator('input[type="file"]').first();
    const buffer = Buffer.from('fake image content');
    await fileInput.setInputFiles({
      name: 'test-image.jpg',
      mimeType: 'image/jpeg',
      buffer,
    });
    
    await expect(page.getByAltText('Preview')).toBeVisible({ timeout: 5000 });
  });

  test('should display file preview after upload', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]').first();
    const buffer = Buffer.from('fake image content');
    await fileInput.setInputFiles({
      name: 'test-image.jpg',
      mimeType: 'image/jpeg',
      buffer,
    });
    
    await expect(page.getByAltText('Preview')).toBeVisible();
    await expect(page.getByText(/imagen seleccionada/i)).toBeVisible();
  });

  test('should allow removing uploaded file', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]').first();
    const buffer = Buffer.from('fake image content');
    await fileInput.setInputFiles({
      name: 'test-image.jpg',
      mimeType: 'image/jpeg',
      buffer,
    });
    await expect(page.getByAltText('Preview')).toBeVisible();
    
    const removeButton = page.getByRole('button', { name: /eliminar imagen/i });
    await removeButton.click();
    
    await expect(page.getByText(/comienza tu análisis/i)).toBeVisible();
  });

  test('should show analyze button when file is selected', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]').first();
    const buffer = Buffer.from('fake image content');
    await fileInput.setInputFiles({
      name: 'test-image.jpg',
      mimeType: 'image/jpeg',
      buffer,
    });
    await expect(page.getByAltText('Preview')).toBeVisible();
    
    await expect(page.getByRole('button', { name: /analizar imagen/i })).toBeVisible();
  });

  test('should show recommendations button for image uploads', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]').first();
    const buffer = Buffer.from('fake image content');
    await fileInput.setInputFiles({
      name: 'test-image.jpg',
      mimeType: 'image/jpeg',
      buffer,
    });
    await expect(page.getByAltText('Preview')).toBeVisible();
    
    await expect(
      page.getByRole('button', { name: /obtener recomendaciones completas/i })
    ).toBeVisible();
  });

  test('should switch between image and video tabs', async ({ page }) => {
    await page.getByRole('tab', { name: /video/i }).click();
    
    await expect(
      page.getByText(/los videos se analizan extrayendo frames clave/i)
    ).toBeVisible();
    
    await page.getByRole('tab', { name: /imagen/i }).click();
    
    await expect(page.getByText(/arrastra una imagen aquí/i)).toBeVisible();
  });

  test('should show error when trying to analyze without selecting file', async ({ page }) => {
    const analyzeButton = page.getByRole('button', { name: /analizar imagen/i });
    
    if (await analyzeButton.isVisible()) {
      await analyzeButton.click();
      await expectToastMessage(page, /por favor selecciona una imagen primero/i);
    }
  });

  test('should display file format information', async ({ page }) => {
    await expect(page.getByText(/jpg, png, webp/i)).toBeVisible();
  });
});

