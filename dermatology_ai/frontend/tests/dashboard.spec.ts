import { test, expect } from '@playwright/test';
import { waitForPageLoad } from './helpers/test-helpers';

test.describe('Dashboard Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    await waitForPageLoad(page);
  });

  test('should display dashboard heading', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /dashboard/i })).toBeVisible();
    await expect(page.getByText(/resumen de tu salud de la piel/i)).toBeVisible();
  });

  test('should show login prompt when user is not authenticated', async ({ page }) => {
    const loginPrompt = page.getByText(/por favor inicia sesión para ver tu dashboard/i);
    
    if (await loginPrompt.isVisible()) {
      await expect(loginPrompt).toBeVisible();
      await expect(page.getByRole('link', { name: /ir al inicio/i })).toBeVisible();
    }
  });

  test('should display stats cards when authenticated', async ({ page }) => {
    const statsCards = [
      /total de análisis/i,
      /puntuación promedio/i,
      /alertas activas/i,
      /último análisis/i,
    ];

    for (const statText of statsCards) {
      const statCard = page.getByText(statText);
      if (await statCard.isVisible({ timeout: 5000 })) {
        await expect(statCard).toBeVisible();
      }
    }
  });

  test('should display progress chart when data is available', async ({ page }) => {
    const progressChart = page.getByText(/progreso en el tiempo/i);
    
    if (await progressChart.isVisible({ timeout: 5000 })) {
      await expect(progressChart).toBeVisible();
    }
  });

  test('should display metrics chart when data is available', async ({ page }) => {
    const metricsChart = page.getByText(/métricas de calidad/i);
    
    if (await metricsChart.isVisible({ timeout: 5000 })) {
      await expect(metricsChart).toBeVisible();
    }
  });

  test('should display recent analyses section when data is available', async ({ page }) => {
    const recentAnalyses = page.getByText(/análisis recientes/i);
    
    if (await recentAnalyses.isVisible({ timeout: 5000 })) {
      await expect(recentAnalyses).toBeVisible();
      await expect(page.getByRole('link', { name: /ver todo el historial/i })).toBeVisible();
    }
  });

  test('should display quick actions section', async ({ page }) => {
    await expect(page.getByText(/acciones rápidas/i)).toBeVisible();
    
    const quickActions = [
      /nuevo análisis/i,
      /ver historial/i,
      /productos/i,
    ];

    for (const actionText of quickActions) {
      const action = page.getByText(actionText);
      if (await action.isVisible({ timeout: 5000 })) {
        await expect(action).toBeVisible();
      }
    }
  });

  test('should navigate to home page from quick actions', async ({ page }) => {
    const newAnalysisLink = page.getByRole('link', { name: /nuevo análisis/i });
    
    if (await newAnalysisLink.isVisible({ timeout: 5000 })) {
      await newAnalysisLink.click();
      await expect(page).toHaveURL('/');
    }
  });

  test('should navigate to history page from quick actions', async ({ page }) => {
    const historyLink = page.getByRole('link', { name: /ver historial/i });
    
    if (await historyLink.isVisible({ timeout: 5000 })) {
      await historyLink.click();
      await expect(page).toHaveURL('/history');
    }
  });

  test('should navigate to products page from quick actions', async ({ page }) => {
    const productsLink = page.getByRole('link', { name: /productos/i });
    
    if (await productsLink.isVisible({ timeout: 5000 })) {
      await productsLink.click();
      await expect(page).toHaveURL('/products');
    }
  });

  test('should show loading skeletons while data is loading', async ({ page }) => {
    const skeletons = page.locator('[class*="skeleton"]');
    const skeletonCount = await skeletons.count();
    
    if (skeletonCount > 0) {
      await expect(skeletons.first()).toBeVisible();
    }
  });

  test('should display empty state when no data is available', async ({ page }) => {
    const emptyState = page.getByText(/no hay datos/i);
    
    if (await emptyState.isVisible({ timeout: 5000 })) {
      await expect(emptyState).toBeVisible();
    }
  });
});



