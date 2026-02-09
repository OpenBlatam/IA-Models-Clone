/**
 * E2E Tests - Complete Integration Flow
 */

import { test, expect } from '@playwright/test'

test.describe('Complete Model Lifecycle', () => {
  test('should complete full model lifecycle', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Step 1: Create model via chat
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('classification model for text categorization')
    await chatInput.press('Enter')

    // Step 2: Wait for model creation
    await page.waitForTimeout(2000)

    // Step 3: Check model appears in completed list
    const modelInList = page.locator('text=/classification|categorization/i').first()
    await expect(modelInList).toBeVisible({ timeout: 30000 })

    // Step 4: Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Step 5: Add model to queue
      const queueInput = page.locator('input[placeholder*="descripción"], textarea[placeholder*="descripción"]').first()
      if (await queueInput.isVisible()) {
        await queueInput.fill('regression model')
        
        const addButton = page.locator('button:has-text("Agregar"), button:has-text("Add")').first()
        if (await addButton.isVisible()) {
          await addButton.click()
          await page.waitForTimeout(1000)

          // Step 6: Start proactive building
          const startButton = page.locator('button:has-text("Iniciar"), button:has-text("Start")').first()
          if (await startButton.isVisible()) {
            await startButton.click()
            await page.waitForTimeout(2000)

            // Step 7: Check metrics
            const metricsButton = page.locator('button[title*="Métricas"], button[title*="Metrics"]').first()
            if (await metricsButton.isVisible()) {
              await metricsButton.click()
              await page.waitForTimeout(500)

              const metricsContent = page.locator('text=/builds|success|duration/i').first()
              await expect(metricsContent).toBeVisible({ timeout: 3000 })
            }
          }
        }
      }
    }
  })
})

test.describe('Batch Operations', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should add multiple models to queue', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      const models = [
        'classification model',
        'regression model',
        'sentiment analysis model',
      ]

      for (const model of models) {
        const queueInput = page.locator('input[placeholder*="descripción"], textarea[placeholder*="descripción"]').first()
        if (await queueInput.isVisible()) {
          await queueInput.fill(model)
          
          const addButton = page.locator('button:has-text("Agregar"), button:has-text("Add")').first()
          if (await addButton.isVisible()) {
            await addButton.click()
            await page.waitForTimeout(500)
          }
        }
      }

      // Check queue has multiple items
      await page.waitForTimeout(1000)
      const queueItems = page.locator('text=/classification|regression|sentiment/i')
      const count = await queueItems.count()
      expect(count).toBeGreaterThan(0)
    }
  })
})

test.describe('Search and Filter', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should search models in history', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Open history panel
      const historyButton = page.locator('button[title*="Historial"], button[title*="History"]').first()
      if (await historyButton.isVisible()) {
        await historyButton.click()
        await page.waitForTimeout(500)

        // Find search input
        const searchInput = page.locator('input[placeholder*="buscar"], input[placeholder*="search"]').first()
        if (await searchInput.isVisible()) {
          await searchInput.fill('classification')
          await page.waitForTimeout(1000)

          // Check for filtered results
          const results = page.locator('text=/classification/i').first()
          await expect(results).toBeVisible({ timeout: 3000 })
        }
      }
    }
  })
})

test.describe('Export Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should export models', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Find export button
      const exportButton = page.locator('button[title*="Exportar"], button[title*="Export"]').first()
      if (await exportButton.isVisible()) {
        await exportButton.click()
        await page.waitForTimeout(500)

        // Check for export menu
        const exportMenu = page.locator('text=/json|csv|yaml|markdown/i').first()
        await expect(exportMenu).toBeVisible({ timeout: 3000 })
      }
    }
  })
})

test.describe('Responsive Design', () => {
  test('should work on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Check main elements are visible
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })

  test('should work on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 })
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Check main elements are visible
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })
})










