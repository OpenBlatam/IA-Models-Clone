/**
 * E2E Tests - User Workflows
 */

import { test, expect } from '@playwright/test'

test.describe('User Workflows - Complete Model Building', () => {
  test('should complete full workflow: create, view, favorite, export', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Step 1: Create model
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('classification model for text categorization')
    await chatInput.press('Enter')
    await page.waitForTimeout(3000)

    // Step 2: View in completed list
    const modelInList = page.locator('text=/classification|categorization/i').first()
    await expect(modelInList).toBeVisible({ timeout: 30000 })

    // Step 3: Add to favorites
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      const favoritesButton = page.locator('button[title*="Favoritos"], button[title*="Favorites"]').first()
      if (await favoritesButton.isVisible()) {
        await favoritesButton.click()
        await page.waitForTimeout(500)
      }
    }

    // Step 4: Export
    const exportButton = page.locator('button[title*="Exportar"], button[title*="Export"]').first()
    if (await exportButton.isVisible()) {
      await exportButton.click()
      await page.waitForTimeout(500)

      const exportMenu = page.locator('text=/json|csv|yaml/i').first()
      await expect(exportMenu).toBeVisible({ timeout: 3000 })
    }
  })
})

test.describe('User Workflows - Proactive Building Session', () => {
  test('should complete proactive building session', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Step 1: Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    await proactiveButton.click()
    await page.waitForTimeout(500)

    // Step 2: Add models to queue
    const queueInput = page.locator('input[placeholder*="descripción"], textarea[placeholder*="descripción"]').first()
    const addButton = page.locator('button:has-text("Agregar"), button:has-text("Add")').first()

    if (await queueInput.isVisible() && await addButton.isVisible()) {
      const models = ['classification', 'regression', 'sentiment']
      
      for (const model of models) {
        await queueInput.fill(`${model} model`)
        await addButton.click()
        await page.waitForTimeout(500)
      }
    }

    // Step 3: Start building
    const startButton = page.locator('button:has-text("Iniciar"), button:has-text("Start")').first()
    if (await startButton.isVisible()) {
      await startButton.click()
      await page.waitForTimeout(2000)

      // Step 4: Monitor metrics
      const metricsButton = page.locator('button[title*="Métricas"], button[title*="Metrics"]').first()
      if (await metricsButton.isVisible()) {
        await metricsButton.click()
        await page.waitForTimeout(1000)

        const metricsContent = page.locator('text=/builds|success|duration/i').first()
        await expect(metricsContent).toBeVisible({ timeout: 3000 })
      }
    }
  })
})

test.describe('User Workflows - Template Usage', () => {
  test('should use template to create model', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Step 1: Open templates
    const templatesButton = page.locator('button:has-text("Plantillas"), button[title*="Plantillas"]').first()
    if (await templatesButton.isVisible()) {
      await templatesButton.click()
      await page.waitForTimeout(500)

      // Step 2: Select template
      const template = page.locator('text=/classification|regression|template/i').first()
      if (await template.isVisible()) {
        await template.click()
        await page.waitForTimeout(500)

        // Step 3: Use template
        const useButton = page.locator('button:has-text("Usar"), button:has-text("Use")').first()
        if (await useButton.isVisible()) {
          await useButton.click()
          await page.waitForTimeout(1000)

          // Should fill form with template
          const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
          const value = await chatInput.inputValue()
          expect(value.length).toBeGreaterThan(0)
        }
      }
    }
  })
})

test.describe('User Workflows - Search and Filter', () => {
  test('should search and filter models', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Create some models first
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    for (let i = 0; i < 3; i++) {
      await chatInput.fill(`classification model ${i}`)
      await chatInput.press('Enter')
      await page.waitForTimeout(2000)
    }

    await page.waitForTimeout(2000)

    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Open history
      const historyButton = page.locator('button[title*="Historial"], button[title*="History"]').first()
      if (await historyButton.isVisible()) {
        await historyButton.click()
        await page.waitForTimeout(500)

        // Search
        const searchInput = page.locator('input[placeholder*="buscar"], input[placeholder*="search"]').first()
        if (await searchInput.isVisible()) {
          await searchInput.fill('classification')
          await page.waitForTimeout(1000)

          // Should show filtered results
          const results = page.locator('text=/classification/i')
          const count = await results.count()
          expect(count).toBeGreaterThan(0)
        }
      }
    }
  })
})

test.describe('User Workflows - Statistics Analysis', () => {
  test('should analyze model statistics', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Create some models
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    for (let i = 0; i < 5; i++) {
      await chatInput.fill(`test model ${i}`)
      await chatInput.press('Enter')
      await page.waitForTimeout(2000)
    }

    await page.waitForTimeout(2000)

    // Open statistics
    const statsButton = page.locator('button:has-text("Estadísticas"), button[title*="Estadísticas"]').first()
    if (await statsButton.isVisible()) {
      await statsButton.click()
      await page.waitForTimeout(1000)

      // Check for stats content
      const statsContent = page.locator('text=/total|success|rate|duration|average/i').first()
      await expect(statsContent).toBeVisible({ timeout: 3000 })
    }
  })
})

test.describe('User Workflows - Keyboard Navigation', () => {
  test('should navigate entirely with keyboard', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Tab to input
    await page.keyboard.press('Tab')
    await page.waitForTimeout(100)

    // Type model description
    await page.keyboard.type('classification model')
    await page.keyboard.press('Enter')
    await page.waitForTimeout(2000)

    // Use keyboard shortcuts
    await page.keyboard.press('Control+?') // Help
    await page.waitForTimeout(500)

    // Close with Escape
    await page.keyboard.press('Escape')
    await page.waitForTimeout(500)

    // Should still be functional
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })
})










