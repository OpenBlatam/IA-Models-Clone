/**
 * E2E Tests - Advanced Features
 */

import { test, expect } from '@playwright/test'

test.describe('Advanced Features', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should open and use templates panel', async ({ page }) => {
    // Find templates button
    const templatesButton = page.locator('button:has-text("Plantillas"), button[title*="Plantillas"]').first()
    if (await templatesButton.isVisible()) {
      await templatesButton.click()
      await page.waitForTimeout(500)

      // Check for templates list
      const templatesList = page.locator('text=/classification|regression|template/i').first()
      await expect(templatesList).toBeVisible({ timeout: 3000 })
    }
  })

  test('should open statistics panel', async ({ page }) => {
    // Find stats button
    const statsButton = page.locator('button:has-text("Estadísticas"), button[title*="Estadísticas"]').first()
    if (await statsButton.isVisible()) {
      await statsButton.click()
      await page.waitForTimeout(500)

      // Check for stats content
      const statsContent = page.locator('text=/total|success|rate|duration/i').first()
      await expect(statsContent).toBeVisible({ timeout: 3000 })
    }
  })

  test('should open real-time metrics panel', async ({ page }) => {
    // Enable proactive mode first
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Find metrics button
      const metricsButton = page.locator('button[title*="Métricas"], button[title*="Metrics"]').first()
      if (await metricsButton.isVisible()) {
        await metricsButton.click()
        await page.waitForTimeout(500)

        // Check for metrics content
        const metricsContent = page.locator('text=/builds|success|duration|queue/i').first()
        await expect(metricsContent).toBeVisible({ timeout: 3000 })
      }
    }
  })

  test('should open alerts panel', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Find alerts button
      const alertsButton = page.locator('button[title*="Alertas"], button[title*="Alerts"]').first()
      if (await alertsButton.isVisible()) {
        await alertsButton.click()
        await page.waitForTimeout(500)

        // Check for alerts content
        const alertsContent = page.locator('text=/alertas|alerts|inteligentes/i').first()
        await expect(alertsContent).toBeVisible({ timeout: 3000 })
      }
    }
  })

  test('should open favorites panel', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Find favorites button
      const favoritesButton = page.locator('button[title*="Favoritos"], button[title*="Favorites"]').first()
      if (await favoritesButton.isVisible()) {
        await favoritesButton.click()
        await page.waitForTimeout(500)

        // Check for favorites content
        const favoritesContent = page.locator('text=/favoritos|favorites/i').first()
        await expect(favoritesContent).toBeVisible({ timeout: 3000 })
      }
    }
  })

  test('should open smart history panel', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Find history button
      const historyButton = page.locator('button[title*="Historial"], button[title*="History"]').first()
      if (await historyButton.isVisible()) {
        await historyButton.click()
        await page.waitForTimeout(500)

        // Check for history content
        const historyContent = page.locator('text=/historial|history|inteligente/i').first()
        await expect(historyContent).toBeVisible({ timeout: 3000 })
      }
    }
  })

  test('should open quick commands panel', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Find commands button
      const commandsButton = page.locator('button[title*="Comandos"], button[title*="Commands"]').first()
      if (await commandsButton.isVisible()) {
        await commandsButton.click()
        await page.waitForTimeout(500)

        // Check for commands content
        const commandsContent = page.locator('text=/comandos|commands|rápidos/i').first()
        await expect(commandsContent).toBeVisible({ timeout: 3000 })
      }
    }
  })

  test('should open help panel', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Find help button
      const helpButton = page.locator('button[title*="Ayuda"], button[title*="Help"]').first()
      if (await helpButton.isVisible()) {
        await helpButton.click()
        await page.waitForTimeout(500)

        // Check for help content
        const helpContent = page.locator('text=/ayuda|help|contextual/i').first()
        await expect(helpContent).toBeVisible({ timeout: 3000 })
      }
    }
  })
})

test.describe('Keyboard Shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should open help with keyboard shortcut', async ({ page }) => {
    // Press Ctrl+? (or Cmd+? on Mac)
    await page.keyboard.press('Control+?')
    await page.waitForTimeout(500)

    // Check for help panel
    const helpContent = page.locator('text=/ayuda|help/i').first()
    await expect(helpContent).toBeVisible({ timeout: 3000 })
  })

  test('should toggle favorites with keyboard shortcut', async ({ page }) => {
    // Enable proactive mode first
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Press Ctrl+F
      await page.keyboard.press('Control+f')
      await page.waitForTimeout(500)

      // Check for favorites panel
      const favoritesContent = page.locator('text=/favoritos|favorites/i').first()
      await expect(favoritesContent).toBeVisible({ timeout: 3000 })
    }
  })
})










