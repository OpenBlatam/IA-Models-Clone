/**
 * E2E Tests - Basic Flow
 */

import { test, expect } from '@playwright/test'

test.describe('Basic Model Creation Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    // Wait for page to load
    await page.waitForLoadState('networkidle')
  })

  test('should create a model from chat interface', async ({ page }) => {
    // Find chat input
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()

    // Type model description
    await chatInput.fill('classification model for text categorization')
    await chatInput.press('Enter')

    // Wait for model creation to start
    await page.waitForTimeout(1000)

    // Check for success message or model status
    const successIndicator = page.locator('text=/completado|success|creado/i').first()
    await expect(successIndicator).toBeVisible({ timeout: 30000 })
  })

  test('should display model in completed builds list', async ({ page }) => {
    // Create model
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('regression model for predictions')
    await chatInput.press('Enter')

    // Wait for completion
    await page.waitForTimeout(2000)

    // Check for model in list
    const modelList = page.locator('text=/regression|prediction/i').first()
    await expect(modelList).toBeVisible({ timeout: 30000 })
  })

  test('should show validation errors for invalid input', async ({ page }) => {
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    
    // Try empty input
    await chatInput.fill('')
    await chatInput.press('Enter')

    // Should show validation error
    const errorMessage = page.locator('text=/error|invalid|required/i').first()
    await expect(errorMessage).toBeVisible({ timeout: 5000 })
  })

  test('should navigate through main sections', async ({ page }) => {
    // Check for main navigation elements
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)
    }

    // Check for stats button
    const statsButton = page.locator('button:has-text("Estadísticas"), button[title*="Estadísticas"]').first()
    if (await statsButton.isVisible()) {
      await statsButton.click()
      await page.waitForTimeout(500)
    }
  })
})

test.describe('Proactive Builder Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should enable proactive mode', async ({ page }) => {
    // Find and click proactive button
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    await expect(proactiveButton).toBeVisible()
    await proactiveButton.click()

    // Check for proactive builder panel
    const proactivePanel = page.locator('text=/constructor proactivo|proactive/i').first()
    await expect(proactivePanel).toBeVisible({ timeout: 2000 })
  })

  test('should add model to queue', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    await proactiveButton.click()
    await page.waitForTimeout(500)

    // Find input in proactive builder
    const queueInput = page.locator('input[placeholder*="descripción"], textarea[placeholder*="descripción"]').first()
    if (await queueInput.isVisible()) {
      await queueInput.fill('classification model')
      
      // Find add button
      const addButton = page.locator('button:has-text("Agregar"), button:has-text("Add")').first()
      if (await addButton.isVisible()) {
        await addButton.click()
        
        // Check for model in queue
        await page.waitForTimeout(1000)
        const queueItem = page.locator('text=/classification/i').first()
        await expect(queueItem).toBeVisible({ timeout: 5000 })
      }
    }
  })

  test('should start and pause proactive building', async ({ page }) => {
    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    await proactiveButton.click()
    await page.waitForTimeout(500)

    // Find start button
    const startButton = page.locator('button:has-text("Iniciar"), button:has-text("Start")').first()
    if (await startButton.isVisible()) {
      await startButton.click()
      await page.waitForTimeout(1000)

      // Check for pause button
      const pauseButton = page.locator('button:has-text("Pausar"), button:has-text("Pause")').first()
      await expect(pauseButton).toBeVisible({ timeout: 5000 })
    }
  })
})










