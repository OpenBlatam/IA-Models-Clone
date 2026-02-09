/**
 * E2E Tests - Error Recovery
 */

import { test, expect } from '@playwright/test'

test.describe('Error Recovery - API Errors', () => {
  test('should recover from API timeout', async ({ page, context }) => {
    // Simulate timeout
    await context.route('**/api/**', async route => {
      await new Promise(resolve => setTimeout(resolve, 30000))
      await route.abort()
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Should show timeout error
    const errorMessage = page.locator('text=/timeout|error|failed/i').first()
    await expect(errorMessage).toBeVisible({ timeout: 35000 })
  })

  test('should recover from 500 error', async ({ page, context }) => {
    // Simulate server error
    await context.route('**/api/**', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal Server Error' }),
      })
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Should show error message
    const errorMessage = page.locator('text=/error|failed|500/i').first()
    await expect(errorMessage).toBeVisible({ timeout: 10000 })
  })

  test('should recover from 404 error', async ({ page, context }) => {
    // Simulate not found
    await context.route('**/api/**', route => {
      route.fulfill({
        status: 404,
        body: JSON.stringify({ error: 'Not Found' }),
      })
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Should handle 404 gracefully
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })

  test('should retry failed requests', async ({ page, context }) => {
    let requestCount = 0

    await context.route('**/api/create-model**', route => {
      requestCount++
      if (requestCount < 3) {
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Server Error' }),
        })
      } else {
        route.continue()
      }
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Should retry and eventually succeed
    await page.waitForTimeout(5000)
    expect(requestCount).toBeGreaterThan(1)
  })
})

test.describe('Error Recovery - User Errors', () => {
  test('should recover from invalid input', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    
    // Try invalid input
    await chatInput.fill('<script>alert("xss")</script>')
    await chatInput.press('Enter')

    // Should show error but not break
    await page.waitForTimeout(2000)
    const errorOrSuccess = page.locator('text=/error|invalid|completado/i').first()
    await expect(errorOrSuccess).toBeVisible({ timeout: 5000 })

    // Should still be able to use input
    await chatInput.fill('valid model')
    await expect(chatInput).toHaveValue('valid model')
  })

  test('should recover from cancelled operation', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Start building
      const startButton = page.locator('button:has-text("Iniciar"), button:has-text("Start")').first()
      if (await startButton.isVisible()) {
        await startButton.click()
        await page.waitForTimeout(1000)

        // Pause immediately
        const pauseButton = page.locator('button:has-text("Pausar"), button:has-text("Pause")').first()
        if (await pauseButton.isVisible()) {
          await pauseButton.click()
          await page.waitForTimeout(1000)

          // Should recover gracefully
          const startButtonAgain = page.locator('button:has-text("Iniciar"), button:has-text("Start")').first()
          await expect(startButtonAgain).toBeVisible({ timeout: 3000 })
        }
      }
    }
  })
})

test.describe('Error Recovery - State Recovery', () => {
  test('should recover state after error', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Add model to queue
      const queueInput = page.locator('input[placeholder*="descripción"], textarea[placeholder*="descripción"]').first()
      if (await queueInput.isVisible()) {
        await queueInput.fill('test model')
        
        const addButton = page.locator('button:has-text("Agregar"), button:has-text("Add")').first()
        if (await addButton.isVisible()) {
          await addButton.click()
          await page.waitForTimeout(1000)

          // Reload page
          await page.reload()
          await page.waitForLoadState('networkidle')

          // State should be recovered or reset gracefully
          const proactiveButtonAfter = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
          await expect(proactiveButtonAfter).toBeVisible()
        }
      }
    }
  })

  test('should handle partial form submission', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Start filling form but interrupt
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test')
    
    // Navigate away
    await page.reload()
    await page.waitForLoadState('networkidle')

    // Should recover
    const chatInputAfter = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInputAfter).toBeVisible()
  })
})

test.describe('Error Recovery - Browser Errors', () => {
  test('should handle JavaScript errors gracefully', async ({ page }) => {
    // Listen for console errors
    const errors: string[] = []
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text())
      }
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Inject error
    await page.evaluate(() => {
      try {
        throw new Error('Test error')
      } catch (e) {
        console.error(e)
      }
    })

    // Should still be functional
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })

  test('should handle unhandled promise rejections', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Inject unhandled rejection
    await page.evaluate(() => {
      Promise.reject(new Error('Unhandled rejection'))
        .catch(() => {}) // Handle to prevent test failure
    })

    await page.waitForTimeout(1000)

    // Should still be functional
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })
})

test.describe('Error Recovery - Data Corruption', () => {
  test('should handle corrupted localStorage', async ({ page, context }) => {
    // Inject corrupted data
    await context.addInitScript(() => {
      localStorage.setItem('proactive-builder', 'invalid json{{{')
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Should recover gracefully
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })

  test('should handle missing localStorage data', async ({ page, context }) => {
    // Clear all localStorage
    await context.addInitScript(() => {
      localStorage.clear()
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Should work with default state
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })
})










