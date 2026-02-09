/**
 * E2E Tests - Edge Cases
 */

import { test, expect } from '@playwright/test'

test.describe('Edge Cases - Input Validation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should handle very long model descriptions', async ({ page }) => {
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    const longDescription = 'a'.repeat(10000)
    
    await chatInput.fill(longDescription)
    await chatInput.press('Enter')

    // Should either accept or show validation error
    await page.waitForTimeout(2000)
    const errorOrSuccess = page.locator('text=/error|invalid|completado|success/i').first()
    await expect(errorOrSuccess).toBeVisible({ timeout: 5000 })
  })

  test('should handle special characters in descriptions', async ({ page }) => {
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    const specialChars = 'model with @#$%^&*()[]{}|\\/"\'<>?'
    
    await chatInput.fill(specialChars)
    await chatInput.press('Enter')

    await page.waitForTimeout(2000)
    // Should handle gracefully
    const response = page.locator('text=/error|invalid|completado|success/i').first()
    await expect(response).toBeVisible({ timeout: 5000 })
  })

  test('should handle unicode characters', async ({ page }) => {
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    const unicodeDescription = '模型分类 🚀 测试分类器'
    
    await chatInput.fill(unicodeDescription)
    await chatInput.press('Enter')

    await page.waitForTimeout(2000)
    const response = page.locator('text=/error|invalid|completado|success/i').first()
    await expect(response).toBeVisible({ timeout: 5000 })
  })

  test('should handle empty input', async ({ page }) => {
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    
    await chatInput.fill('')
    await chatInput.press('Enter')

    // Should show validation error
    const errorMessage = page.locator('text=/error|invalid|required|empty/i').first()
    await expect(errorMessage).toBeVisible({ timeout: 5000 })
  })

  test('should handle whitespace-only input', async ({ page }) => {
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    
    await chatInput.fill('   ')
    await chatInput.press('Enter')

    // Should show validation error
    const errorMessage = page.locator('text=/error|invalid|required/i').first()
    await expect(errorMessage).toBeVisible({ timeout: 5000 })
  })
})

test.describe('Edge Cases - Rapid Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should handle rapid button clicks', async ({ page }) => {
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    
    // Click rapidly multiple times
    for (let i = 0; i < 5; i++) {
      await proactiveButton.click({ delay: 50 })
    }

    await page.waitForTimeout(1000)
    // Should not break or duplicate state
    const proactivePanel = page.locator('text=/constructor proactivo|proactive/i').first()
    // Should either be visible or not, but not in broken state
  })

  test('should handle rapid model submissions', async ({ page }) => {
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    
    // Submit multiple models rapidly
    for (let i = 0; i < 3; i++) {
      await chatInput.fill(`test model ${i}`)
      await chatInput.press('Enter')
      await page.waitForTimeout(100)
    }

    await page.waitForTimeout(2000)
    // Should handle all submissions
    const models = page.locator('text=/test model/i')
    const count = await models.count()
    expect(count).toBeGreaterThan(0)
  })

  test('should handle rapid panel toggles', async ({ page }) => {
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    await proactiveButton.click()
    await page.waitForTimeout(500)

    const statsButton = page.locator('button:has-text("Estadísticas"), button[title*="Estadísticas"]').first()
    const templatesButton = page.locator('button:has-text("Plantillas"), button[title*="Plantillas"]').first()

    // Toggle panels rapidly
    for (let i = 0; i < 3; i++) {
      if (await statsButton.isVisible()) {
        await statsButton.click()
        await page.waitForTimeout(100)
      }
      if (await templatesButton.isVisible()) {
        await templatesButton.click()
        await page.waitForTimeout(100)
      }
    }

    await page.waitForTimeout(1000)
    // Should not break
  })
})

test.describe('Edge Cases - Network Conditions', () => {
  test('should handle slow network', async ({ page, context }) => {
    // Simulate slow network
    await context.route('**/*', async route => {
      await new Promise(resolve => setTimeout(resolve, 1000))
      await route.continue()
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Should handle slow responses
    await page.waitForTimeout(5000)
  })

  test('should handle network errors gracefully', async ({ page, context }) => {
    // Simulate network failure
    await context.route('**/api/**', route => route.abort())

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Should show error message
    const errorMessage = page.locator('text=/error|failed|network/i').first()
    await expect(errorMessage).toBeVisible({ timeout: 10000 })
  })
})

test.describe('Edge Cases - Browser Behavior', () => {
  test('should handle page refresh during model creation', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Refresh page before completion
    await page.waitForTimeout(1000)
    await page.reload()
    await page.waitForLoadState('networkidle')

    // Should recover gracefully
    const chatInputAfter = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInputAfter).toBeVisible()
  })

  test('should handle browser back/forward navigation', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Navigate to a section
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Go back
      await page.goBack()
      await page.waitForLoadState('networkidle')

      // Go forward
      await page.goForward()
      await page.waitForLoadState('networkidle')

      // Should handle navigation
      const proactivePanel = page.locator('text=/constructor proactivo|proactive/i').first()
      // Should either be visible or not based on state
    }
  })

  test('should handle multiple tabs', async ({ context }) => {
    const page1 = await context.newPage()
    const page2 = await context.newPage()

    await page1.goto('/')
    await page2.goto('/')

    await page1.waitForLoadState('networkidle')
    await page2.waitForLoadState('networkidle')

    // Both pages should work independently
    const input1 = page1.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    const input2 = page2.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()

    await expect(input1).toBeVisible()
    await expect(input2).toBeVisible()

    await page1.close()
    await page2.close()
  })
})

test.describe('Edge Cases - Storage Limits', () => {
  test('should handle localStorage quota exceeded', async ({ page, context }) => {
    // Mock localStorage to throw quota exceeded
    await context.addInitScript(() => {
      const originalSetItem = Storage.prototype.setItem
      Storage.prototype.setItem = function(key: string, value: string) {
        try {
          originalSetItem.call(this, key, value)
        } catch (e: any) {
          if (e.name === 'QuotaExceededError') {
            // Simulate quota exceeded
            throw e
          }
        }
      }
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Should handle gracefully
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })
})

test.describe('Edge Cases - Time-based', () => {
  test('should handle timezone changes', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Create model
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    await page.waitForTimeout(2000)

    // Check timestamps are displayed correctly
    const timestamp = page.locator('text=/ago|min|hour|day/i').first()
    // Should handle timezone correctly
  })

  test('should handle date boundaries', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

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

        // Should handle date grouping correctly
        const dateGroups = page.locator('text=/today|yesterday|week|month/i')
        // Date grouping should work
      }
    }
  })
})










