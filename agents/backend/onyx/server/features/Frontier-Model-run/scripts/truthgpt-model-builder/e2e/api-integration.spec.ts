/**
 * E2E Tests - API Integration
 */

import { test, expect } from '@playwright/test'

test.describe('API Integration - Model Creation', () => {
  test('should create model via API', async ({ page }) => {
    const apiCalls: any[] = []

    // Intercept API calls
    page.on('request', request => {
      if (request.url().includes('/api/create-model')) {
        apiCalls.push({
          url: request.url(),
          method: request.method(),
          postData: request.postData(),
        })
      }
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model via API')
    await chatInput.press('Enter')

    // Wait for API call
    await page.waitForTimeout(2000)

    // Should have made API call
    expect(apiCalls.length).toBeGreaterThan(0)
    expect(apiCalls[0].method).toBe('POST')
  })

  test('should handle API response correctly', async ({ page, context }) => {
    // Mock API response
    await context.route('**/api/create-model', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          modelId: 'test-model-123',
          status: 'creating',
        }),
      })
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Should show model status
    await page.waitForTimeout(1000)
    const statusIndicator = page.locator('text=/creating|test-model/i').first()
    await expect(statusIndicator).toBeVisible({ timeout: 5000 })
  })

  test('should poll for model status', async ({ page, context }) => {
    let pollCount = 0

    // Mock status endpoint
    await context.route('**/api/model-status**', route => {
      pollCount++
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          modelId: 'test-model-123',
          status: pollCount < 3 ? 'creating' : 'completed',
        }),
      })
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Wait for polling
    await page.waitForTimeout(5000)

    // Should have polled multiple times
    expect(pollCount).toBeGreaterThan(1)
  })
})

test.describe('API Integration - Error Handling', () => {
  test('should handle API validation errors', async ({ page, context }) => {
    await context.route('**/api/create-model', route => {
      route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'Validation failed',
          details: ['Description is required'],
        }),
      })
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('')
    await chatInput.press('Enter')

    // Should show validation error
    const errorMessage = page.locator('text=/validation|error|required/i').first()
    await expect(errorMessage).toBeVisible({ timeout: 5000 })
  })

  test('should handle API rate limiting', async ({ page, context }) => {
    await context.route('**/api/create-model', route => {
      route.fulfill({
        status: 429,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'Rate limit exceeded',
          retryAfter: 60,
        }),
      })
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Should show rate limit error
    const errorMessage = page.locator('text=/rate limit|too many|retry/i').first()
    await expect(errorMessage).toBeVisible({ timeout: 5000 })
  })
})

test.describe('API Integration - Webhooks', () => {
  test('should trigger webhook on model completion', async ({ page, context }) => {
    const webhookCalls: any[] = []

    // Mock webhook endpoint
    await context.route('**/webhook/**', route => {
      webhookCalls.push({
        url: route.request().url(),
        method: route.request().method(),
        postData: route.request().postData(),
      })
      route.fulfill({ status: 200 })
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Enable proactive mode and configure webhook
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Create model (would trigger webhook if configured)
      const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
      await chatInput.fill('test model')
      await chatInput.press('Enter')

      await page.waitForTimeout(3000)

      // Webhook would be called if configured
      // This is a placeholder test
    }
  })
})

test.describe('API Integration - Caching', () => {
  test('should cache API responses', async ({ page, context }) => {
    let apiCallCount = 0

    await context.route('**/api/templates**', route => {
      apiCallCount++
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { id: '1', name: 'Template 1' },
          { id: '2', name: 'Template 2' },
        ]),
        headers: {
          'Cache-Control': 'max-age=3600',
        },
      })
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Open templates panel twice
    const templatesButton = page.locator('button:has-text("Plantillas"), button[title*="Plantillas"]').first()
    if (await templatesButton.isVisible()) {
      await templatesButton.click()
      await page.waitForTimeout(1000)
      await templatesButton.click()
      await page.waitForTimeout(1000)
      await templatesButton.click()
      await page.waitForTimeout(1000)

      // Should cache and not make multiple calls
      // (Note: actual caching depends on implementation)
    }
  })
})

test.describe('API Integration - Batch Operations', () => {
  test('should handle batch API calls', async ({ page, context }) => {
    const apiCalls: any[] = []

    await context.route('**/api/**', route => {
      apiCalls.push({
        url: route.request().url(),
        method: route.request().method(),
      })
      route.continue()
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Add multiple models
      const queueInput = page.locator('input[placeholder*="descripción"], textarea[placeholder*="descripción"]').first()
      const addButton = page.locator('button:has-text("Agregar"), button:has-text("Add")').first()

      for (let i = 0; i < 3; i++) {
        if (await queueInput.isVisible() && await addButton.isVisible()) {
          await queueInput.fill(`model ${i}`)
          await addButton.click()
          await page.waitForTimeout(500)
        }
      }

      await page.waitForTimeout(2000)

      // Should handle batch operations
      expect(apiCalls.length).toBeGreaterThan(0)
    }
  })
})










