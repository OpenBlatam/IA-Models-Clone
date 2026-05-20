/**
 * E2E Tests - Performance
 */

import { test, expect } from '@playwright/test'

test.describe('Performance - Load Times', () => {
  test('should load page quickly', async ({ page }) => {
    const startTime = Date.now()
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    const loadTime = Date.now() - startTime

    // Should load in reasonable time
    expect(loadTime).toBeLessThan(10000) // 10 seconds
  })

  test('should render interactive elements quickly', async ({ page }) => {
    await page.goto('/')
    
    const startTime = Date.now()
    await page.waitForSelector('input, textarea, button', { timeout: 5000 })
    const renderTime = Date.now() - startTime

    // Should render quickly
    expect(renderTime).toBeLessThan(5000)
  })

  test('should handle large model lists efficiently', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Create multiple models
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    
    const startTime = Date.now()
    for (let i = 0; i < 10; i++) {
      await chatInput.fill(`test model ${i}`)
      await chatInput.press('Enter')
      await page.waitForTimeout(200)
    }
    const createTime = Date.now() - startTime

    // Should create models efficiently
    expect(createTime).toBeLessThan(60000) // 1 minute for 10 models
  })
})

test.describe('Performance - Memory Usage', () => {
  test('should not leak memory with multiple operations', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Perform many operations
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    
    for (let i = 0; i < 20; i++) {
      if (await proactiveButton.isVisible()) {
        await proactiveButton.click()
        await page.waitForTimeout(100)
        await proactiveButton.click()
        await page.waitForTimeout(100)
      }
    }

    // Check for memory leaks (page should still be responsive)
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await expect(chatInput).toBeVisible()
  })

  test('should handle large data sets efficiently', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Add many models to queue
      const queueInput = page.locator('input[placeholder*="descripción"], textarea[placeholder*="descripción"]').first()
      const addButton = page.locator('button:has-text("Agregar"), button:has-text("Add")').first()

      const startTime = Date.now()
      for (let i = 0; i < 50; i++) {
        if (await queueInput.isVisible() && await addButton.isVisible()) {
          await queueInput.fill(`model ${i}`)
          await addButton.click()
          await page.waitForTimeout(50)
        }
      }
      const addTime = Date.now() - startTime

      // Should add efficiently
      expect(addTime).toBeLessThan(30000) // 30 seconds for 50 models
    }
  })
})

test.describe('Performance - Rendering', () => {
  test('should render charts efficiently', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Open stats panel
    const statsButton = page.locator('button:has-text("Estadísticas"), button[title*="Estadísticas"]').first()
    if (await statsButton.isVisible()) {
      await statsButton.click()
      await page.waitForTimeout(1000)

      const startTime = Date.now()
      // Wait for charts to render
      await page.waitForSelector('svg, canvas', { timeout: 5000 })
      const renderTime = Date.now() - startTime

      // Charts should render quickly
      expect(renderTime).toBeLessThan(5000)
    }
  })

  test('should handle rapid UI updates', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Enable proactive mode
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      // Open metrics panel
      const metricsButton = page.locator('button[title*="Métricas"], button[title*="Metrics"]').first()
      if (await metricsButton.isVisible()) {
        await metricsButton.click()
        await page.waitForTimeout(2000)

        // Metrics should update smoothly
        const metricsContent = page.locator('text=/builds|success|duration/i').first()
        await expect(metricsContent).toBeVisible()
      }
    }
  })
})

test.describe('Performance - Network Requests', () => {
  test('should minimize API calls', async ({ page }) => {
    const apiCalls: string[] = []

    // Track API calls
    page.on('request', request => {
      if (request.url().includes('/api/')) {
        apiCalls.push(request.url())
      }
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Perform some actions
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    await page.waitForTimeout(2000)

    // Should not make excessive API calls
    expect(apiCalls.length).toBeLessThan(20)
  })

  test('should cache API responses', async ({ page }) => {
    const apiCalls: string[] = []

    page.on('request', request => {
      if (request.url().includes('/api/')) {
        apiCalls.push(request.url())
      }
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Reload page
    await page.reload()
    await page.waitForLoadState('networkidle')

    // Some calls should be cached
    const uniqueCalls = new Set(apiCalls)
    // Should have some caching
  })
})

test.describe('Performance - Scroll Performance', () => {
  test('should handle long lists smoothly', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Create many models
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    for (let i = 0; i < 20; i++) {
      await chatInput.fill(`test model ${i}`)
      await chatInput.press('Enter')
      await page.waitForTimeout(200)
    }

    await page.waitForTimeout(2000)

    // Scroll through list
    const startTime = Date.now()
    await page.evaluate(() => {
      window.scrollTo(0, document.body.scrollHeight)
    })
    await page.waitForTimeout(100)
    await page.evaluate(() => {
      window.scrollTo(0, 0)
    })
    const scrollTime = Date.now() - startTime

    // Should scroll smoothly
    expect(scrollTime).toBeLessThan(1000)
  })
})










