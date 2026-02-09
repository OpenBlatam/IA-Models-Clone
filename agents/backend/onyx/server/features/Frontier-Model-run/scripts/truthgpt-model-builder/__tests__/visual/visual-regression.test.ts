/**
 * Visual Regression Tests
 */

import { test, expect } from '@playwright/test'

test.describe('Visual Regression', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should match main page snapshot', async ({ page }) => {
    await expect(page).toHaveScreenshot('main-page.png', {
      fullPage: true,
      maxDiffPixels: 100,
    })
  })

  test('should match proactive builder panel snapshot', async ({ page }) => {
    const proactiveButton = page.locator('button:has-text("Proactivo"), button[title*="Proactivo"]').first()
    if (await proactiveButton.isVisible()) {
      await proactiveButton.click()
      await page.waitForTimeout(500)

      const panel = page.locator('text=/constructor proactivo|proactive/i').first()
      if (await panel.isVisible()) {
        await expect(page).toHaveScreenshot('proactive-panel.png', {
          fullPage: false,
          maxDiffPixels: 100,
        })
      }
    }
  })

  test('should match statistics panel snapshot', async ({ page }) => {
    const statsButton = page.locator('button:has-text("Estadísticas"), button[title*="Estadísticas"]').first()
    if (await statsButton.isVisible()) {
      await statsButton.click()
      await page.waitForTimeout(1000)

      await expect(page).toHaveScreenshot('statistics-panel.png', {
        fullPage: false,
        maxDiffPixels: 100,
      })
    }
  })

  test('should match dark theme snapshot', async ({ page }) => {
    // Ensure dark theme
    await page.emulateMedia({ colorScheme: 'dark' })
    await page.waitForTimeout(500)

    await expect(page).toHaveScreenshot('dark-theme.png', {
      fullPage: true,
      maxDiffPixels: 100,
    })
  })

  test('should match light theme snapshot', async ({ page }) => {
    // Ensure light theme
    await page.emulateMedia({ colorScheme: 'light' })
    await page.waitForTimeout(500)

    await expect(page).toHaveScreenshot('light-theme.png', {
      fullPage: true,
      maxDiffPixels: 100,
    })
  })

  test('should match mobile viewport snapshot', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.waitForTimeout(500)

    await expect(page).toHaveScreenshot('mobile-view.png', {
      fullPage: true,
      maxDiffPixels: 100,
    })
  })

  test('should match tablet viewport snapshot', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 })
    await page.waitForTimeout(500)

    await expect(page).toHaveScreenshot('tablet-view.png', {
      fullPage: true,
      maxDiffPixels: 100,
    })
  })
})

test.describe('Component Visual Tests', () => {
  test('should match button states', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const button = page.locator('button').first()
    if (await button.isVisible()) {
      // Default state
      await expect(button).toHaveScreenshot('button-default.png')

      // Hover state
      await button.hover()
      await expect(button).toHaveScreenshot('button-hover.png')

      // Active state
      await button.click({ delay: 100 })
      await expect(button).toHaveScreenshot('button-active.png')
    }
  })

  test('should match input states', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const input = page.locator('input[type="text"], textarea').first()
    if (await input.isVisible()) {
      // Default state
      await expect(input).toHaveScreenshot('input-default.png')

      // Focus state
      await input.focus()
      await expect(input).toHaveScreenshot('input-focus.png')

      // Filled state
      await input.fill('test input')
      await expect(input).toHaveScreenshot('input-filled.png')
    }
  })
})










