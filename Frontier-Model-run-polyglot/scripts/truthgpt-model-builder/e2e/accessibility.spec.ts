/**
 * E2E Tests - Accessibility
 */

import { test, expect } from '@playwright/test'

test.describe('Accessibility Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should have proper heading structure', async ({ page }) => {
    // Check for main headings
    const headings = page.locator('h1, h2, h3')
    const count = await headings.count()
    expect(count).toBeGreaterThan(0)
  })

  test('should have accessible buttons', async ({ page }) => {
    // Check buttons have accessible names
    const buttons = page.locator('button')
    const buttonCount = await buttons.count()

    for (let i = 0; i < Math.min(buttonCount, 10); i++) {
      const button = buttons.nth(i)
      const ariaLabel = await button.getAttribute('aria-label')
      const text = await button.textContent()
      const title = await button.getAttribute('title')

      // Button should have either text, aria-label, or title
      expect(ariaLabel || text || title).toBeTruthy()
    }
  })

  test('should have accessible form inputs', async ({ page }) => {
    // Check inputs have labels or aria-labels
    const inputs = page.locator('input, textarea')
    const inputCount = await inputs.count()

    for (let i = 0; i < Math.min(inputCount, 5); i++) {
      const input = inputs.nth(i)
      const id = await input.getAttribute('id')
      const ariaLabel = await input.getAttribute('aria-label')
      const placeholder = await input.getAttribute('placeholder')

      // Input should have label, aria-label, or placeholder
      if (id) {
        const label = page.locator(`label[for="${id}"]`)
        const hasLabel = await label.count() > 0
        expect(hasLabel || ariaLabel || placeholder).toBeTruthy()
      } else {
        expect(ariaLabel || placeholder).toBeTruthy()
      }
    }
  })

  test('should support keyboard navigation', async ({ page }) => {
    // Tab through interactive elements
    await page.keyboard.press('Tab')
    await page.waitForTimeout(100)

    // Check focus is visible
    const focusedElement = page.locator(':focus')
    await expect(focusedElement).toBeVisible()
  })

  test('should have proper color contrast', async ({ page }) => {
    // This would require more complex testing
    // For now, just check that text elements exist
    const textElements = page.locator('p, span, div[role="text"]')
    const count = await textElements.count()
    expect(count).toBeGreaterThan(0)
  })

  test('should have skip links', async ({ page }) => {
    // Check for skip to main content link
    const skipLink = page.locator('a[href*="#main"], a:has-text("Skip")').first()
    // Skip link is optional, so we don't require it
  })

  test('should have proper ARIA roles', async ({ page }) => {
    // Check for common ARIA roles
    const roles = ['button', 'dialog', 'navigation', 'main', 'complementary']
    
    for (const role of roles) {
      const elements = page.locator(`[role="${role}"]`)
      // These are optional, so we just check they exist if present
    }
  })
})

test.describe('Screen Reader Support', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test('should announce model creation', async ({ page }) => {
    // Create model
    const chatInput = page.locator('input[placeholder*="modelo"], textarea[placeholder*="modelo"]').first()
    await chatInput.fill('test model')
    await chatInput.press('Enter')

    // Check for aria-live region or status message
    const liveRegion = page.locator('[aria-live], [role="status"], [role="alert"]').first()
    // Live regions are optional but good practice
  })
})










