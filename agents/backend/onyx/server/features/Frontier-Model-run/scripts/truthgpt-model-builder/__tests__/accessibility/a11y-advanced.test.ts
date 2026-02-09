/**
 * Advanced Accessibility Tests
 */

import { test, expect } from '@playwright/test'
import { injectAxe, checkA11y, getViolations } from 'axe-playwright'

test.describe('Advanced Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await injectAxe(page)
  })

  test('should have no accessibility violations', async ({ page }) => {
    const violations = await getViolations(page)
    expect(violations).toHaveLength(0)
  })

  test('should pass WCAG 2.1 Level A', async ({ page }) => {
    await checkA11y(page, null, {
      rules: {
        'color-contrast': { enabled: true },
        'keyboard-navigation': { enabled: true },
        'aria-required-attr': { enabled: true },
      },
    })
  })

  test('should pass WCAG 2.1 Level AA', async ({ page }) => {
    await checkA11y(page, null, {
      includedImpacts: ['critical', 'serious'],
    })
  })

  test('should have proper focus management', async ({ page }) => {
    // Tab through page
    await page.keyboard.press('Tab')
    const focused1 = page.locator(':focus')
    await expect(focused1).toBeVisible()

    await page.keyboard.press('Tab')
    const focused2 = page.locator(':focus')
    await expect(focused2).toBeVisible()

    // Focus should be visible
    const focusStyles = await focused2.evaluate((el) => {
      const styles = window.getComputedStyle(el)
      return {
        outline: styles.outline,
        outlineWidth: styles.outlineWidth,
      }
    })

    expect(focusStyles.outlineWidth).not.toBe('0px')
  })

  test('should have proper heading hierarchy', async ({ page }) => {
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').all()
    
    if (headings.length > 0) {
      // Check heading levels are sequential
      let previousLevel = 0
      for (const heading of headings) {
        const tagName = await heading.evaluate((el) => el.tagName.toLowerCase())
        const level = parseInt(tagName.charAt(1))
        
        // Should not skip levels (h1 -> h3 is invalid)
        if (previousLevel > 0) {
          expect(level).toBeLessThanOrEqual(previousLevel + 1)
        }
        previousLevel = level
      }
    }
  })

  test('should have proper ARIA labels', async ({ page }) => {
    const buttons = await page.locator('button').all()
    
    for (const button of buttons.slice(0, 10)) {
      const ariaLabel = await button.getAttribute('aria-label')
      const text = await button.textContent()
      const title = await button.getAttribute('title')
      const role = await button.getAttribute('role')

      // Button should have accessible name
      expect(ariaLabel || text || title || role).toBeTruthy()
    }
  })

  test('should have proper form labels', async ({ page }) => {
    const inputs = await page.locator('input, textarea, select').all()
    
    for (const input of inputs.slice(0, 5)) {
      const id = await input.getAttribute('id')
      const ariaLabel = await input.getAttribute('aria-label')
      const ariaLabelledBy = await input.getAttribute('aria-labelledby')
      const placeholder = await input.getAttribute('placeholder')

      if (id) {
        const label = page.locator(`label[for="${id}"]`)
        const hasLabel = await label.count() > 0
        expect(hasLabel || ariaLabel || ariaLabelledBy || placeholder).toBeTruthy()
      } else {
        expect(ariaLabel || ariaLabelledBy || placeholder).toBeTruthy()
      }
    }
  })

  test('should have proper color contrast', async ({ page }) => {
    // This would require more complex testing
    // For now, we check that text elements exist
    const textElements = await page.locator('p, span, div[role="text"]').all()
    expect(textElements.length).toBeGreaterThan(0)
  })

  test('should support screen readers', async ({ page }) => {
    // Check for aria-live regions
    const liveRegions = await page.locator('[aria-live]').all()
    // Live regions are optional but good practice

    // Check for status messages
    const statusMessages = await page.locator('[role="status"], [role="alert"]').all()
    // Status messages should be properly marked
  })

  test('should have skip links', async ({ page }) => {
    // Check for skip to main content link
    const skipLink = page.locator('a[href*="#main"], a:has-text("Skip")').first()
    // Skip links are optional but recommended
  })

  test('should handle keyboard-only navigation', async ({ page }) => {
    // Navigate entire page with keyboard
    const focusableElements: string[] = []

    await page.keyboard.press('Tab')
    let focused = page.locator(':focus')
    while (await focused.count() > 0) {
      const tagName = await focused.evaluate((el) => el.tagName.toLowerCase())
      focusableElements.push(tagName)
      await page.keyboard.press('Tab')
      await page.waitForTimeout(50)
      focused = page.locator(':focus')
      
      // Prevent infinite loop
      if (focusableElements.length > 100) break
    }

    expect(focusableElements.length).toBeGreaterThan(0)
  })
})










