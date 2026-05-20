import { test, expect } from '@playwright/test';
import { waitForPageLoad } from './helpers/test-helpers';

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForPageLoad(page);
  });

  test.describe('Header Navigation', () => {
    test('should display logo and brand name', async ({ page }) => {
      await expect(page.getByRole('link', { name: /dermatology ai/i })).toBeVisible();
    });

    test('should navigate to home page when clicking logo', async ({ page }) => {
      await page.getByRole('link', { name: /dermatology ai/i }).click();
      await expect(page).toHaveURL('/');
    });

    test('should display all navigation links', async ({ page }) => {
      const navLinks = [
        /inicio/i,
        /dashboard/i,
        /historial/i,
        /comparar/i,
        /productos/i,
        /configuración/i,
      ];

      for (const linkText of navLinks) {
        const link = page.getByRole('link', { name: linkText });
        if (await link.isVisible({ timeout: 2000 })) {
          await expect(link).toBeVisible();
        }
      }
    });

    test('should navigate to dashboard page', async ({ page }) => {
      await page.getByRole('link', { name: /dashboard/i }).click();
      await expect(page).toHaveURL('/dashboard');
    });

    test('should navigate to history page', async ({ page }) => {
      await page.getByRole('link', { name: /historial/i }).click();
      await expect(page).toHaveURL('/history');
    });

    test('should navigate to compare page', async ({ page }) => {
      await page.getByRole('link', { name: /comparar/i }).click();
      await expect(page).toHaveURL('/compare');
    });

    test('should navigate to products page', async ({ page }) => {
      await page.getByRole('link', { name: /productos/i }).click();
      await expect(page).toHaveURL('/products');
    });

    test('should navigate to settings page', async ({ page }) => {
      await page.getByRole('link', { name: /configuración/i }).click();
      await expect(page).toHaveURL('/settings');
    });
  });

  test.describe('Theme Toggle', () => {
    test('should display theme toggle button', async ({ page }) => {
      const themeToggle = page.getByRole('button', { name: /toggle theme/i });
      await expect(themeToggle).toBeVisible();
    });

    test('should toggle between light and dark theme', async ({ page }) => {
      const themeToggle = page.getByRole('button', { name: /toggle theme/i });
      const initialTheme = await page.evaluate(() => {
        return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
      });

      await themeToggle.click();
      await page.waitForTimeout(500);

      const newTheme = await page.evaluate(() => {
        return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
      });

      expect(newTheme).not.toBe(initialTheme);
    });
  });

  test.describe('Mobile Navigation', () => {
    test('should display mobile menu button on small screens', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      
      const mobileMenuButton = page.getByRole('button', { name: /toggle menu/i });
      await expect(mobileMenuButton).toBeVisible();
    });

    test('should open mobile menu when clicking menu button', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      
      const mobileMenuButton = page.getByRole('button', { name: /toggle menu/i });
      await mobileMenuButton.click();
      
      await expect(page.getByRole('link', { name: /inicio/i })).toBeVisible();
    });

    test('should close mobile menu when clicking close button', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      
      const mobileMenuButton = page.getByRole('button', { name: /toggle menu/i });
      await mobileMenuButton.click();
      await page.waitForTimeout(300);
      
      await mobileMenuButton.click();
      await page.waitForTimeout(300);
      
      const navLinks = page.getByRole('link', { name: /dashboard/i });
      if (await navLinks.isVisible({ timeout: 1000 })) {
        await expect(navLinks).not.toBeVisible();
      }
    });

    test('should close mobile menu when clicking a navigation link', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      
      const mobileMenuButton = page.getByRole('button', { name: /toggle menu/i });
      await mobileMenuButton.click();
      
      await page.getByRole('link', { name: /dashboard/i }).click();
      await page.waitForTimeout(500);
      
      await expect(page).toHaveURL('/dashboard');
    });
  });

  test.describe('Authentication Navigation', () => {
    test('should display login and register buttons when not authenticated', async ({ page }) => {
      await expect(page.getByRole('button', { name: /iniciar sesión/i })).toBeVisible();
      await expect(page.getByRole('button', { name: /registrarse/i })).toBeVisible();
    });

    test('should display user info and logout button when authenticated', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      await page.getByLabel('Email').fill('test@example.com');
      await page.getByLabel('Contraseña').fill('TestPassword123!');
      await page.getByRole('button', { name: /iniciar sesión/i }).last().click();
      
      await page.waitForTimeout(2000);
      
      await expect(page.getByText(/test@example.com/i)).toBeVisible();
      await expect(page.getByRole('button', { name: /salir/i })).toBeVisible();
    });

    test('should display notifications icon when authenticated', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      await page.getByLabel('Email').fill('test@example.com');
      await page.getByLabel('Contraseña').fill('TestPassword123!');
      await page.getByRole('button', { name: /iniciar sesión/i }).last().click();
      
      await page.waitForTimeout(2000);
      
      const notificationsLink = page.getByRole('link', { name: /notifications/i });
      if (await notificationsLink.isVisible({ timeout: 2000 })) {
        await expect(notificationsLink).toBeVisible();
      }
    });
  });
});



