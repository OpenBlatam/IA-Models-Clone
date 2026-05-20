import { test, expect } from '@playwright/test';
import { waitForPageLoad, fillLoginForm, fillRegisterForm, expectToastMessage } from './helpers/test-helpers';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForPageLoad(page);
  });

  test.describe('Login Modal', () => {
    test('should open login modal when clicking login button', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      
      await expect(page.getByRole('heading', { name: /iniciar sesión/i })).toBeVisible();
      await expect(page.getByLabel('Email')).toBeVisible();
      await expect(page.getByLabel('Contraseña')).toBeVisible();
    });

    test('should close login modal when clicking cancel', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      await expect(page.getByRole('heading', { name: /iniciar sesión/i })).toBeVisible();
      
      await page.getByRole('button', { name: /cancelar/i }).click();
      
      await expect(page.getByRole('heading', { name: /iniciar sesión/i })).not.toBeVisible();
    });

    test('should switch to register modal from login', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      await expect(page.getByRole('heading', { name: /iniciar sesión/i })).toBeVisible();
      
      await page.getByRole('link', { name: /¿no tienes cuenta\? regístrate/i }).click();
      
      await expect(page.getByRole('heading', { name: /registrarse/i })).toBeVisible();
      await expect(page.getByRole('heading', { name: /iniciar sesión/i })).not.toBeVisible();
    });

    test('should validate required fields in login form', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      
      const emailInput = page.getByLabel('Email');
      const passwordInput = page.getByLabel('Contraseña');
      
      await expect(emailInput).toHaveAttribute('required');
      await expect(passwordInput).toHaveAttribute('required');
    });

    test('should display error for invalid login credentials', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      
      await fillLoginForm(page, 'invalid@example.com', 'wrongpassword');
      await page.getByRole('button', { name: /iniciar sesión/i }).last().click();
      
      await expectToastMessage(page, /error/i);
    });

    test('should show loading state during login', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      
      await fillLoginForm(page, 'test@example.com', 'password123');
      
      const submitButton = page.getByRole('button', { name: /iniciar sesión/i }).last();
      await submitButton.click();
      
      await expect(submitButton).toBeDisabled();
    });
  });

  test.describe('Register Modal', () => {
    test('should open register modal when clicking register button', async ({ page }) => {
      await page.getByRole('button', { name: /registrarse/i }).click();
      
      await expect(page.getByRole('heading', { name: /registrarse/i })).toBeVisible();
      await expect(page.getByLabel(/nombre/i)).toBeVisible();
      await expect(page.getByLabel('Email')).toBeVisible();
      await expect(page.getByLabel('Contraseña')).toBeVisible();
    });

    test('should close register modal when clicking cancel', async ({ page }) => {
      await page.getByRole('button', { name: /registrarse/i }).click();
      await expect(page.getByRole('heading', { name: /registrarse/i })).toBeVisible();
      
      await page.getByRole('button', { name: /cancelar/i }).click();
      
      await expect(page.getByRole('heading', { name: /registrarse/i })).not.toBeVisible();
    });

    test('should switch to login modal from register', async ({ page }) => {
      await page.getByRole('button', { name: /registrarse/i }).click();
      await expect(page.getByRole('heading', { name: /registrarse/i })).toBeVisible();
      
      await page.getByRole('link', { name: /¿ya tienes cuenta\? inicia sesión/i }).click();
      
      await expect(page.getByRole('heading', { name: /iniciar sesión/i })).toBeVisible();
      await expect(page.getByRole('heading', { name: /registrarse/i })).not.toBeVisible();
    });

    test('should validate required fields in register form', async ({ page }) => {
      await page.getByRole('button', { name: /registrarse/i }).click();
      
      const nameInput = page.getByLabel(/nombre/i);
      const emailInput = page.getByLabel('Email');
      const passwordInput = page.getByLabel('Contraseña');
      
      await expect(nameInput).toHaveAttribute('required');
      await expect(emailInput).toHaveAttribute('required');
      await expect(passwordInput).toHaveAttribute('required');
    });

    test('should display error for invalid registration', async ({ page }) => {
      await page.getByRole('button', { name: /registrarse/i }).click();
      
      await fillRegisterForm(page, 'Test User', 'invalid-email', 'short');
      await page.getByRole('button', { name: /registrarse/i }).last().click();
      
      await expectToastMessage(page, /error/i);
    });

    test('should show loading state during registration', async ({ page }) => {
      await page.getByRole('button', { name: /registrarse/i }).click();
      
      await fillRegisterForm(page, 'Test User', 'test@example.com', 'password123');
      
      const submitButton = page.getByRole('button', { name: /registrarse/i }).last();
      await submitButton.click();
      
      await expect(submitButton).toBeDisabled();
    });
  });

  test.describe('Authentication Flow', () => {
    test('should successfully register a new user', async ({ page }) => {
      await page.getByRole('button', { name: /registrarse/i }).click();
      
      await fillRegisterForm(page, 'Test User', 'newuser@example.com', 'TestPassword123!');
      await page.getByRole('button', { name: /registrarse/i }).last().click();
      
      await expectToastMessage(page, /éxito/i);
      await expect(page.getByRole('heading', { name: /registrarse/i })).not.toBeVisible();
    });

    test('should successfully login with valid credentials', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      
      await fillLoginForm(page, 'test@example.com', 'TestPassword123!');
      await page.getByRole('button', { name: /iniciar sesión/i }).last().click();
      
      await expectToastMessage(page, /éxito/i);
      await expect(page.getByText(/salir/i)).toBeVisible();
    });

    test('should logout successfully', async ({ page }) => {
      await page.getByRole('button', { name: /iniciar sesión/i }).click();
      await fillLoginForm(page, 'test@example.com', 'TestPassword123!');
      await page.getByRole('button', { name: /iniciar sesión/i }).last().click();
      
      await page.waitForTimeout(2000);
      
      await page.getByRole('button', { name: /salir/i }).click();
      
      await expect(page.getByRole('button', { name: /iniciar sesión/i })).toBeVisible();
    });
  });
});



