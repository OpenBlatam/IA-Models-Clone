import { test as base, expect } from '@playwright/test';

type AuthFixtures = {
  authenticatedPage: any;
};

export const test = base.extend<AuthFixtures>({
  authenticatedPage: async ({ page }, use) => {
    await page.goto('/');
    
    const testEmail = 'test@example.com';
    const testPassword = 'TestPassword123!';
    const testName = 'Test User';

    await page.getByRole('button', { name: /registrarse/i }).click();
    await page.waitForSelector('text=Registrarse', { state: 'visible' });

    await page.getByLabel(/nombre/i).fill(testName);
    await page.getByLabel('Email').fill(testEmail);
    await page.getByLabel('Contraseña').fill(testPassword);
    await page.getByRole('button', { name: /registrarse/i }).click();

    await page.waitForTimeout(2000);

    await use(page);
  },
});

export { expect };

