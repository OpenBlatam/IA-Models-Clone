import { Page, expect } from '@playwright/test';

export const TEST_IDS = {
  HEADER: {
    LOGO: 'header-logo',
    NAV_LINK: (label: string) => `nav-link-${label.toLowerCase()}`,
    LOGIN_BUTTON: 'header-login-button',
    REGISTER_BUTTON: 'header-register-button',
    USER_MENU: 'header-user-menu',
    LOGOUT_BUTTON: 'header-logout-button',
    THEME_TOGGLE: 'header-theme-toggle',
    MOBILE_MENU_BUTTON: 'header-mobile-menu-button',
  },
  AUTH: {
    LOGIN_MODAL: 'login-modal',
    REGISTER_MODAL: 'register-modal',
    EMAIL_INPUT: 'auth-email-input',
    PASSWORD_INPUT: 'auth-password-input',
    NAME_INPUT: 'auth-name-input',
    SUBMIT_BUTTON: 'auth-submit-button',
    SWITCH_TO_REGISTER: 'auth-switch-to-register',
    SWITCH_TO_LOGIN: 'auth-switch-to-login',
  },
  UPLOAD: {
    DROPZONE: 'image-upload-dropzone',
    FILE_INPUT: 'image-upload-input',
    PREVIEW: 'image-upload-preview',
    REMOVE_BUTTON: 'image-upload-remove',
  },
  ANALYSIS: {
    ANALYZE_BUTTON: 'analyze-button',
    RECOMMENDATIONS_BUTTON: 'recommendations-button',
    RESULTS_CONTAINER: 'analysis-results',
    RECOMMENDATIONS_CONTAINER: 'recommendations-container',
    TAB_ANALYSIS: 'tab-analysis',
    TAB_RECOMMENDATIONS: 'tab-recommendations',
  },
  DASHBOARD: {
    STATS_CARD: (index: number) => `stats-card-${index}`,
    PROGRESS_CHART: 'progress-chart',
    RECENT_ANALYSES: 'recent-analyses',
  },
} as const;

export async function waitForPageLoad(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle');
  await page.waitForLoadState('domcontentloaded');
}

export async function createTestImageFile(name: string = 'test-image.jpg'): Promise<File> {
  const canvas = document.createElement('canvas');
  canvas.width = 100;
  canvas.height = 100;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#FF0000';
    ctx.fillRect(0, 0, 100, 100);
  }
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], name, { type: 'image/jpeg' });
        resolve(file);
      }
    }, 'image/jpeg');
  });
}

export async function mockApiResponse(
  page: Page,
  url: string | RegExp,
  response: any,
  status: number = 200
): Promise<void> {
  await page.route(url, (route) => {
    route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(response),
    });
  });
}

export async function fillLoginForm(
  page: Page,
  email: string,
  password: string
): Promise<void> {
  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Contraseña').fill(password);
}

export async function fillRegisterForm(
  page: Page,
  name: string,
  email: string,
  password: string
): Promise<void> {
  await page.getByLabel(/nombre/i).fill(name);
  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Contraseña').fill(password);
}

export async function expectToastMessage(page: Page, message: string): Promise<void> {
  await expect(page.getByText(message)).toBeVisible({ timeout: 5000 });
}

export async function waitForNavigation(page: Page, url: string): Promise<void> {
  await page.waitForURL(url, { timeout: 10000 });
}



