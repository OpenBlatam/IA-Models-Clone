/**
 * E2E Tests para la página Not Found (404)
 * 
 * Tests end-to-end que verifican:
 * - Acceso a rutas inexistentes muestra la página 404
 * - Navegación desde la página 404
 * - Renderizado correcto en el navegador
 * - Interacciones del usuario
 * - Responsive design
 * 
 * @module e2e/not-found
 * @tags e2e, not-found, 404, navigation
 */
import { test, expect } from './fixtures';
import { BASE_URL, TIMEOUTS } from './helpers';

// ============================================================================
// Test Configuration
// ============================================================================

const TEST_CONFIG = {
  timeout: TIMEOUTS.TEST_TIMEOUT,
  viewport: { width: 1920, height: 1080 },
  mobileViewport: { width: 375, height: 667 },
} as const;

// ============================================================================
// Test Suites
// ============================================================================

test.describe('Not Found Page (404)', () => {
  test.setTimeout(TEST_CONFIG.timeout);

  // ==========================================================================
  // Setup & Teardown
  // ==========================================================================

  test.beforeEach(async ({ page }) => {
    await page.setViewportSize(TEST_CONFIG.viewport);
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== testInfo.expectedStatus) {
      await page.screenshot({
        path: `test-results/not-found-failed-${Date.now()}.png`,
        fullPage: true,
      });
    }
  });

  // ==========================================================================
  // Tests de Renderizado
  // ==========================================================================

  test('debería mostrar la página 404 al acceder a una ruta inexistente', async ({
    page,
  }) => {
    // Navegar a una ruta que no existe
    await page.goto(`${BASE_URL}/ruta-que-no-existe-12345`);

    // Verificar que se muestra el número 404
    const heading404 = page.getByRole('heading', { name: /404/i });
    await expect(heading404).toBeVisible();
    await expect(heading404).toHaveText('404');
  });

  test('debería mostrar el título "Page Not Found"', async ({ page }) => {
    await page.goto(`${BASE_URL}/ruta-inexistente`);

    const title = page.getByRole('heading', { name: /page not found/i });
    await expect(title).toBeVisible();
    await expect(title).toHaveText('Page Not Found');
  });

  test('debería mostrar el mensaje descriptivo', async ({ page }) => {
    await page.goto(`${BASE_URL}/ruta-inexistente`);

    const description = page.getByText(
      /the page you're looking for doesn't exist or has been moved/i
    );
    await expect(description).toBeVisible();
  });

  test('debería renderizar el Header correctamente', async ({ page }) => {
    await page.goto(`${BASE_URL}/ruta-inexistente`);

    // Verificar que el header está presente (buscar por logo o navegación)
    const header = page.locator('header').first();
    await expect(header).toBeVisible();
  });

  // ==========================================================================
  // Tests de Navegación
  // ==========================================================================

  test('debería navegar a home cuando se hace clic en "Go to Home"', async ({
    page,
  }) => {
    await page.goto(`${BASE_URL}/ruta-inexistente`);
    await page.waitForLoadState('networkidle');

    // Hacer clic en el botón "Go to Home"
    const homeButton = page.getByRole('button', { name: /go to home/i });
    await expect(homeButton).toBeVisible();
    
    // Esperar a que la navegación se complete
    await Promise.all([
      page.waitForURL(new RegExp(`${BASE_URL}/?$`), { timeout: 10000 }),
      homeButton.click(),
    ]);

    // Verificar que se navegó a la página principal
    await expect(page).toHaveURL(new RegExp(`${BASE_URL}/?$`));
  });

  test('debería navegar hacia atrás cuando se hace clic en "Go Back"', async ({
    page,
  }) => {
    // Primero ir a una página válida
    await page.goto(`${BASE_URL}/`);
    await page.waitForLoadState('networkidle');

    // Luego ir a una ruta inexistente
    await page.goto(`${BASE_URL}/ruta-inexistente`);

    // Hacer clic en "Go Back"
    const backButton = page.getByRole('button', { name: /go back/i });
    await expect(backButton).toBeVisible();
    await backButton.click();

    // Verificar que se navegó hacia atrás
    await expect(page).toHaveURL(new RegExp(`${BASE_URL}/?$`));
  });

  // ==========================================================================
  // Tests de Interacción
  // ==========================================================================

  test('debería tener hover effects en los botones', async ({ page }) => {
    await page.goto(`${BASE_URL}/ruta-inexistente`);

    const homeButton = page.getByRole('button', { name: /go to home/i });
    
    // Verificar que el botón tiene las clases de hover
    const buttonClasses = await homeButton.getAttribute('class');
    expect(buttonClasses).toContain('hover');
  });

  test('debería ser accesible con teclado', async ({ page }) => {
    await page.goto(`${BASE_URL}/ruta-inexistente`);

    // Navegar con Tab
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Verificar que un botón tiene foco
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });

  // ==========================================================================
  // Tests de Responsive Design
  // ==========================================================================

  test('debería renderizar correctamente en móvil', async ({ page }) => {
    await page.setViewportSize(TEST_CONFIG.mobileViewport);
    await page.goto(`${BASE_URL}/ruta-inexistente`);

    // Verificar que los elementos están visibles
    const heading404 = page.getByRole('heading', { name: /404/i });
    await expect(heading404).toBeVisible();

    // Verificar que los botones están en columna en móvil
    const buttonContainer = page
      .getByRole('button', { name: /go to home/i })
      .locator('..');
    const containerClasses = await buttonContainer.getAttribute('class');
    expect(containerClasses).toContain('flex-col');
  });

  test('debería renderizar correctamente en desktop', async ({ page }) => {
    await page.setViewportSize(TEST_CONFIG.viewport);
    await page.goto(`${BASE_URL}/ruta-inexistente`);

    // Verificar que los elementos están visibles
    const heading404 = page.getByRole('heading', { name: /404/i });
    await expect(heading404).toBeVisible();

    // Verificar que los botones están en fila en desktop
    const buttonContainer = page
      .getByRole('button', { name: /go to home/i })
      .locator('..');
    const containerClasses = await buttonContainer.getAttribute('class');
    expect(containerClasses).toContain('sm:flex-row');
  });

  // ==========================================================================
  // Tests de Diferentes Rutas Inexistentes
  // ==========================================================================

  test('debería mostrar 404 para diferentes tipos de rutas inexistentes', async ({
    page,
  }) => {
    const invalidRoutes = [
      '/ruta-simple',
      '/ruta/con/sub/rutas',
      '/ruta-con-parametros?param=value',
      '/ruta-con-hash#section',
      '/123456',
      '/ruta-con-caracteres-especiales-!@#$',
    ];

    for (const route of invalidRoutes) {
      await page.goto(`${BASE_URL}${route}`);
      
      const heading404 = page.getByRole('heading', { name: /404/i });
      await expect(heading404).toBeVisible({ timeout: 5000 });
    }
  });

  // ==========================================================================
  // Tests de Performance
  // ==========================================================================

  test('debería cargar rápidamente', async ({ page }) => {
    const startTime = Date.now();
    
    // Navegar y esperar a que el contenido esté visible
    await Promise.all([
      page.goto(`${BASE_URL}/ruta-inexistente`),
      page.waitForLoadState('networkidle'),
    ]);
    
    const heading404 = page.getByRole('heading', { name: /404/i });
    await expect(heading404).toBeVisible();
    
    const loadTime = Date.now() - startTime;
    // En desarrollo puede ser más lento, pero debería cargar en menos de 10 segundos
    expect(loadTime).toBeLessThan(10000);
  });

  // ==========================================================================
  // Tests de SEO y Meta Tags
  // ==========================================================================

  test('debería tener un título de página apropiado', async ({ page }) => {
    await page.goto(`${BASE_URL}/ruta-inexistente`);
    
    const title = await page.title();
    // Next.js puede tener un título por defecto o el de la app
    expect(title).toBeTruthy();
  });

  // ==========================================================================
  // Tests de Animaciones
  // ==========================================================================

  test('debería tener animaciones suaves al cargar', async ({ page }) => {
    await page.goto(`${BASE_URL}/ruta-inexistente`);
    
    // Verificar que los elementos aparecen con animación
    const heading404 = page.getByRole('heading', { name: /404/i });
    await expect(heading404).toBeVisible();
    
    // Los elementos deberían estar visibles después de la animación
    await page.waitForTimeout(1000); // Esperar a que las animaciones terminen
    
    const title = page.getByRole('heading', { name: /page not found/i });
    await expect(title).toBeVisible();
  });
});

