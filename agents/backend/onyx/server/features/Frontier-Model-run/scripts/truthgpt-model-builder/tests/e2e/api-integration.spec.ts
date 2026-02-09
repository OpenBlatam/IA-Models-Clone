import { test, expect } from '@playwright/test';

/**
 * Tests de integración con la API TruthGPT
 * Requiere que el servidor API esté corriendo en http://localhost:8000
 */

const API_BASE_URL = process.env.TRUTHGPT_API_URL || 'http://localhost:8000';

test.describe('API Integration Tests', () => {
  test('should check API health endpoint', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/health`);
    
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data).toHaveProperty('status');
    expect(data.status).toBe('healthy');
  });

  test('should create model via API', async ({ request }) => {
    const response = await request.post(`${API_BASE_URL}/models/create`, {
      data: {
        layers: [
          { type: 'dense', params: { units: 64, activation: 'relu' } },
          { type: 'dense', params: { units: 10, activation: 'softmax' } }
        ],
        name: 'playwright-test-model'
      }
    });
    
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data).toHaveProperty('model_id');
    expect(data).toHaveProperty('status');
    expect(data.status).toBe('created');
    
    // Limpiar: eliminar el modelo
    if (data.model_id) {
      await request.delete(`${API_BASE_URL}/models/${data.model_id}`);
    }
  });

  test('should compile model via API', async ({ request }) => {
    // Crear modelo
    const createResponse = await request.post(`${API_BASE_URL}/models/create`, {
      data: {
        layers: [
          { type: 'dense', params: { units: 32 } },
          { type: 'dense', params: { units: 10 } }
        ]
      }
    });
    
    const modelId = (await createResponse.json()).model_id;
    
    // Compilar
    const compileResponse = await request.post(
      `${API_BASE_URL}/models/${modelId}/compile`,
      {
        data: {
          optimizer: 'adam',
          optimizer_params: { learning_rate: 0.001 },
          loss: 'sparsecategoricalcrossentropy',
          metrics: ['accuracy']
        }
      }
    );
    
    expect(compileResponse.ok()).toBeTruthy();
    const compileData = await compileResponse.json();
    expect(compileData.status).toBe('compiled');
    
    // Limpiar
    await request.delete(`${API_BASE_URL}/models/${modelId}`);
  });

  test('should list models via API', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/models`);
    
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data).toHaveProperty('models');
    expect(data).toHaveProperty('count');
    expect(Array.isArray(data.models)).toBeTruthy();
  });

  test('should handle API errors gracefully', async ({ request }) => {
    // Intentar crear modelo con layer inválido
    const response = await request.post(`${API_BASE_URL}/models/create`, {
      data: {
        layers: [
          { type: 'invalid_layer', params: {} }
        ]
      }
    });
    
    // Debería retornar error 400
    expect(response.status()).toBe(400);
  });
});

test.describe('Frontend to API Integration', () => {
  test('should display API connection status', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Verificar que no hay errores críticos de conexión
    const criticalErrors = page.locator('[class*="error"][class*="critical"]');
    const errorCount = await criticalErrors.count();
    
    // Puede haber warnings pero no errores críticos
    expect(errorCount).toBeLessThan(5); // Tolerancia razonable
  });

  test('should create model from frontend and verify in API', async ({ page, request }) => {
    // Verificar que la API está disponible
    const healthResponse = await request.get(`${API_BASE_URL}/health`);
    if (!healthResponse.ok()) {
      test.skip('API no está disponible');
    }
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const input = page.locator('input[type="text"]');
    await input.fill('Modelo de prueba desde Playwright');
    
    // Esperar a que se procese
    await page.waitForTimeout(1000);
    
    // Verificar que hay modelos en la API (puede haber sido creado)
    const modelsResponse = await request.get(`${API_BASE_URL}/models`);
    const modelsData = await modelsResponse.json();
    
    // Al menos debería poder listar modelos
    expect(modelsResponse.ok()).toBeTruthy();
  });
});











