/**
 * Custom Playwright Fixtures
 * 
 * Fixtures personalizados para tests E2E que proporcionan
 * funcionalidad común y configuración compartida
 */
import { test as base, Page, APIRequestContext } from '@playwright/test';
import { AgentControlPage } from './page-objects/agent-control-page';
import {
  BASE_URL,
  TIMEOUTS,
  createTaskViaApi,
  processTaskViaApi,
  getTaskFromApi,
  getTasksFromApi,
  type ApiTask,
} from './helpers';

// ============================================================================
// Fixture Types
// ============================================================================

type TestFixtures = {
  agentControlPage: AgentControlPage;
  apiContext: APIRequestContext;
  testTask: ApiTask;
};

// ============================================================================
// Base Fixtures
// ============================================================================

export const test = base.extend<TestFixtures>({
  /**
   * Fixture que proporciona una instancia de AgentControlPage
   * ya inicializada y lista para usar
   */
  agentControlPage: async ({ page }, use) => {
    const agentPage = new AgentControlPage(page);
    await agentPage.goto();
    await use(agentPage);
  },

  /**
   * Fixture que proporciona un contexto de API request
   * configurado con la URL base
   */
  apiContext: async ({ request }, use) => {
    await use(request);
  },

  /**
   * Fixture que crea una tarea de prueba automáticamente
   * y la limpia después del test
   */
  testTask: async ({ apiContext }, use) => {
    // Crear tarea de prueba
    const task = await createTaskViaApi(
      apiContext as any,
      'Test task created by fixture',
      'test/repo'
    );

    // Usar la tarea en el test
    await use(task);

    // Cleanup: La tarea se mantiene para inspección manual
    // En producción, podrías agregar lógica de limpieza aquí
  },
});

export { expect } from '@playwright/test';



