/**
 * E2E Test completo del flujo: desde el frontend hasta la ejecución
 * 
 * Refactorizado con:
 * - Sistema de métricas integrado
 * - Reporting automático
 * - Helpers especializados
 * - Mejor separación de responsabilidades
 * - Tests organizados por categorías
 * - Tags para organización
 * - Documentación inline mejorada
 * 
 * @module e2e/complete-flow
 * @tags e2e, complete-flow, critical
 */
import { test, expect } from './fixtures';
import {
  TIMEOUTS,
  navigateToAgentControl,
  createTask,
  waitForTaskToAppear,
} from './helpers';
import { TEST_INSTRUCTIONS } from './constants';
import {
  executeCompleteFlowWithMetrics,
  verifyConsoleLogsWithNetworkAnalysis,
  createMultipleTasks,
  validatePerformanceMetrics,
} from './test-helpers/test-steps';
import {
  generateReport,
  generateTextReport,
  saveReport,
} from './test-helpers/reporting';
import {
  testWithLongInstruction,
  testWithEmptyInstruction,
  testStatePersistence,
  testNetworkErrorHandling,
} from './test-helpers/edge-cases';
import {
  createTasksInParallel,
  validateParallelTaskProcessing,
} from './test-helpers/parallel-execution';
import {
  testInvalidDataHandling,
  testRateLimiting,
  testSlowServerResponse,
} from './test-helpers/error-scenarios';

// ============================================================================
// Test Configuration
// ============================================================================

/**
 * Configuración global de tests
 */
const TEST_CONFIG = {
  timeout: TIMEOUTS.TEST_TIMEOUT,
  viewport: { width: 1920, height: 1080 },
  retries: process.env.CI ? 2 : 0,
} as const;

// ============================================================================
// Test Suites
// ============================================================================

test.describe('Complete Flow E2E', () => {
  test.setTimeout(TEST_CONFIG.timeout);

  // ==========================================================================
  // Setup & Teardown
  // ==========================================================================

  /**
   * Setup antes de cada test
   * - Configura viewport estándar
   * - Prepara el entorno de testing
   */
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize(TEST_CONFIG.viewport);
    // Limpiar localStorage y sessionStorage para tests aislados
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
    });
  });

  /**
   * Teardown después de cada test
   * - Captura screenshots en caso de fallo
   * - Limpia recursos si es necesario
   */
  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== testInfo.expectedStatus) {
      const sanitizedTitle = testInfo.title.replace(/[^a-z0-9]/gi, '-').toLowerCase();
      await page.screenshot({
        path: `test-results/failed-${sanitizedTitle}-${Date.now()}.png`,
        fullPage: true,
      });
    }
  });

  // ==========================================================================
  // Core Flow Tests
  // ==========================================================================

  test.describe('Core Flow', () => {
    /**
     * Test principal del flujo completo
     * 
     * @tags @critical @smoke
     * @description Valida el flujo completo desde la creación de tarea hasta su finalización
     */
    test('debería completar el flujo completo desde el frontend', async ({
      page,
    }) => {
      const { metrics, result } = await executeCompleteFlowWithMetrics(
        page,
        TEST_INSTRUCTIONS.DEFAULT
      );

      // Generar y mostrar reporte
      const report = generateReport(metrics);
      console.log(generateTextReport(report));

      // Guardar reporte HTML para inspección
      const reportPath = await saveReport(report, 'html', 'test-results');
      console.log(`📄 Reporte guardado en: ${reportPath}`);

      // Verificar que el test pasó
      expect(result).not.toBeNull();
      expect(result?.completed || result?.contentReceived).toBe(true);
    });

    /**
     * Test de verificación de logs de consola
     * 
     * @tags @logging @debugging
     * @description Valida que no hay logs problemáticos en la consola del navegador
     */
    test('debería verificar los logs en la consola del navegador', async ({
      page,
    }) => {
      await verifyConsoleLogsWithNetworkAnalysis(page, TEST_INSTRUCTIONS.SIMPLE);
    });

    /**
     * Test usando fixtures personalizados
     * 
     * @tags @fixtures @page-objects
     * @description Demuestra el uso de fixtures y page objects
     */
    test('debería usar el fixture de página de control', async ({
      agentControlPage,
    }) => {
      const isLoaded = await agentControlPage.isLoaded();
      expect(isLoaded).toBe(true);

      await agentControlPage.createTask(TEST_INSTRUCTIONS.DEFAULT);
      await agentControlPage.waitForTask();

      const lastTask = await agentControlPage.getLastTask();
      expect(lastTask).not.toBeNull();
      expect(lastTask?.text).toBeTruthy();
    });
  });

  // ==========================================================================
  // Performance Tests
  // ==========================================================================

  test.describe('Performance', () => {
    /**
     * Test de validación de performance
     * 
     * @tags @performance @metrics
     * @description Valida que las operaciones críticas completan en tiempo razonable
     */
    test('debería validar performance de operaciones críticas', async ({
      page,
    }) => {
      const { metrics, report } = await validatePerformanceMetrics(page, [
        {
          name: 'Navegación',
          action: async () => {
            await navigateToAgentControl(page);
          },
        },
        {
          name: 'Creación de tarea',
          action: async () => {
            await createTask(page, TEST_INSTRUCTIONS.SIMPLE);
          },
        },
      ]);

      // Validaciones de performance
      if (metrics.performance) {
        expect(metrics.performance.pageLoadTime).toBeLessThan(5000);
        expect(metrics.performance.failedRequests).toBe(0);
      }

      // Generar y mostrar reporte
      console.log(generateTextReport(report));
    });
  });

  // ==========================================================================
  // Concurrency Tests
  // ==========================================================================

  test.describe('Concurrency', () => {
    /**
     * Test de múltiples tareas en paralelo
     * 
     * @tags @concurrency @parallel
     * @description Valida que la aplicación maneja correctamente múltiples tareas simultáneas
     */
    test('debería manejar múltiples tareas en paralelo', async ({ page }) => {
      const tasks = [
        TEST_INSTRUCTIONS.DEFAULT,
        TEST_INSTRUCTIONS.SIMPLE,
        'Crea un archivo test.txt',
      ];

      await createMultipleTasks(page, tasks, 500);

      // Verificar que todas las tareas aparecen
      const taskCards = await page
        .locator('[data-testid="task-card"], .task-card, [class*="task"]')
        .count();
      expect(taskCards).toBeGreaterThanOrEqual(tasks.length);
    });
  });

  // ==========================================================================
  // Accessibility & Visual Tests
  // ==========================================================================

  test.describe('Accessibility & Visual', () => {
    /**
     * Test de accesibilidad básica
     * 
     * @tags @accessibility @a11y
     * @description Valida que la aplicación cumple con estándares básicos de accesibilidad
     */
    test('debería validar accesibilidad básica', async ({ page }) => {
      await navigateToAgentControl(page);

      // Importar validación de accesibilidad
      const {
        validateBasicAccessibility,
        validateKeyboardAccessibility,
        validateVisualElements,
      } = await import('./test-helpers/visual-testing');
      
      await validateBasicAccessibility(page);
      await validateKeyboardAccessibility(page);
      await validateVisualElements(page);
    });

    /**
     * Test de layout responsive
     * 
     * @tags @responsive @visual
     * @description Valida que el layout funciona correctamente en diferentes viewports
     */
    test('debería validar layout responsive', async ({ page }) => {
      const {
        validatePageLayout,
        validateResponsiveLayout,
        compareScreenshot,
      } = await import('./test-helpers/visual-testing');

      await navigateToAgentControl(page);

      // Probar diferentes viewports
      const viewports = [
        { width: 1920, height: 1080 }, // Desktop
        { width: 1440, height: 900 },  // Laptop
        { width: 768, height: 1024 },   // Tablet
        { width: 375, height: 667 },    // Mobile
      ];

      for (const viewport of viewports) {
        await validatePageLayout(page, viewport);
        // Tomar screenshot para cada viewport
        await compareScreenshot(page, `layout-${viewport.width}x${viewport.height}`, {
          fullPage: true,
        });
      }

      // También validar con función helper
      await validateResponsiveLayout(page);
    });
  });

  // ==========================================================================
  // Edge Cases & Error Handling
  // ==========================================================================

  test.describe('Edge Cases & Error Handling', () => {
    /**
     * Test de manejo de errores de red
     * 
     * @tags @error-handling @network
     * @description Valida que la aplicación maneja gracefully los errores de red
     */
    test('debería manejar errores de red gracefully', async ({ page }) => {
      await testNetworkErrorHandling(page);
    });

    /**
     * Test de persistencia de estado
     * 
     * @tags @state-management @persistence
     * @description Valida que el estado persiste después de refresh
     */
    test('debería persistir estado después de refresh', async ({ page }) => {
      await testStatePersistence(page);
    });

    /**
     * Test con instrucciones muy largas
     * 
     * @tags @edge-case @input-validation
     * @description Valida el comportamiento con inputs muy grandes
     */
    test('debería validar comportamiento con instrucciones muy largas', async ({
      page,
    }) => {
      await testWithLongInstruction(page, 5000);
    });

    /**
     * Test con instrucciones vacías
     * 
     * @tags @edge-case @input-validation
     * @description Valida la validación de formularios con inputs vacíos
     */
  test('debería validar comportamiento con instrucciones vacías', async ({
    page,
  }) => {
    await testWithEmptyInstruction(page);
  });

  // ==========================================================================
  // Data-Driven Tests
  // ==========================================================================

  test('debería ejecutar tests data-driven con múltiples variaciones', async ({
    page,
  }) => {
    const { runDataDrivenTest, createTestDataScenarios, validateDataDrivenResults } = await import(
      './test-helpers/data-driven'
    );

    const testData = createTestDataScenarios();
    const results = await runDataDrivenTest(page, testData, async (p, data) => {
      await navigateToAgentControl(p);
      await createTask(p, data.instruction);
      await waitForTaskToAppear(p);
    });

    const validation = validateDataDrivenResults(results);
    expect(validation.passed).toBe(true);
    expect(validation.summary.passRate).toBeGreaterThanOrEqual(80);
  });

  // ==========================================================================
  // Comparison Tests
  // ==========================================================================

  test('debería comparar resultados de performance con baseline', async ({
    page,
  }) => {
    const { comparePerformance, collectPerformanceMetrics } = await import(
      './test-helpers/comparison'
    );
    const { collectPerformanceMetrics: collectMetrics } = await import(
      './test-helpers/metrics'
    );

    await navigateToAgentControl(page);

    // Baseline de performance
    const baseline = {
      pageLoadTime: 3000,
      totalRequests: 10,
      failedRequests: 0,
    };

    // Obtener métricas actuales
    const metrics = await collectMetrics(page);
    const actual = {
      pageLoadTime: metrics.pageLoadTime || 0,
      totalRequests: metrics.totalRequests || 0,
      failedRequests: metrics.failedRequests || 0,
    };

    // Comparar con tolerancia
    const comparison = comparePerformance(baseline, actual, {
      pageLoadTime: 50, // 50% de tolerancia
      totalRequests: 5, // 5 requests de diferencia
      failedRequests: 0, // 0 requests fallidos permitidos
    });

    expect(comparison.passed).toBe(true);
  });

  // ==========================================================================
  // CI/CD Integration Tests
  // ==========================================================================

  test('debería generar reportes para CI/CD', async ({ page }, testInfo) => {
    const {
      createCICDReportFromTestInfo,
      generateJUnitXML,
      generateJSONReport,
      saveCICDReport,
      isCI,
    } = await import('./test-helpers/ci-cd');

    await navigateToAgentControl(page);
    await createTask(page, TEST_INSTRUCTIONS.DEFAULT);

    // Crear reporte
    const report = createCICDReportFromTestInfo(testInfo, {
      environment: 'test',
      version: '1.0.0',
    });

    // Generar reportes en diferentes formatos
    const junitXML = generateJUnitXML([report]);
    const jsonReport = generateJSONReport([report]);

    // Guardar reportes si estamos en CI
    if (isCI()) {
      await saveCICDReport(junitXML, 'xml');
      await saveCICDReport(jsonReport, 'json');
    }

    expect(report.status).toBe('passed');
  });

  // ==========================================================================
  // Advanced Accessibility Tests
  // ==========================================================================

  test('debería ejecutar validación completa de accesibilidad', async ({
    page,
  }) => {
    const { runFullAccessibilityCheck } = await import(
      './test-helpers/accessibility'
    );

    await navigateToAgentControl(page);

    const result = await runFullAccessibilityCheck(page);

    // Verificar que no hay violaciones críticas o serias
    const criticalViolations = result.axe.violations.filter(
      (v) => v.impact === 'critical' || v.impact === 'serious'
    );
    expect(criticalViolations.length).toBe(0);

    // Verificar accesibilidad por teclado
    expect(result.keyboard.passed).toBe(true);

    // Verificar ARIA
    expect(result.aria.passed).toBe(true);
  });

  // ==========================================================================
  // Advanced Comparison Tests
  // ==========================================================================

  test('debería comparar objetos y arrays correctamente', async ({ page }) => {
    const { compareObjects, compareArrays, validateArrayContains } = await import(
      './test-helpers/comparison'
    );

    // Comparar objetos
    const obj1 = { name: 'test', value: 123 };
    const obj2 = { name: 'test', value: 123 };
    const objComparison = compareObjects(obj1, obj2);
    expect(objComparison.match).toBe(true);

    // Comparar arrays
    const arr1 = [1, 2, 3];
    const arr2 = [1, 2, 3];
    const arrComparison = compareArrays(arr1, arr2);
    expect(arrComparison.match).toBe(true);

    // Validar que array contiene elementos
    const contains = validateArrayContains([1, 2, 3, 4, 5], [2, 4]);
    expect(contains.passed).toBe(true);
    expect(contains.missing.length).toBe(0);
  });
});

  // ==========================================================================
  // Parallel Execution Tests
  // ==========================================================================

  test.describe('Parallel Execution', () => {
    test('debería crear tareas en paralelo eficientemente', async ({ page }) => {
      const tasks = Array.from(
        { length: 5 },
        (_, i) => `Crea archivo parallel-${i}.txt`
      );

      await createTasksInParallel(page, tasks, 3);

      const taskCards = await page
        .locator('[data-testid="task-card"], .task-card, [class*="task"]')
        .count();
      expect(taskCards).toBeGreaterThanOrEqual(tasks.length);
    });

    test('debería validar procesamiento paralelo de tareas', async ({ page }) => {
      const result = await validateParallelTaskProcessing(page, 4);

      expect(result.visible).toBeGreaterThanOrEqual(result.created);
      console.log(
        `📊 Tareas: ${result.created} creadas, ${result.visible} visibles, ${result.processing} procesando`
      );
    });
  });

  // ==========================================================================
  // Error Scenarios Tests
  // ==========================================================================

  test.describe('Error Scenarios', () => {
    test('debería manejar datos inválidos correctamente', async ({ page }) => {
      const { validationShown, submissionBlocked } = await testInvalidDataHandling(page);
      expect(validationShown || submissionBlocked).toBeTruthy();
    });

    test('debería manejar rate limiting', async ({ page }) => {
      const { rateLimited, rateLimitMessage } = await testRateLimiting(page, 5);
      if (rateLimited) {
        expect(rateLimitMessage).toBeTruthy();
        console.log(`⚠️ Rate limit detectado: ${rateLimitMessage}`);
      }
    });

    test('debería mostrar indicadores de carga en servidor lento', async ({
      page,
    }) => {
      const { loadingIndicatorShown, timeoutHandled } = await testSlowServerResponse(
        page,
        3000
      );
      expect(loadingIndicatorShown || timeoutHandled).toBeTruthy();
    });
  });
});
