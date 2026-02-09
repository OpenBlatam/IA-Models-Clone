/**
 * Utilidades para Data-Driven Testing
 * 
 * Proporciona funciones para ejecutar tests con diferentes datos de entrada
 */
import { Page } from '@playwright/test';
import { createTask, navigateToAgentControl } from '../helpers';

// ============================================================================
// Types
// ============================================================================

export interface TestData {
  name: string;
  instruction: string;
  expectedResult?: 'success' | 'failure' | 'partial';
  metadata?: Record<string, any>;
}

export interface DataDrivenTestResult {
  data: TestData;
  passed: boolean;
  duration: number;
  error?: Error;
  result?: any;
}

// ============================================================================
// Data-Driven Testing Helpers
// ============================================================================

/**
 * Ejecuta un test con múltiples conjuntos de datos
 */
export async function runDataDrivenTest(
  page: Page,
  testData: TestData[],
  testFn: (page: Page, data: TestData) => Promise<any>
): Promise<DataDrivenTestResult[]> {
  const results: DataDrivenTestResult[] = [];

  for (const data of testData) {
    const startTime = Date.now();
    let passed = false;
    let error: Error | undefined;
    let result: any;

    try {
      result = await testFn(page, data);
      passed = true;
    } catch (e) {
      error = e instanceof Error ? e : new Error(String(e));
      passed = false;
    }

    results.push({
      data,
      passed,
      duration: Date.now() - startTime,
      error,
      result,
    });
  }

  return results;
}

/**
 * Ejecuta un test con datos parametrizados
 */
export async function runParametrizedTest<T>(
  page: Page,
  parameters: T[],
  testFn: (page: Page, param: T) => Promise<void>
): Promise<Array<{ param: T; passed: boolean; error?: Error }>> {
  const results: Array<{ param: T; passed: boolean; error?: Error }> = [];

  for (const param of parameters) {
    try {
      await testFn(page, param);
      results.push({ param, passed: true });
    } catch (error) {
      results.push({
        param,
        passed: false,
        error: error instanceof Error ? error : new Error(String(error)),
      });
    }
  }

  return results;
}

/**
 * Genera datos de prueba basados en templates
 */
export function generateTestDataFromTemplate(
  template: string,
  variations: Array<Record<string, string>>
): TestData[] {
  return variations.map((vars, index) => {
    let instruction = template;
    for (const [key, value] of Object.entries(vars)) {
      instruction = instruction.replace(new RegExp(`\\{\\{${key}\\}\\}`, 'g'), value);
    }

    return {
      name: `Variation ${index + 1}`,
      instruction,
      metadata: vars,
    };
  });
}

/**
 * Crea datos de prueba para diferentes escenarios
 */
export function createTestDataScenarios(): TestData[] {
  return [
    {
      name: 'Simple file creation',
      instruction: 'Crea un archivo README.md con "Hello World"',
      expectedResult: 'success',
    },
    {
      name: 'Multiple files',
      instruction: 'Crea archivos README.md, package.json y .gitignore',
      expectedResult: 'success',
    },
    {
      name: 'File with content',
      instruction: 'Crea un archivo config.json con {"version": "1.0.0"}',
      expectedResult: 'success',
    },
    {
      name: 'Nested directory',
      instruction: 'Crea la estructura src/components/Button.tsx',
      expectedResult: 'success',
    },
    {
      name: 'Update existing',
      instruction: 'Actualiza README.md agregando una sección de instalación',
      expectedResult: 'success',
    },
  ];
}

/**
 * Valida resultados de data-driven tests
 */
export function validateDataDrivenResults(
  results: DataDrivenTestResult[]
): {
  passed: boolean;
  summary: {
    total: number;
    passed: number;
    failed: number;
    passRate: number;
  };
  failures: DataDrivenTestResult[];
} {
  const passed = results.filter((r) => r.passed).length;
  const failed = results.filter((r) => !r.passed).length;
  const failures = results.filter((r) => !r.passed);

  return {
    passed: failed === 0,
    summary: {
      total: results.length,
      passed,
      failed,
      passRate: results.length > 0 ? (passed / results.length) * 100 : 0,
    },
    failures,
  };
}



