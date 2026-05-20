/**
 * Sistema de Métricas para Tests E2E
 * 
 * Proporciona tracking de métricas de performance y duración de pasos
 */
import { Page } from '@playwright/test';

// ============================================================================
// Types
// ============================================================================

export interface StepMetrics {
  name: string;
  duration: number;
  passed: boolean;
  startTime: number;
  endTime?: number;
}

export interface PerformanceMetrics {
  pageLoadTime: number;
  domContentLoadedTime: number;
  totalRequests: number;
  failedRequests: number;
  totalResponseTime: number;
  averageResponseTime: number;
  memoryUsage?: number;
}

export interface TestMetrics {
  testName: string;
  totalDuration: number;
  passed: boolean;
  steps: StepMetrics[];
  performance?: PerformanceMetrics;
  screenshots: string[];
  errors: string[];
  startTime: number;
  endTime?: number;
}

// ============================================================================
// Metrics Tracker
// ============================================================================

export class MetricsTracker {
  private testName: string;
  private startTime: number = 0;
  private endTime?: number;
  private steps: StepMetrics[] = [];
  private currentStep?: StepMetrics;
  private screenshots: string[] = [];
  private errors: string[] = [];
  private passed: boolean = true;

  constructor(testName: string) {
    this.testName = testName;
  }

  start(): void {
    this.startTime = Date.now();
  }

  startStep(name: string): void {
    // Finalizar paso anterior si existe
    if (this.currentStep) {
      this.endStep(this.currentStep.passed);
    }

    this.currentStep = {
      name,
      duration: 0,
      passed: true,
      startTime: Date.now(),
    };
  }

  endStep(passed: boolean = true): void {
    if (!this.currentStep) return;

    this.currentStep.endTime = Date.now();
    this.currentStep.duration = this.currentStep.endTime - this.currentStep.startTime;
    this.currentStep.passed = passed;

    if (!passed) {
      this.passed = false;
    }

    this.steps.push(this.currentStep);
    this.currentStep = undefined;
  }

  addScreenshot(path: string): void {
    this.screenshots.push(path);
  }

  addError(error: string): void {
    this.errors.push(error);
    this.passed = false;
  }

  finish(passed: boolean = true): TestMetrics {
    // Finalizar paso actual si existe
    if (this.currentStep) {
      this.endStep(this.currentStep.passed);
    }

    this.endTime = Date.now();
    this.passed = passed && this.passed;

    return {
      testName: this.testName,
      totalDuration: this.endTime - this.startTime,
      passed: this.passed,
      steps: this.steps,
      screenshots: this.screenshots,
      errors: this.errors,
      startTime: this.startTime,
      endTime: this.endTime,
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Crea un nuevo tracker de métricas
 */
export function createMetricsTracker(testName: string): MetricsTracker {
  return new MetricsTracker(testName);
}

// ============================================================================
// Performance Collection
// ============================================================================

/**
 * Recolecta métricas de performance de la página
 */
export async function collectPerformanceMetrics(page: Page): Promise<PerformanceMetrics> {
  // Obtener métricas de performance desde el navegador
  const performanceTiming = await page.evaluate(() => {
    const timing = performance.timing;
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    
    return {
      pageLoadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
      domContentLoadedTime: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : 0,
    };
  });

  // Obtener información de requests
  const requests = await page.evaluate(() => {
    const entries = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    return {
      total: entries.length,
      failed: entries.filter(e => e.transferSize === 0 && e.decodedBodySize === 0).length,
      totalResponseTime: entries.reduce((sum, e) => sum + (e.responseEnd - e.requestStart), 0),
    };
  });

  const averageResponseTime = requests.total > 0 
    ? requests.totalResponseTime / requests.total 
    : 0;

  return {
    pageLoadTime: performanceTiming.pageLoadTime,
    domContentLoadedTime: performanceTiming.domContentLoadedTime,
    totalRequests: requests.total,
    failedRequests: requests.failed,
    totalResponseTime: requests.totalResponseTime,
    averageResponseTime,
  };
}

/**
 * Compara métricas de performance con valores esperados
 */
export function comparePerformanceMetrics(
  actual: PerformanceMetrics,
  expected: Partial<PerformanceMetrics>
): {
  passed: boolean;
  differences: Array<{ metric: string; actual: number; expected: number }>;
} {
  const differences: Array<{ metric: string; actual: number; expected: number }> = [];

  if (expected.pageLoadTime && actual.pageLoadTime > expected.pageLoadTime) {
    differences.push({
      metric: 'pageLoadTime',
      actual: actual.pageLoadTime,
      expected: expected.pageLoadTime,
    });
  }

  if (expected.failedRequests !== undefined && actual.failedRequests > expected.failedRequests) {
    differences.push({
      metric: 'failedRequests',
      actual: actual.failedRequests,
      expected: expected.failedRequests,
    });
  }

  if (expected.averageResponseTime && actual.averageResponseTime > expected.averageResponseTime) {
    differences.push({
      metric: 'averageResponseTime',
      actual: actual.averageResponseTime,
      expected: expected.averageResponseTime,
    });
  }

  return {
    passed: differences.length === 0,
    differences,
  };
}
