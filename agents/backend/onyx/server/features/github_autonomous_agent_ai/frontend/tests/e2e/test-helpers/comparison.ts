/**
 * Utilidades para Comparación y Validación Avanzada
 * 
 * Proporciona funciones para comparar resultados, validar diferencias y hacer assertions avanzadas
 */
import { Page, Locator } from '@playwright/test';

// ============================================================================
// Types
// ============================================================================

export interface ComparisonResult {
  match: boolean;
  differences: Array<{ field: string; expected: any; actual: any }>;
  similarity: number; // 0-1
}

export interface SnapshotComparison {
  match: boolean;
  diffPercentage: number;
  differences: string[];
}

// ============================================================================
// Comparison Helpers
// ============================================================================

/**
 * Compara dos objetos y encuentra diferencias
 */
export function compareObjects(
  expected: Record<string, any>,
  actual: Record<string, any>
): ComparisonResult {
  const differences: Array<{ field: string; expected: any; actual: any }> = [];
  const allKeys = new Set([...Object.keys(expected), ...Object.keys(actual)]);

  for (const key of allKeys) {
    const expectedValue = expected[key];
    const actualValue = actual[key];

    if (JSON.stringify(expectedValue) !== JSON.stringify(actualValue)) {
      differences.push({
        field: key,
        expected: expectedValue,
        actual: actualValue,
      });
    }
  }

  const similarity =
    allKeys.size > 0
      ? (allKeys.size - differences.length) / allKeys.size
      : 1;

  return {
    match: differences.length === 0,
    differences,
    similarity,
  };
}

/**
 * Compara dos arrays y encuentra diferencias
 */
export function compareArrays<T>(
  expected: T[],
  actual: T[],
  compareFn?: (a: T, b: T) => boolean
): ComparisonResult {
  const differences: Array<{ field: string; expected: any; actual: any }> = [];
  const maxLength = Math.max(expected.length, actual.length);

  for (let i = 0; i < maxLength; i++) {
    const expectedItem = expected[i];
    const actualItem = actual[i];

    if (expectedItem === undefined) {
      differences.push({
        field: `[${i}]`,
        expected: undefined,
        actual: actualItem,
      });
    } else if (actualItem === undefined) {
      differences.push({
        field: `[${i}]`,
        expected: expectedItem,
        actual: undefined,
      });
    } else if (compareFn) {
      if (!compareFn(expectedItem, actualItem)) {
        differences.push({
          field: `[${i}]`,
          expected: expectedItem,
          actual: actualItem,
        });
      }
    } else if (JSON.stringify(expectedItem) !== JSON.stringify(actualItem)) {
      differences.push({
        field: `[${i}]`,
        expected: expectedItem,
        actual: actualItem,
      });
    }
  }

  const similarity =
    maxLength > 0 ? (maxLength - differences.length) / maxLength : 1;

  return {
    match: differences.length === 0,
    differences,
    similarity,
  };
}

/**
 * Compara texto con tolerancia a diferencias menores
 */
export function compareText(
  expected: string,
  actual: string,
  options: {
    caseSensitive?: boolean;
    ignoreWhitespace?: boolean;
    tolerance?: number; // 0-1, porcentaje de diferencia permitido
  } = {}
): ComparisonResult {
  const {
    caseSensitive = true,
    ignoreWhitespace = false,
    tolerance = 0,
  } = options;

  let expectedText = expected;
  let actualText = actual;

  if (!caseSensitive) {
    expectedText = expectedText.toLowerCase();
    actualText = actualText.toLowerCase();
  }

  if (ignoreWhitespace) {
    expectedText = expectedText.replace(/\s+/g, ' ').trim();
    actualText = actualText.replace(/\s+/g, ' ').trim();
  }

  const differences: Array<{ field: string; expected: any; actual: any }> = [];

  if (expectedText !== actualText) {
    // Calcular similitud usando Levenshtein distance aproximado
    const maxLength = Math.max(expectedText.length, actualText.length);
    let diffCount = 0;
    const minLength = Math.min(expectedText.length, actualText.length);

    for (let i = 0; i < minLength; i++) {
      if (expectedText[i] !== actualText[i]) {
        diffCount++;
      }
    }
    diffCount += Math.abs(expectedText.length - actualText.length);

    const diffPercentage = maxLength > 0 ? diffCount / maxLength : 0;

    if (diffPercentage > tolerance) {
      differences.push({
        field: 'text',
        expected: expectedText,
        actual: actualText,
      });
    }
  }

  const similarity =
    differences.length === 0 || tolerance > 0 ? 1 - (differences.length > 0 ? 0.1 : 0) : 0;

  return {
    match: differences.length === 0,
    differences,
    similarity,
  };
}

/**
 * Compara elementos del DOM
 */
export async function compareElements(
  expected: Locator,
  actual: Locator,
  options: {
    compareText?: boolean;
    compareAttributes?: string[];
    compareStyles?: string[];
  } = {}
): Promise<ComparisonResult> {
  const {
    compareText: shouldCompareText = true,
    compareAttributes = [],
    compareStyles = [],
  } = options;

  const differences: Array<{ field: string; expected: any; actual: any }> = [];

  // Comparar texto
  if (shouldCompareText) {
    const expectedText = await expected.textContent();
    const actualText = await actual.textContent();
    if (expectedText !== actualText) {
      differences.push({
        field: 'text',
        expected: expectedText,
        actual: actualText,
      });
    }
  }

  // Comparar atributos
  for (const attr of compareAttributes) {
    const expectedValue = await expected.getAttribute(attr);
    const actualValue = await actual.getAttribute(attr);
    if (expectedValue !== actualValue) {
      differences.push({
        field: `attribute:${attr}`,
        expected: expectedValue,
        actual: actualValue,
      });
    }
  }

  // Comparar estilos
  for (const style of compareStyles) {
    const expectedValue = await expected.evaluate(
      (el, s) => window.getComputedStyle(el).getPropertyValue(s),
      style
    );
    const actualValue = await actual.evaluate(
      (el, s) => window.getComputedStyle(el).getPropertyValue(s),
      style
    );
    if (expectedValue !== actualValue) {
      differences.push({
        field: `style:${style}`,
        expected: expectedValue,
        actual: actualValue,
      });
    }
  }

  return {
    match: differences.length === 0,
    differences,
    similarity: differences.length === 0 ? 1 : 0.5,
  };
}

/**
 * Valida que un valor está dentro de un rango
 */
export function validateRange(
  value: number,
  min: number,
  max: number,
  inclusive: boolean = true
): boolean {
  if (inclusive) {
    return value >= min && value <= max;
  }
  return value > min && value < max;
}

/**
 * Valida que un array contiene elementos esperados
 */
export function validateArrayContains<T>(
  array: T[],
  expected: T[],
  allRequired: boolean = true
): {
  passed: boolean;
  missing: T[];
  extra: T[];
} {
  const missing: T[] = [];
  const extra: T[] = [...array];

  for (const item of expected) {
    const index = extra.indexOf(item);
    if (index === -1) {
      missing.push(item);
    } else {
      extra.splice(index, 1);
    }
  }

  const passed = allRequired
    ? missing.length === 0
    : missing.length < expected.length;

  return {
    passed,
    missing,
    extra,
  };
}

/**
 * Compara resultados de performance
 */
export function comparePerformance(
  baseline: {
    pageLoadTime: number;
    totalRequests: number;
    failedRequests: number;
  },
  actual: {
    pageLoadTime: number;
    totalRequests: number;
    failedRequests: number;
  },
  tolerance: {
    pageLoadTime?: number; // porcentaje
    totalRequests?: number; // diferencia absoluta
    failedRequests?: number; // diferencia absoluta
  } = {}
): {
  passed: boolean;
  differences: Array<{ metric: string; baseline: number; actual: number; diff: number }>;
} {
  const differences: Array<{
    metric: string;
    baseline: number;
    actual: number;
    diff: number;
  }> = [];

  // Comparar pageLoadTime
  if (tolerance.pageLoadTime !== undefined) {
    const diff = Math.abs(actual.pageLoadTime - baseline.pageLoadTime);
    const diffPercentage = (diff / baseline.pageLoadTime) * 100;
    if (diffPercentage > tolerance.pageLoadTime) {
      differences.push({
        metric: 'pageLoadTime',
        baseline: baseline.pageLoadTime,
        actual: actual.pageLoadTime,
        diff: diffPercentage,
      });
    }
  }

  // Comparar totalRequests
  if (tolerance.totalRequests !== undefined) {
    const diff = Math.abs(actual.totalRequests - baseline.totalRequests);
    if (diff > tolerance.totalRequests) {
      differences.push({
        metric: 'totalRequests',
        baseline: baseline.totalRequests,
        actual: actual.totalRequests,
        diff,
      });
    }
  }

  // Comparar failedRequests
  if (tolerance.failedRequests !== undefined) {
    const diff = Math.abs(actual.failedRequests - baseline.failedRequests);
    if (diff > tolerance.failedRequests) {
      differences.push({
        metric: 'failedRequests',
        baseline: baseline.failedRequests,
        actual: actual.failedRequests,
        diff,
      });
    }
  }

  return {
    passed: differences.length === 0,
    differences,
  };
}



