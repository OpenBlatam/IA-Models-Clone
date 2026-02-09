/**
 * Utilidades para Organización de Tests
 * 
 * Proporciona funciones para agrupar y organizar tests de forma lógica
 */
import { Page } from '@playwright/test';

// ============================================================================
// Types
// ============================================================================

export interface TestGroup {
  name: string;
  description: string;
  tags: string[];
  tests: Array<{
    name: string;
    fn: (page: Page) => Promise<void>;
  }>;
}

export interface TestSuite {
  name: string;
  groups: TestGroup[];
  setup?: (page: Page) => Promise<void>;
  teardown?: (page: Page) => Promise<void>;
}

// ============================================================================
// Test Organization Helpers
// ============================================================================

/**
 * Crea un grupo de tests relacionado
 */
export function createTestGroup(
  name: string,
  description: string,
  tags: string[] = []
): TestGroup {
  return {
    name,
    description,
    tags,
    tests: [],
  };
}

/**
 * Agrega un test a un grupo
 */
export function addTestToGroup(
  group: TestGroup,
  name: string,
  testFn: (page: Page) => Promise<void>
): void {
  group.tests.push({ name, fn: testFn });
}

/**
 * Crea una suite de tests
 */
export function createTestSuite(
  name: string,
  groups: TestGroup[],
  options: {
    setup?: (page: Page) => Promise<void>;
    teardown?: (page: Page) => Promise<void>;
  } = {}
): TestSuite {
  return {
    name,
    groups,
    setup: options.setup,
    teardown: options.teardown,
  };
}

// ============================================================================
// Test Filtering
// ============================================================================

/**
 * Filtra tests por tags
 */
export function filterTestsByTags(
  suite: TestSuite,
  tags: string[]
): TestSuite {
  const filteredGroups = suite.groups
    .filter((group) => group.tags.some((tag) => tags.includes(tag)))
    .map((group) => ({
      ...group,
      tests: group.tests.filter(() => true), // Mantener todos los tests del grupo
    }));

  return {
    ...suite,
    groups: filteredGroups,
  };
}

/**
 * Obtiene estadísticas de una suite de tests
 */
export function getTestSuiteStats(suite: TestSuite): {
  totalGroups: number;
  totalTests: number;
  tags: string[];
  groupsByTag: Record<string, number>;
} {
  const allTags = new Set<string>();
  const groupsByTag: Record<string, number> = {};

  suite.groups.forEach((group) => {
    group.tags.forEach((tag) => {
      allTags.add(tag);
      groupsByTag[tag] = (groupsByTag[tag] || 0) + 1;
    });
  });

  const totalTests = suite.groups.reduce(
    (sum, group) => sum + group.tests.length,
    0
  );

  return {
    totalGroups: suite.groups.length,
    totalTests,
    tags: Array.from(allTags),
    groupsByTag,
  };
}

// ============================================================================
// Test Execution Helpers
// ============================================================================

/**
 * Ejecuta una suite de tests con setup/teardown
 */
export async function executeTestSuite(
  suite: TestSuite,
  page: Page
): Promise<{
  passed: number;
  failed: number;
  results: Array<{ group: string; test: string; passed: boolean; error?: Error }>;
}> {
  const results: Array<{
    group: string;
    test: string;
    passed: boolean;
    error?: Error;
  }> = [];

  // Setup
  if (suite.setup) {
    await suite.setup(page);
  }

  // Ejecutar tests
  for (const group of suite.groups) {
    for (const test of group.tests) {
      try {
        await test.fn(page);
        results.push({
          group: group.name,
          test: test.name,
          passed: true,
        });
      } catch (error) {
        results.push({
          group: group.name,
          test: test.name,
          passed: false,
          error: error instanceof Error ? error : new Error(String(error)),
        });
      }
    }
  }

  // Teardown
  if (suite.teardown) {
    await suite.teardown(page);
  }

  const passed = results.filter((r) => r.passed).length;
  const failed = results.filter((r) => !r.passed).length;

  return {
    passed,
    failed,
    results,
  };
}



