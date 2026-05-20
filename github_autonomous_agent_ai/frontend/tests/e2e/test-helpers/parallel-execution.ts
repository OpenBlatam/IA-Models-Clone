/**
 * Utilidades para Ejecución Paralela de Tests
 * 
 * Proporciona helpers para ejecutar múltiples operaciones en paralelo
 * y validar comportamiento concurrente
 */
import { Page } from '@playwright/test';
import { navigateToAgentControl, createTask, waitForTaskToAppear } from '../helpers';
import { TEST_INSTRUCTIONS } from '../constants';

// ============================================================================
// Parallel Execution Helpers
// ============================================================================

/**
 * Crea múltiples tareas en paralelo usando Promise.all
 */
export async function createTasksInParallel(
  page: Page,
  instructions: string[],
  maxConcurrent: number = 3
): Promise<void> {
  await navigateToAgentControl(page);

  // Procesar en lotes para evitar sobrecarga
  for (let i = 0; i < instructions.length; i += maxConcurrent) {
    const batch = instructions.slice(i, i + maxConcurrent);
    
    await Promise.all(
      batch.map((instruction) => createTask(page, instruction))
    );
    
    // Pequeña pausa entre lotes
    if (i + maxConcurrent < instructions.length) {
      await page.waitForTimeout(200);
    }
  }

  // Esperar a que aparezcan las tareas
  await waitForTaskToAppear(page);
}

/**
 * Ejecuta múltiples operaciones en paralelo con límite de concurrencia
 */
export async function executeInParallel<T>(
  operations: Array<() => Promise<T>>,
  maxConcurrent: number = 3
): Promise<T[]> {
  const results: T[] = [];
  
  for (let i = 0; i < operations.length; i += maxConcurrent) {
    const batch = operations.slice(i, i + maxConcurrent);
    const batchResults = await Promise.all(
      batch.map((op) => op())
    );
    results.push(...batchResults);
  }
  
  return results;
}

/**
 * Valida que múltiples tareas se procesan correctamente en paralelo
 */
export async function validateParallelTaskProcessing(
  page: Page,
  taskCount: number = 3
): Promise<{
  created: number;
  visible: number;
  processing: number;
}> {
  await navigateToAgentControl(page);

  // Crear tareas
  const instructions = Array.from(
    { length: taskCount },
    (_, i) => `Tarea paralela ${i + 1}: Crea archivo test-${i}.txt`
  );

  await createTasksInParallel(page, instructions);

  // Contar tareas visibles
  const taskCards = page.locator(
    '[data-testid="task-card"], .task-card, [class*="task"]'
  );
  const visibleCount = await taskCards.count();

  // Contar tareas en procesamiento
  const processingCount = await taskCards
    .filter({ hasText: /procesando|processing/i })
    .count();

  return {
    created: instructions.length,
    visible: visibleCount,
    processing: processingCount,
  };
}



