/**
 * Utilidades para Testing en Paralelo
 * 
 * Proporciona funciones para ejecutar tests y operaciones en paralelo
 */
import { Page } from '@playwright/test';
import { createTask, waitForTaskToAppear } from '../helpers';

// ============================================================================
// Types
// ============================================================================

export interface ParallelTask {
  instruction: string;
  id: string;
  status?: 'pending' | 'completed' | 'failed';
  result?: any;
  error?: Error;
}

export interface ParallelTestResult {
  tasks: ParallelTask[];
  totalDuration: number;
  successCount: number;
  failureCount: number;
  averageDuration: number;
}

// ============================================================================
// Parallel Execution
// ============================================================================

/**
 * Ejecuta múltiples tareas en paralelo
 */
export async function executeTasksInParallel(
  page: Page,
  instructions: string[],
  options: {
    maxConcurrent?: number;
    delayBetweenBatches?: number;
  } = {}
): Promise<ParallelTestResult> {
  const { maxConcurrent = 3, delayBetweenBatches = 1000 } = options;
  const startTime = Date.now();
  const tasks: ParallelTask[] = instructions.map((instruction, index) => ({
    instruction,
    id: `task-${index + 1}`,
    status: 'pending',
  }));

  // Ejecutar en batches
  for (let i = 0; i < tasks.length; i += maxConcurrent) {
    const batch = tasks.slice(i, i + maxConcurrent);
    
    await Promise.all(
      batch.map(async (task) => {
        try {
          await createTask(page, task.instruction);
          task.status = 'completed';
        } catch (error) {
          task.status = 'failed';
          task.error = error instanceof Error ? error : new Error(String(error));
        }
      })
    );

    // Pequeña pausa entre batches
    if (i + maxConcurrent < tasks.length) {
      await page.waitForTimeout(delayBetweenBatches);
    }
  }

  // Esperar a que todas las tareas aparezcan
  await waitForTaskToAppear(page);

  const totalDuration = Date.now() - startTime;
  const successCount = tasks.filter((t) => t.status === 'completed').length;
  const failureCount = tasks.filter((t) => t.status === 'failed').length;
  const averageDuration = totalDuration / tasks.length;

  return {
    tasks,
    totalDuration,
    successCount,
    failureCount,
    averageDuration,
  };
}

/**
 * Ejecuta una operación múltiples veces en paralelo
 */
export async function executeOperationInParallel<T>(
  operation: () => Promise<T>,
  count: number,
  options: {
    maxConcurrent?: number;
    delayBetweenBatches?: number;
  } = {}
): Promise<Array<{ success: boolean; result?: T; error?: Error }>> {
  const { maxConcurrent = 5, delayBetweenBatches = 100 } = options;
  const results: Array<{ success: boolean; result?: T; error?: Error }> = [];

  // Ejecutar en batches
  for (let i = 0; i < count; i += maxConcurrent) {
    const batchSize = Math.min(maxConcurrent, count - i);
    const batch = Array.from({ length: batchSize }, () => operation());

    const batchResults = await Promise.allSettled(batch);
    
    results.push(
      ...batchResults.map((result) => {
        if (result.status === 'fulfilled') {
          return { success: true, result: result.value };
        } else {
          return {
            success: false,
            error: result.reason instanceof Error
              ? result.reason
              : new Error(String(result.reason)),
          };
        }
      })
    );

    // Pequeña pausa entre batches
    if (i + maxConcurrent < count) {
      await new Promise((resolve) => setTimeout(resolve, delayBetweenBatches));
    }
  }

  return results;
}

/**
 * Valida que múltiples operaciones completan correctamente en paralelo
 */
export async function validateParallelExecution(
  operations: Array<{ name: string; action: () => Promise<void> }>,
  options: {
    maxConcurrent?: number;
    timeout?: number;
  } = {}
): Promise<{
  passed: boolean;
  results: Array<{ name: string; success: boolean; duration: number; error?: Error }>;
  totalDuration: number;
}> {
  const { maxConcurrent = 3, timeout = 30000 } = options;
  const startTime = Date.now();
  const results: Array<{
    name: string;
    success: boolean;
    duration: number;
    error?: Error;
  }> = [];

  // Ejecutar en batches
  for (let i = 0; i < operations.length; i += maxConcurrent) {
    const batch = operations.slice(i, i + maxConcurrent);

    const batchResults = await Promise.allSettled(
      batch.map(async (op) => {
        const opStartTime = Date.now();
        await Promise.race([
          op.action(),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Timeout')), timeout)
          ),
        ]);
        return {
          name: op.name,
          duration: Date.now() - opStartTime,
        };
      })
    );

    results.push(
      ...batchResults.map((result, index) => {
        if (result.status === 'fulfilled') {
          return {
            name: batch[index].name,
            success: true,
            duration: result.value.duration,
          };
        } else {
          return {
            name: batch[index].name,
            success: false,
            duration: 0,
            error:
              result.reason instanceof Error
                ? result.reason
                : new Error(String(result.reason)),
          };
        }
      })
    );
  }

  const totalDuration = Date.now() - startTime;
  const passed = results.every((r) => r.success);

  return {
    passed,
    results,
    totalDuration,
  };
}



