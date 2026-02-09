/**
 * Helpers de aserción personalizados para tests E2E
 */
import { expect } from '@playwright/test';
import { TaskMonitorResult, TaskInfo } from '../helpers';

/**
 * Verifica que el resultado del monitoreo de tarea sea exitoso
 */
export function expectTaskSuccess(result: TaskMonitorResult): void {
  if (result.error) {
    throw new Error(`Task failed: ${result.error}`);
  }

  expect(
    result.completed || result.contentReceived,
    'Task should be completed or have received content'
  ).toBe(true);
}

/**
 * Verifica que una tarea tenga contenido suficiente
 */
export function expectTaskHasContent(
  task: TaskInfo | null,
  minLength: number = 50
): void {
  expect(task, 'Task should exist').not.toBeNull();
  expect(
    task?.text?.length ?? 0,
    `Task should have at least ${minLength} characters`
  ).toBeGreaterThanOrEqual(minLength);
}

/**
 * Verifica que una tarea esté completada
 */
export function expectTaskCompleted(task: TaskInfo | null): void {
  expect(task, 'Task should exist').not.toBeNull();
  expect(
    task?.status,
    'Task should be completed or pending approval'
  ).toMatch(/completed|pending_approval/);
}

/**
 * Verifica que no haya logs problemáticos
 */
export function expectNoProblematicLogs(
  problematicLogs: string[],
  totalLogs: number
): void {
  expect(
    problematicLogs.length,
    `Should not have problematic logs. Found ${problematicLogs.length} problematic logs out of ${totalLogs} total logs`
  ).toBe(0);
}



