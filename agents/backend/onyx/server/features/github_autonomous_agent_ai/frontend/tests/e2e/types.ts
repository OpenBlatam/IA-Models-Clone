/**
 * Type definitions for E2E tests
 */

/**
 * Task status type
 */
export type TaskStatus = 
  | 'pending'
  | 'completed'
  | 'pending_approval'
  | 'failed'
  | '';

/**
 * Task data from API
 */
export interface TaskData {
  id: string;
  status: TaskStatus;
  streamingContent?: string;
  instruction?: string;
  createdAt?: string;
  updatedAt?: string;
}

/**
 * Task card element data
 */
export interface TaskCardData {
  text: string | null;
  status: TaskStatus;
  element: any; // Playwright ElementHandle
}

/**
 * Polling result
 */
export interface PollingResult {
  taskCompleted: boolean;
  contentReceived: boolean;
  taskData?: TaskCardData;
  error?: Error;
}

/**
 * Console log entry
 */
export interface ConsoleLogEntry {
  text: string;
  timestamp: number;
  type: 'log' | 'error' | 'warn' | 'info';
}



