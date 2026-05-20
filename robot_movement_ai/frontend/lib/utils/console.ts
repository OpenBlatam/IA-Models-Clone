/**
 * Console utilities with better formatting and filtering
 */

import { logger } from './logger';

/**
 * Enhanced console with better formatting
 */
export const consoleUtils = {
  /**
   * Log with context
   */
  logWithContext: (message: string, context?: Record<string, any>) => {
    if (process.env.NODE_ENV === 'development') {
      console.log(`[${new Date().toISOString()}] ${message}`, context || '');
    }
  },

  /**
   * Group logs
   */
  group: (label: string, fn: () => void) => {
    if (process.env.NODE_ENV === 'development') {
      console.group(label);
      fn();
      console.groupEnd();
    }
  },

  /**
   * Time operations
   */
  time: (label: string) => {
    if (process.env.NODE_ENV === 'development') {
      console.time(label);
    }
  },

  timeEnd: (label: string) => {
    if (process.env.NODE_ENV === 'development') {
      console.timeEnd(label);
    }
  },

  /**
   * Table display
   */
  table: (data: any) => {
    if (process.env.NODE_ENV === 'development') {
      console.table(data);
    }
  },

  /**
   * Trace with stack
   */
  trace: (message?: string) => {
    if (process.env.NODE_ENV === 'development') {
      console.trace(message);
    }
  },
};

/**
 * Replace console methods in development
 */
export function setupConsoleUtils() {
  if (process.env.NODE_ENV === 'development' && typeof window !== 'undefined') {
    // Use logger instead of console for better tracking
    (window as any).__consoleUtils = consoleUtils;
  }
}



