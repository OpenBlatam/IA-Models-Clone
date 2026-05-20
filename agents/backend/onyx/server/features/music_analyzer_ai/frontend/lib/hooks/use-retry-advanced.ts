/**
 * Custom hook for advanced retry.
 * Provides reactive advanced retry functionality.
 */

import { useCallback } from 'react';
import {
  retryAdvanced,
  createRetryFunction,
  AdvancedRetryOptions,
} from '../utils/retry-advanced';

/**
 * Custom hook for advanced retry.
 * Provides reactive advanced retry functionality.
 *
 * @param options - Retry options
 * @returns Retry function
 */
export function useRetryAdvanced(options: AdvancedRetryOptions = {}) {
  return useCallback(
    async <T>(fn: () => Promise<T>): Promise<T> => {
      return retryAdvanced(fn, options);
    },
    [options]
  );
}

/**
 * Custom hook for creating retry function with preset options.
 * Provides reactive retry function creation.
 *
 * @param options - Retry options
 * @returns Retry function
 */
export function useCreateRetryFunction(options: AdvancedRetryOptions) {
  return useCallback(
    () => createRetryFunction(options),
    [options]
  );
}

