/**
 * Custom hook for error handling.
 * Provides convenient error handling with toast notifications.
 */

import { useCallback } from 'react';
import { handleError, type ErrorHandlerOptions } from '../utils/error-handler';
import { toast } from 'react-hot-toast';

/**
 * Return type for useErrorHandler hook.
 */
export interface UseErrorHandlerReturn {
  handleError: (error: unknown, options?: ErrorHandlerOptions) => string;
  handleErrorWithToast: (error: unknown, options?: Omit<ErrorHandlerOptions, 'showToast'>) => string;
}

/**
 * Custom hook for error handling.
 * Provides convenient error handling with optional toast notifications.
 *
 * @param defaultOptions - Default error handler options
 * @returns Error handler functions
 */
export function useErrorHandler(
  defaultOptions: ErrorHandlerOptions = {}
): UseErrorHandlerReturn {
  const handleErrorWithOptions = useCallback(
    (error: unknown, options: ErrorHandlerOptions = {}) => {
      const mergedOptions = { ...defaultOptions, ...options };
      return handleError(error, mergedOptions);
    },
    [defaultOptions]
  );

  const handleErrorWithToast = useCallback(
    (error: unknown, options: Omit<ErrorHandlerOptions, 'showToast'> = {}) => {
      const errorMessage = handleError(error, {
        ...defaultOptions,
        ...options,
        showToast: true,
      });
      return errorMessage;
    },
    [defaultOptions]
  );

  return {
    handleError: handleErrorWithOptions,
    handleErrorWithToast,
  };
}

