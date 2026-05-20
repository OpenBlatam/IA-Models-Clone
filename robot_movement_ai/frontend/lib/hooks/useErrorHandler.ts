import { useCallback } from 'react';
import { handleApiError, handleError } from '@/lib/utils/error-handler';
import { logger } from '@/lib/utils/logger';

export interface UseErrorHandlerOptions {
  showToast?: boolean;
  logError?: boolean;
  onError?: (error: unknown) => void;
}

/**
 * Hook for consistent error handling
 */
export function useErrorHandler(options: UseErrorHandlerOptions = {}) {
  const { showToast = true, logError = true, onError } = options;

  const handleError = useCallback(
    (error: unknown, context?: string) => {
      if (logError) {
        logger.error('Error occurred', { error, context });
      }

      if (showToast) {
        if (error instanceof Error) {
          handleApiError(error, context || error.message);
        } else {
          handleError(error, context || 'Ha ocurrido un error');
        }
      }

      if (onError) {
        onError(error);
      }
    },
    [showToast, logError, onError]
  );

  const handleApiErrorWrapper = useCallback(
    (error: unknown, message?: string) => {
      handleApiError(error, message);
      if (logError) {
        logger.error('API error', { error, message });
      }
      if (onError) {
        onError(error);
      }
    },
    [logError, onError]
  );

  return {
    handleError,
    handleApiError: handleApiErrorWrapper,
  };
}
