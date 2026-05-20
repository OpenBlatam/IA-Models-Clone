import { useCallback } from 'react';
import { handleError, AppError } from '@/utils/error-handling';
import { showToast } from '@/components';

export function useErrorHandler() {
  const handle = useCallback(
    (error: unknown, context?: Record<string, unknown>) => {
      const appError = handleError(error, context);

      // Show user-friendly message
      showToast({
        type: 'error',
        text1: 'Error',
        text2: appError.message,
      });

      return appError;
    },
    []
  );

  const handleSilent = useCallback(
    (error: unknown, context?: Record<string, unknown>) => {
      return handleError(error, context);
    },
    []
  );

  return {
    handle,
    handleSilent,
  };
}

