import { useState, useCallback } from 'react';
import { handleError, logError, AppError } from '../utils/errorHandler';
import { useToastContext } from '../context/ToastContext';

export const useErrorHandler = () => {
  const { showError } = useToastContext();

  const handleErrorWithToast = useCallback(
    (error: any, context?: string) => {
      logError(error, context);
      const message = handleError(error);
      showError(message);
    },
    [showError]
  );

  const handleErrorSilently = useCallback((error: any, context?: string) => {
    logError(error, context);
  }, []);

  return {
    handleError: handleErrorWithToast,
    handleErrorSilently,
    AppError,
  };
};

