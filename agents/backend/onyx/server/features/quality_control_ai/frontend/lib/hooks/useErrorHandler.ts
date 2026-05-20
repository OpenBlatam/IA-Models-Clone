import { useCallback } from 'react';
import { useToast } from './useToast';
import { getErrorMessage } from '../utils/error';

export const useErrorHandler = () => {
  const toast = useToast();

  const handleError = useCallback(
    (error: unknown, defaultMessage = 'An error occurred'): void => {
      const message = getErrorMessage(error, defaultMessage);
      toast.error(message);
    },
    [toast]
  );

  const handleWarning = useCallback(
    (message: string): void => {
      toast.warning(message);
    },
    [toast]
  );

  return { handleError, handleWarning };
};

