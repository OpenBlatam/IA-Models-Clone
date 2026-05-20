import { useCallback } from 'react';
import { showToast } from '@/lib/integrations/react-hot-toast';
import { getErrorMessage } from '@/lib/utils';

interface UseErrorHandlerOptions {
  showToast?: boolean;
  defaultMessage?: string;
  onError?: (error: unknown) => void;
}

export const useErrorHandler = (options: UseErrorHandlerOptions = {}) => {
  const { showToast: showToastOnError = true, defaultMessage = 'An error occurred', onError } = options;

  const handleError = useCallback(
    (error: unknown, customMessage?: string) => {
      const message = customMessage || getErrorMessage(error) || defaultMessage;

      if (showToastOnError) {
        showToast.error(message);
      }

      if (onError) {
        onError(error);
      }

      console.error('Error handled:', error);
    },
    [showToastOnError, defaultMessage, onError]
  );

  return { handleError };
};



