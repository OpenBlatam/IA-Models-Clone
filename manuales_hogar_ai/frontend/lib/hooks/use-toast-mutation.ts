import { useMutationWithInvalidation } from './use-mutation-with-invalidation';
import { showErrorToast, showSuccessToast } from '../utils/error-handler';

interface UseToastMutationOptions<TData, TError, TVariables> {
  mutationFn: (variables: TVariables) => Promise<TData>;
  invalidateQueries?: string[][];
  successMessage?: string | ((data: TData) => string);
  onSuccess?: (data: TData, variables: TVariables) => void;
  onError?: (error: TError, variables: TVariables) => void;
  showSuccessToast?: boolean;
  showErrorToast?: boolean;
}

export const useToastMutation = <TData, TError, TVariables>({
  mutationFn,
  invalidateQueries = [],
  successMessage,
  onSuccess,
  onError,
  showSuccessToast: showSuccess = true,
  showErrorToast: showError = true,
}: UseToastMutationOptions<TData, TError, TVariables>) => {
  return useMutationWithInvalidation<TData, TError, TVariables>({
    mutationFn,
    invalidateQueries,
    onSuccess: (data, variables) => {
      if (showSuccess && successMessage) {
        const message = typeof successMessage === 'function' 
          ? successMessage(data) 
          : successMessage;
        showSuccessToast(message);
      }
      onSuccess?.(data, variables);
    },
    onError: (error, variables) => {
      if (showError) {
        showErrorToast(error);
      }
      onError?.(error, variables);
    },
  });
};

