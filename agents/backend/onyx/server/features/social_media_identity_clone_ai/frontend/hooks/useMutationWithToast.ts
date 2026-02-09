import { useMutation, UseMutationResult } from 'react-query';
import { showToast } from '@/lib/integrations/react-hot-toast';

interface UseMutationWithToastOptions<TData, TVariables> {
  mutationFn: (variables: TVariables) => Promise<TData>;
  successMessage?: string;
  errorMessage?: string;
  onSuccess?: (data: TData) => void;
  onError?: (error: unknown) => void;
}

export const useMutationWithToast = <TData, TVariables>({
  mutationFn,
  successMessage,
  errorMessage = 'An error occurred',
  onSuccess,
  onError,
}: UseMutationWithToastOptions<TData, TVariables>): UseMutationResult<TData, unknown, TVariables> => {
  return useMutation(mutationFn, {
    onSuccess: (data) => {
      if (successMessage) {
        showToast.success(successMessage);
      }
      if (onSuccess) {
        onSuccess(data);
      }
    },
    onError: (error) => {
      showToast.error(errorMessage);
      if (onError) {
        onError(error);
      }
    },
  });
};



