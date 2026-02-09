import { useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { handleApiError, handleApiSuccess } from '@/lib/error-handler';

interface UseMutationHandlerOptions<TData, TVariables> {
  mutationFn: (variables: TVariables) => Promise<TData>;
  onSuccess?: (data: TData) => void;
  onError?: (error: unknown) => void;
  successMessage?: string;
  errorMessage?: string;
  invalidateQueries?: string[][];
}

export const useMutationHandler = <TData, TVariables>({
  mutationFn,
  onSuccess,
  onError,
  successMessage,
  errorMessage,
  invalidateQueries,
}: UseMutationHandlerOptions<TData, TVariables>) => {
  const mutation = useMutation({
    mutationFn,
    onSuccess: (data) => {
      if (successMessage) {
        handleApiSuccess(successMessage);
      }
      if (onSuccess) {
        onSuccess(data);
      }
    },
    onError: (error) => {
      handleApiError(error, errorMessage);
      if (onError) {
        onError(error);
      }
    },
  });

  return mutation;
};

