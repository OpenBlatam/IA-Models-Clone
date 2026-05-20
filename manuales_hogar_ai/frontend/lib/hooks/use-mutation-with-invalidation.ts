import { useMutation, useQueryClient, UseMutationOptions } from '@tanstack/react-query';

interface MutationWithInvalidationOptions<TData, TError, TVariables> {
  mutationFn: (variables: TVariables) => Promise<TData>;
  invalidateQueries?: string[][];
  onSuccess?: (data: TData, variables: TVariables) => void;
  onError?: (error: TError, variables: TVariables) => void;
}

export const useMutationWithInvalidation = <TData, TError, TVariables>({
  mutationFn,
  invalidateQueries = [],
  onSuccess,
  onError,
}: MutationWithInvalidationOptions<TData, TError, TVariables>) => {
  const queryClient = useQueryClient();

  return useMutation<TData, TError, TVariables>({
    mutationFn,
    onSuccess: (data, variables) => {
      invalidateQueries.forEach((queryKey) => {
        queryClient.invalidateQueries({ queryKey });
      });
      onSuccess?.(data, variables);
    },
    onError,
  });
};

