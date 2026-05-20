/**
 * Custom hooks for React Query.
 * Provides optimized hooks for common data fetching patterns.
 */

import { useQuery, useMutation, useQueryClient, type UseQueryOptions, type UseMutationOptions } from '@tanstack/react-query';
import { QUERY_KEYS } from '@/lib/constants';
import { getErrorMessage } from '@/lib/errors';
import toast from 'react-hot-toast';

/**
 * Options for query hooks with error handling.
 */
interface QueryHookOptions<TData, TError = Error> extends Omit<UseQueryOptions<TData, TError>, 'queryKey' | 'queryFn'> {
  showErrorToast?: boolean;
  errorMessage?: string;
}

/**
 * Options for mutation hooks with error handling.
 */
interface MutationHookOptions<TData, TVariables, TError = Error>
  extends Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'> {
  showSuccessToast?: boolean;
  showErrorToast?: boolean;
  successMessage?: string;
  errorMessage?: string;
  invalidateQueries?: string[][];
}

/**
 * Enhanced useQuery with automatic error handling.
 */
export function useQueryWithErrorHandling<TData, TError = Error>(
  queryKey: unknown[],
  queryFn: () => Promise<TData>,
  options: QueryHookOptions<TData, TError> = {}
) {
  const { showErrorToast = true, errorMessage, ...queryOptions } = options;

  return useQuery<TData, TError>({
    queryKey,
    queryFn,
    ...queryOptions,
    onError: (error) => {
      if (showErrorToast) {
        const message = errorMessage || getErrorMessage(error);
        toast.error(message);
      }
      options.onError?.(error);
    },
  });
}

/**
 * Enhanced useMutation with automatic error and success handling.
 */
export function useMutationWithHandling<TData, TVariables, TError = Error>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options: MutationHookOptions<TData, TVariables, TError> = {}
) {
  const {
    showSuccessToast = false,
    showErrorToast = true,
    successMessage = 'Operation completed successfully',
    errorMessage,
    invalidateQueries = [],
    ...mutationOptions
  } = options;

  const queryClient = useQueryClient();

  return useMutation<TData, TError, TVariables>({
    mutationFn,
    ...mutationOptions,
    onSuccess: (data, variables, context) => {
      if (showSuccessToast) {
        toast.success(successMessage);
      }

      // Invalidate specified queries
      invalidateQueries.forEach((queryKey) => {
        queryClient.invalidateQueries({ queryKey });
      });

      options.onSuccess?.(data, variables, context);
    },
    onError: (error, variables, context) => {
      if (showErrorToast) {
        const message = errorMessage || getErrorMessage(error);
        toast.error(message);
      }
      options.onError?.(error, variables, context);
    },
  });
}

/**
 * Hook for optimistic updates.
 * Updates cache optimistically before mutation completes.
 */
export function useOptimisticMutation<TData, TVariables, TError = Error>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options: MutationHookOptions<TData, TVariables, TError> & {
    queryKey: unknown[];
    optimisticUpdate: (variables: TVariables) => TData;
    rollback?: (previousData: TData | undefined) => void;
  }
) {
  const {
    queryKey,
    optimisticUpdate,
    rollback,
    ...mutationOptions
  } = options;

  const queryClient = useQueryClient();

  return useMutationWithHandling(mutationFn, {
    ...mutationOptions,
    onMutate: async (variables) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey });

      // Snapshot previous value
      const previousData = queryClient.getQueryData<TData>(queryKey);

      // Optimistically update
      queryClient.setQueryData<TData>(queryKey, optimisticUpdate(variables));

      return { previousData };
    },
    onError: (error, variables, context) => {
      // Rollback on error
      if (context?.previousData !== undefined) {
        if (rollback) {
          rollback(context.previousData);
        } else {
          queryClient.setQueryData(queryKey, context.previousData);
        }
      }
      mutationOptions.onError?.(error, variables, context);
    },
    onSettled: () => {
      // Refetch after mutation
      queryClient.invalidateQueries({ queryKey });
      mutationOptions.onSettled?.();
    },
  });
}

