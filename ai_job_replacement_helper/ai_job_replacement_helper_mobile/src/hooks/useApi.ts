import { useQuery, useMutation, useQueryClient, UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { useAuthStore } from '@/store/authStore';

export function useAuthenticatedQuery<TData, TError = Error>(
  options: Omit<UseQueryOptions<TData, TError>, 'enabled'> & {
    queryFn: (userId: string) => Promise<TData>;
  }
) {
  const { user } = useAuthStore();
  const userId = user?.id;

  return useQuery<TData, TError>({
    ...options,
    queryKey: [...(options.queryKey || []), userId],
    queryFn: () => {
      if (!userId) {
        throw new Error('User not authenticated');
      }
      return options.queryFn(userId);
    },
    enabled: options.enabled !== false && !!userId,
  });
}

export function useAuthenticatedMutation<TData, TVariables, TError = Error>(
  options: UseMutationOptions<TData, TError, TVariables> & {
    mutationFn: (userId: string, variables: TVariables) => Promise<TData>;
  }
) {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();
  const userId = user?.id;

  return useMutation<TData, TError, TVariables>({
    ...options,
    mutationFn: (variables: TVariables) => {
      if (!userId) {
        throw new Error('User not authenticated');
      }
      return options.mutationFn(userId, variables);
    },
    onSuccess: (data, variables, context) => {
      if (options.onSuccess) {
        options.onSuccess(data, variables, context);
      }
      // Invalidate relevant queries
      queryClient.invalidateQueries();
    },
  });
}


