import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { progressApi } from '@/services/api';

export function useProgress(userId: string | null) {
  return useQuery({
    queryKey: ['progress', userId],
    queryFn: () => progressApi.getProgress(userId!),
    enabled: !!userId,
    refetchInterval: 60000, // Refetch every minute
  });
}

export function useStats(userId: string | null) {
  return useQuery({
    queryKey: ['stats', userId],
    queryFn: () => progressApi.getStats(userId!),
    enabled: !!userId,
  });
}

export function useLogEntry() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: progressApi.logEntry,
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['progress', variables.user_id] });
      queryClient.invalidateQueries({ queryKey: ['stats', variables.user_id] });
    },
  });
}

export function useTimeline(userId: string | null) {
  return useQuery({
    queryKey: ['timeline', userId],
    queryFn: () => progressApi.getTimeline(userId!),
    enabled: !!userId,
  });
}

