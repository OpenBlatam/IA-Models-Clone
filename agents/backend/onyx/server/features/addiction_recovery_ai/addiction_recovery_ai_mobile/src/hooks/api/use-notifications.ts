import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { notificationsApi } from '@/services/api';

export function useNotifications(userId: string | null) {
  return useQuery({
    queryKey: ['notifications', userId],
    queryFn: () => notificationsApi.getNotifications(userId!),
    enabled: !!userId,
    refetchInterval: 30000,
  });
}

export function useMarkNotificationRead() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: notificationsApi.markNotificationRead,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
    },
  });
}

export function useReminders(userId: string | null) {
  return useQuery({
    queryKey: ['reminders', userId],
    queryFn: () => notificationsApi.getReminders(userId!),
    enabled: !!userId,
  });
}

