import { useQuery } from '@tanstack/react-query';
import { dashboardApi } from '@/services/api';

export function useDashboard(userId: string | null) {
  return useQuery({
    queryKey: ['dashboard', userId],
    queryFn: () => dashboardApi.getDashboard(userId!),
    enabled: !!userId,
    refetchInterval: 60000,
  });
}

