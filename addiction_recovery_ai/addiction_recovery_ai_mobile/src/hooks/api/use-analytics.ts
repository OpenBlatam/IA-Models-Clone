import { useQuery } from '@tanstack/react-query';
import { analyticsApi } from '@/services/api';

export function useAnalytics(userId: string | null) {
  return useQuery({
    queryKey: ['analytics', userId],
    queryFn: () => analyticsApi.getAnalytics(userId!),
    enabled: !!userId,
  });
}

export function useAdvancedAnalytics(userId: string | null) {
  return useQuery({
    queryKey: ['advanced-analytics', userId],
    queryFn: () => analyticsApi.getAdvancedAnalytics(userId!),
    enabled: !!userId,
  });
}

export function useInsights(userId: string | null) {
  return useQuery({
    queryKey: ['insights', userId],
    queryFn: () => analyticsApi.getInsights(userId!),
    enabled: !!userId,
  });
}

