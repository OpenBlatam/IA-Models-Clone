import { useQuery } from '@tanstack/react-query';
import { dashboardApi } from '@/lib/api-client';
import type { DashboardData, DailySummary } from '@/types';

export const useDashboard = (artistId: string) => {
  return useQuery<DashboardData>({
    queryKey: ['dashboard', artistId],
    queryFn: () => dashboardApi.getDashboard(artistId),
    enabled: !!artistId,
  });
};

export const useDailySummary = (artistId: string) => {
  return useQuery<DailySummary>({
    queryKey: ['daily-summary', artistId],
    queryFn: () => dashboardApi.getDailySummary(artistId),
    enabled: !!artistId,
  });
};

