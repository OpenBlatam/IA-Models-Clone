import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { queryKeys } from '@/lib/query-keys';
import { calendarApi, dashboardApi } from '@/lib/api-client';

export const usePrefetch = () => {
  const queryClient = useQueryClient();

  const prefetchDashboard = useCallback(
    async (artistId: string) => {
      await queryClient.prefetchQuery({
        queryKey: queryKeys.dashboard(artistId),
        queryFn: () => dashboardApi.getDashboard(artistId),
      });
    },
    [queryClient]
  );

  const prefetchEvents = useCallback(
    async (artistId: string, params?: Record<string, any>) => {
      await queryClient.prefetchQuery({
        queryKey: queryKeys.events(artistId, params),
        queryFn: () => calendarApi.getEvents(artistId, params),
      });
    },
    [queryClient]
  );

  return {
    prefetchDashboard,
    prefetchEvents,
  };
};

