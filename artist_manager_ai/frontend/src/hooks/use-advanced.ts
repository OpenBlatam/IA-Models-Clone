import { useQuery, useMutation } from '@tanstack/react-query';
import { advancedApi } from '@/lib/api-client';
import type { Alert, CalendarEvent, Prediction, SearchRequest } from '@/types';

export const useAlerts = (artistId: string) => {
  return useQuery<Alert[]>({
    queryKey: ['alerts', artistId],
    queryFn: () => advancedApi.getAlerts(artistId),
    enabled: !!artistId,
    refetchInterval: 60000,
  });
};

export const useSearchEvents = (artistId: string) => {
  return useMutation({
    mutationFn: (searchRequest: SearchRequest) => advancedApi.searchEvents(artistId, searchRequest),
  });
};

export const usePredictEventDuration = (artistId: string) => {
  return useMutation({
    mutationFn: (eventType: string) => advancedApi.predictEventDuration(artistId, eventType),
  });
};

export const usePredictRoutineCompletion = (artistId: string) => {
  return useMutation({
    mutationFn: ({ routineId, days }: { routineId: string; days?: number }) =>
      advancedApi.predictRoutineCompletion(artistId, routineId, days),
  });
};

export const useSyncCalendar = (artistId: string) => {
  return useMutation({
    mutationFn: ({
      provider,
      credentials,
    }: {
      provider: string;
      credentials: Record<string, unknown>;
    }) => advancedApi.syncCalendar(artistId, provider, credentials),
  });
};

