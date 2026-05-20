import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { calendarApi } from '@/lib/api-client';
import { queryKeys } from '@/lib/query-keys';
import type { CalendarEvent } from '@/types';

export const useEvents = (artistId: string, params?: { date?: string; days?: number }) => {
  return useQuery<CalendarEvent[]>({
    queryKey: queryKeys.events(artistId, params),
    queryFn: () => calendarApi.getEvents(artistId, params),
    enabled: !!artistId,
  });
};

export const useEvent = (artistId: string, eventId: string) => {
  return useQuery<CalendarEvent>({
    queryKey: queryKeys.event(artistId, eventId),
    queryFn: () => calendarApi.getEvent(artistId, eventId),
    enabled: !!artistId && !!eventId,
  });
};

export const useCreateEvent = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (event: Omit<CalendarEvent, 'id'>) => calendarApi.createEvent(artistId, event),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.events(artistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.dashboard(artistId) });
    },
  });
};

export const useUpdateEvent = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ eventId, updates }: { eventId: string; updates: Partial<CalendarEvent> }) =>
      calendarApi.updateEvent(artistId, eventId, updates),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.events(artistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.event(artistId, variables.eventId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.dashboard(artistId) });
    },
  });
};

export const useDeleteEvent = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (eventId: string) => calendarApi.deleteEvent(artistId, eventId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.events(artistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.dashboard(artistId) });
    },
  });
};

export const useWardrobeRecommendation = (artistId: string, eventId: string) => {
  return useQuery({
    queryKey: queryKeys.wardrobeRecommendation(artistId, eventId),
    queryFn: () => calendarApi.getWardrobeRecommendation(artistId, eventId),
    enabled: !!artistId && !!eventId,
  });
};

