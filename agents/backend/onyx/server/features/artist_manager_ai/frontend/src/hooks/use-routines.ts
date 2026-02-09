import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { routinesApi } from '@/lib/api-client';
import type { RoutineTask, RoutineCompletion } from '@/types';

export const useRoutines = (
  artistId: string,
  params?: { routine_type?: string; today_only?: boolean }
) => {
  return useQuery<RoutineTask[]>({
    queryKey: ['routines', artistId, params],
    queryFn: () => routinesApi.getRoutines(artistId, params),
    enabled: !!artistId,
  });
};

export const useRoutine = (artistId: string, taskId: string) => {
  return useQuery<RoutineTask>({
    queryKey: ['routine', artistId, taskId],
    queryFn: () => routinesApi.getRoutine(artistId, taskId),
    enabled: !!artistId && !!taskId,
  });
};

export const usePendingRoutines = (artistId: string) => {
  return useQuery<RoutineTask[]>({
    queryKey: ['pending-routines', artistId],
    queryFn: () => routinesApi.getPendingRoutines(artistId),
    enabled: !!artistId,
  });
};

export const useCreateRoutine = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (routine: Omit<RoutineTask, 'id'>) => routinesApi.createRoutine(artistId, routine),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['routines', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useUpdateRoutine = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ taskId, updates }: { taskId: string; updates: Partial<RoutineTask> }) =>
      routinesApi.updateRoutine(artistId, taskId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['routines', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useDeleteRoutine = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (taskId: string) => routinesApi.deleteRoutine(artistId, taskId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['routines', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useCompleteRoutine = (artistId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ taskId, completion }: { taskId: string; completion: Partial<RoutineCompletion> }) =>
      routinesApi.completeRoutine(artistId, taskId, completion),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['routines', artistId] });
      queryClient.invalidateQueries({ queryKey: ['pending-routines', artistId] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', artistId] });
    },
  });
};

export const useCompletionRate = (artistId: string, taskId: string, days = 30) => {
  return useQuery<number>({
    queryKey: ['completion-rate', artistId, taskId, days],
    queryFn: () => routinesApi.getCompletionRate(artistId, taskId, days),
    enabled: !!artistId && !!taskId,
  });
};

