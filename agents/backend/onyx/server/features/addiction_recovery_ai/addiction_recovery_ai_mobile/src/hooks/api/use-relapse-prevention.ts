import { useQuery, useMutation } from '@tanstack/react-query';
import { relapsePreventionApi } from '@/services/api';

export function useCheckRelapseRisk() {
  return useMutation({
    mutationFn: relapsePreventionApi.checkRelapseRisk,
  });
}

export function useTriggers(userId: string | null) {
  return useQuery({
    queryKey: ['triggers', userId],
    queryFn: () => relapsePreventionApi.getTriggers(userId!),
    enabled: !!userId,
  });
}

export function useCopingStrategies() {
  return useMutation({
    mutationFn: relapsePreventionApi.getCopingStrategies,
  });
}

