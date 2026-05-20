import { useQuery, useMutation } from '@tanstack/react-query';
import { supportApi } from '@/services/api';

export function useCoachingSession() {
  return useMutation({
    mutationFn: supportApi.coachingSession,
  });
}

export function useMotivation(userId: string | null) {
  return useQuery({
    queryKey: ['motivation', userId],
    queryFn: () => supportApi.getMotivation(userId!),
    enabled: !!userId,
  });
}

export function useCelebrateMilestone() {
  return useMutation({
    mutationFn: supportApi.celebrateMilestone,
  });
}

export function useAchievements(userId: string | null) {
  return useQuery({
    queryKey: ['achievements', userId],
    queryFn: () => supportApi.getAchievements(userId!),
    enabled: !!userId,
  });
}

