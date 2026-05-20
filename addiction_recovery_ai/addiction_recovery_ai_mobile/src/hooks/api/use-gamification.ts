import { useQuery } from '@tanstack/react-query';
import { gamificationApi } from '@/services/api';

export function usePoints(userId: string | null) {
  return useQuery({
    queryKey: ['points', userId],
    queryFn: () => gamificationApi.getPoints(userId!),
    enabled: !!userId,
  });
}

export function useGamificationAchievements(userId: string | null) {
  return useQuery({
    queryKey: ['gamification-achievements', userId],
    queryFn: () => gamificationApi.getGamificationAchievements(userId!),
    enabled: !!userId,
  });
}

