import { useQuery } from '@tanstack/react-query';
import { analyticsService } from '@/services/analytics-service';

export function useAnalytics() {
  return useQuery({
    queryKey: ['analytics'],
    queryFn: () => analyticsService.getAnalytics(),
    staleTime: 60000, // 1 minute
  });
}

export function useRecommendations(
  scriptText: string,
  language = 'es',
  platform?: string,
  contentType = 'general',
  enabled = true
) {
  return useQuery({
    queryKey: ['recommendations', scriptText, language, platform, contentType],
    queryFn: () =>
      analyticsService.getRecommendations(scriptText, language, platform, contentType),
    enabled: enabled && scriptText.length > 10,
    staleTime: 300000, // 5 minutes
  });
}

export function useQuota() {
  return useQuery({
    queryKey: ['quota'],
    queryFn: () => analyticsService.getQuota(),
    staleTime: 30000, // 30 seconds
  });
}


