import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { musicApiService } from '../services/music-api';
import type { AnalyzeTrackRequest, TrackAnalysis } from '../services/music-api';
import { useMusic } from '../stores/music';

export function useSearchTracks(query: string, limit = 10) {
  return useQuery({
    queryKey: ['tracks', 'search', query, limit],
    queryFn: () =>
      musicApiService.searchTracks({ query, limit }),
    enabled: query.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useAnalyzeTrack() {
  const queryClient = useQueryClient();
  const { setCurrentAnalysis } = useMusic();

  return useMutation({
    mutationFn: (request: AnalyzeTrackRequest) =>
      musicApiService.analyzeTrack(request),
    onSuccess: (data) => {
      setCurrentAnalysis(data);
      queryClient.setQueryData(
        ['analysis', data.track_basic_info.name],
        data
      );
    },
  });
}

export function useAnalyzeTrackById(trackId: string, includeCoaching = true) {
  const { setCurrentAnalysis } = useMusic();

  return useQuery({
    queryKey: ['analysis', trackId, includeCoaching],
    queryFn: () => musicApiService.analyzeTrackById(trackId, includeCoaching),
    enabled: trackId.length > 0,
    onSuccess: (data) => {
      setCurrentAnalysis(data);
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

export function useTrackRecommendations(trackId: string) {
  return useQuery({
    queryKey: ['recommendations', trackId],
    queryFn: () => musicApiService.getRecommendations(trackId),
    enabled: trackId.length > 0,
    staleTime: 15 * 60 * 1000, // 15 minutes
  });
}

export function useHealthCheck() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => musicApiService.healthCheck(),
    refetchInterval: 30000, // 30 seconds
    retry: 3,
  });
}

