import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { musicApiService } from '../services/music-api';
import type {
  AnalyzeTrackRequest,
  CompareTracksRequest,
  ContextualRecommendationRequest,
  ArtistComparisonRequest,
} from '../services/music-api';
import type { TrackAnalysis, HistoryResponse, ExportResponse } from '../types/api';

export function useCoaching(request: AnalyzeTrackRequest) {
  return useMutation({
    mutationFn: () => musicApiService.getCoaching(request),
  });
}

export function useCompareTracks() {
  return useMutation({
    mutationFn: (request: CompareTracksRequest) =>
      musicApiService.compareTracks(request),
  });
}

export function useHistory(limit = 50) {
  return useQuery({
    queryKey: ['history', limit],
    queryFn: () => musicApiService.getHistory(limit),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
}

export function useHistoryStats() {
  return useQuery({
    queryKey: ['history', 'stats'],
    queryFn: () => musicApiService.getHistoryStats(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useExportAnalysis() {
  return useMutation({
    mutationFn: ({
      trackId,
      format,
      includeCoaching,
    }: {
      trackId: string;
      format?: 'json' | 'text' | 'markdown';
      includeCoaching?: boolean;
    }) =>
      musicApiService.exportAnalysis(
        trackId,
        format || 'json',
        includeCoaching
      ),
  });
}

export function useContextualRecommendations() {
  return useMutation({
    mutationFn: (request: ContextualRecommendationRequest) =>
      musicApiService.getContextualRecommendations(request),
  });
}

export function useTimeOfDayRecommendations(timeOfDay: string) {
  return useQuery({
    queryKey: ['recommendations', 'time-of-day', timeOfDay],
    queryFn: () => musicApiService.getTimeOfDayRecommendations(timeOfDay),
    enabled: timeOfDay.length > 0,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

export function useActivityRecommendations(activity: string) {
  return useQuery({
    queryKey: ['recommendations', 'activity', activity],
    queryFn: () => musicApiService.getActivityRecommendations(activity),
    enabled: activity.length > 0,
    staleTime: 10 * 60 * 1000,
  });
}

export function useMoodRecommendations(mood: string) {
  return useQuery({
    queryKey: ['recommendations', 'mood', mood],
    queryFn: () => musicApiService.getMoodRecommendations(mood),
    enabled: mood.length > 0,
    staleTime: 10 * 60 * 1000,
  });
}

export function useDiscoverSimilarArtists(artistId: string) {
  return useQuery({
    queryKey: ['discovery', 'similar-artists', artistId],
    queryFn: () => musicApiService.discoverSimilarArtists(artistId),
    enabled: artistId.length > 0,
    staleTime: 15 * 60 * 1000,
  });
}

export function useDiscoverUnderground() {
  return useQuery({
    queryKey: ['discovery', 'underground'],
    queryFn: () => musicApiService.discoverUnderground(),
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}

export function useDiscoverMoodTransition(fromMood: string, toMood: string) {
  return useQuery({
    queryKey: ['discovery', 'mood-transition', fromMood, toMood],
    queryFn: () =>
      musicApiService.discoverMoodTransition(fromMood, toMood),
    enabled: fromMood.length > 0 && toMood.length > 0,
    staleTime: 15 * 60 * 1000,
  });
}

export function useDiscoverFresh() {
  return useQuery({
    queryKey: ['discovery', 'fresh'],
    queryFn: () => musicApiService.discoverFresh(),
    staleTime: 30 * 60 * 1000,
  });
}

export function useCompareArtists() {
  return useMutation({
    mutationFn: (request: ArtistComparisonRequest) =>
      musicApiService.compareArtists(request),
  });
}

export function useArtistEvolution(artistId: string) {
  return useQuery({
    queryKey: ['artist', 'evolution', artistId],
    queryFn: () => musicApiService.getArtistEvolution(artistId),
    enabled: artistId.length > 0,
    staleTime: 30 * 60 * 1000,
  });
}

export function useTrendsPopularity(
  timeRange: 'short_term' | 'medium_term' | 'long_term' = 'medium_term'
) {
  return useQuery({
    queryKey: ['trends', 'popularity', timeRange],
    queryFn: () => musicApiService.getTrendsPopularity(timeRange),
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

export function useTrendsArtists(
  timeRange: 'short_term' | 'medium_term' | 'long_term' = 'medium_term'
) {
  return useQuery({
    queryKey: ['trends', 'artists', timeRange],
    queryFn: () => musicApiService.getTrendsArtists(timeRange),
    staleTime: 60 * 60 * 1000,
  });
}

