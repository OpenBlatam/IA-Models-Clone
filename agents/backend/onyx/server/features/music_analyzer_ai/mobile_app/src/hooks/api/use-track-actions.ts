import { useCallback } from 'react';
import { useRouter } from 'expo-router';
import { useMusicStore } from '../../stores/music';
import { useFavorites } from './use-favorites';
import { useRecentSearches } from './use-recent-searches';
import type { Track, TrackAnalysis } from '../../types/api';

export function useTrackActions() {
  const router = useRouter();
  const { toggleFavorite, isFavorite } = useFavorites();
  const { addRecentSearch } = useRecentSearches();
  const setCurrentAnalysis = useMusicStore(
    (state) => state.setCurrentAnalysis
  );

  const handleTrackPress = useCallback(
    (track: Track) => {
      addRecentSearch(track);
      router.push({
        pathname: '/analysis',
        params: { trackId: track.id },
      });
    },
    [router, addRecentSearch]
  );

  const handleTrackFavorite = useCallback(
    (track: Track) => {
      toggleFavorite(track);
    },
    [toggleFavorite]
  );

  const handleViewAnalysis = useCallback(
    (analysis: TrackAnalysis) => {
      setCurrentAnalysis(analysis);
      router.push('/analysis');
    },
    [router, setCurrentAnalysis]
  );

  const handleViewRecommendations = useCallback(
    (trackId: string) => {
      router.push({
        pathname: '/recommendations',
        params: { trackId },
      });
    },
    [router]
  );

  return {
    handleTrackPress,
    handleTrackFavorite,
    handleViewAnalysis,
    handleViewRecommendations,
    isFavorite,
  };
}


