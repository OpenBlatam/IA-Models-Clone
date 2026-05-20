import { useCallback } from 'react';
import { useMusicStore } from '../../stores/music';
import type { Track } from '../../types/api';

export function useFavorites() {
  const favorites = useMusicStore((state) => state.favorites);
  const addFavorite = useMusicStore((state) => state.addFavorite);
  const removeFavorite = useMusicStore((state) => state.removeFavorite);

  const isFavorite = useCallback(
    (trackId: string) => {
      return favorites.some((track) => track.id === trackId);
    },
    [favorites]
  );

  const toggleFavorite = useCallback(
    (track: Track) => {
      if (isFavorite(track.id)) {
        removeFavorite(track.id);
      } else {
        addFavorite(track);
      }
    },
    [isFavorite, addFavorite, removeFavorite]
  );

  const favoriteCount = favorites.length;

  return {
    favorites,
    isFavorite,
    toggleFavorite,
    addFavorite,
    removeFavorite,
    favoriteCount,
  };
}


