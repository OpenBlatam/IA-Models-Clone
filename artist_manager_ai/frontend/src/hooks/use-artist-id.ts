import { useState, useEffect } from 'react';

const ARTIST_ID_KEY = 'artist-manager-artist-id';
const DEFAULT_ARTIST_ID = 'artist_001';

export const useArtistId = () => {
  const [artistId, setArtistIdState] = useState<string>(() => {
    if (typeof window === 'undefined') {
      return DEFAULT_ARTIST_ID;
    }
    const stored = localStorage.getItem(ARTIST_ID_KEY);
    return stored || DEFAULT_ARTIST_ID;
  });

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(ARTIST_ID_KEY, artistId);
    }
  }, [artistId]);

  const setArtistId = (id: string) => {
    setArtistIdState(id);
  };

  return [artistId, setArtistId] as const;
};

