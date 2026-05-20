import { useAppStore } from '@/lib/store';

export const useArtist = () => {
  const artistId = useAppStore((state) => state.artistId);
  const setArtistId = useAppStore((state) => state.setArtistId);

  return { artistId, setArtistId };
};

