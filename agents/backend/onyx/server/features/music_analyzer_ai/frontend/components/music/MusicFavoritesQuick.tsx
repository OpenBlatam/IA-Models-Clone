'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Heart, Music, Plus } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicFavoritesQuickProps {
  onTrackSelect?: (track: Track) => void;
}

export function MusicFavoritesQuick({ onTrackSelect }: MusicFavoritesQuickProps) {
  const { data: favorites } = useQuery({
    queryKey: ['favorites', 'user123'],
    queryFn: () => musicApiService.getFavorites('user123') || Promise.resolve({ favorites: [] }),
  });

  const favoriteTracks = favorites?.favorites || [];

  if (favoriteTracks.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <Heart className="w-12 h-12 text-gray-500 mx-auto mb-2" />
        <p className="text-gray-400 text-sm">No hay favoritos aún</p>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Heart className="w-5 h-5 text-red-400" />
        <h3 className="text-lg font-semibold text-white">Favoritos Rápidos</h3>
        <span className="text-sm text-gray-400">({favoriteTracks.length})</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {favoriteTracks.slice(0, 8).map((track: any, idx: number) => (
          <button
            key={track.id || idx}
            onClick={() => onTrackSelect?.(track)}
            className="p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left group"
          >
            <div className="w-full aspect-square rounded bg-red-500/20 flex items-center justify-center mb-2 relative group-hover:bg-red-500/30 transition-colors">
              <Music className="w-8 h-8 text-red-400" />
              <div className="absolute top-1 right-1">
                <Heart className="w-4 h-4 text-red-400 fill-red-400" />
              </div>
            </div>
            <p className="text-white font-medium truncate text-xs mb-1">
              {track.track_name || track.name || 'Canción'}
            </p>
            <p className="text-gray-400 truncate text-xs">
              {Array.isArray(track.artists) ? track.artists[0] : track.artists || 'Artista'}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}


