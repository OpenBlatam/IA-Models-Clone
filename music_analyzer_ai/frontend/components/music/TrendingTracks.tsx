'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { TrendingUp, Music, ArrowUp } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface TrendingTracksProps {
  onTrackSelect?: (track: Track) => void;
}

export function TrendingTracks({ onTrackSelect }: TrendingTracksProps) {
  const { data: trends } = useQuery({
    queryKey: ['trends'],
    queryFn: () => musicApiService.getTrends?.(10) || Promise.resolve({ trends: [] }),
  });

  const trendingTracks = trends?.trends || [];

  if (trendingTracks.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <p className="text-gray-400 text-sm">No hay tendencias disponibles</p>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Tendencias</h3>
      </div>

      <div className="space-y-3">
        {trendingTracks.slice(0, 10).map((track: any, idx: number) => (
          <button
            key={track.id || idx}
            onClick={() => onTrackSelect?.(track)}
            className="w-full flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
          >
            <div className="relative">
              <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                <Music className="w-6 h-6 text-white" />
              </div>
              {idx < 3 && (
                <div className="absolute -top-1 -right-1 w-5 h-5 bg-yellow-400 rounded-full flex items-center justify-center">
                  <ArrowUp className="w-3 h-3 text-gray-900" />
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-white font-medium truncate">{track.name || 'Canción desconocida'}</p>
              <p className="text-sm text-gray-300 truncate">
                {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists || 'Artista desconocido'}
              </p>
            </div>
            <div className="text-right">
              <span className="text-sm text-purple-300 font-medium">#{idx + 1}</span>
              {track.trend_score && (
                <p className="text-xs text-gray-400">
                  +{Math.round(track.trend_score * 100)}%
                </p>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}


