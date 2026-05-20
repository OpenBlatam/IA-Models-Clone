'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Compass, Music, TrendingUp, Sparkles } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicDiscoveryProps {
  onTrackSelect?: (track: Track) => void;
}

export function MusicDiscovery({ onTrackSelect }: MusicDiscoveryProps) {
  const [discoveryType, setDiscoveryType] = useState<'underground' | 'similar' | 'trending'>('underground');

  const { data: underground } = useQuery({
    queryKey: ['underground-tracks'],
    queryFn: () => musicApiService.getUndergroundTracks?.(20) || Promise.resolve({ tracks: [] }),
    enabled: discoveryType === 'underground',
  });

  const { data: trends } = useQuery({
    queryKey: ['trends'],
    queryFn: () => musicApiService.getTrends?.(20) || Promise.resolve({ trends: [] }),
    enabled: discoveryType === 'trending',
  });

  const getTracks = () => {
    switch (discoveryType) {
      case 'underground':
        return underground?.tracks || [];
      case 'trending':
        return trends?.trends || [];
      default:
        return [];
    }
  };

  const tracks = getTracks();

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Compass className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Descubrimiento Musical</h2>
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setDiscoveryType('underground')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            discoveryType === 'underground'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-gray-300 hover:bg-white/20'
          }`}
        >
          Underground
        </button>
        <button
          onClick={() => setDiscoveryType('trending')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            discoveryType === 'trending'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-gray-300 hover:bg-white/20'
          }`}
        >
          Tendencias
        </button>
      </div>

      {tracks.length === 0 ? (
        <div className="text-center py-12">
          <Music className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No hay canciones disponibles</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
          {tracks.map((track: any, idx: number) => (
            <button
              key={track.id || idx}
              onClick={() => onTrackSelect?.(track)}
              className="p-4 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
            >
              <div className="w-full aspect-square rounded bg-purple-500 flex items-center justify-center mb-3">
                <Music className="w-12 h-12 text-white" />
              </div>
              <p className="text-white font-medium truncate text-sm mb-1">
                {track.name || track.track_name || 'Canción desconocida'}
              </p>
              <p className="text-xs text-gray-300 truncate">
                {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists || 'Artista'}
              </p>
              {discoveryType === 'trending' && track.trend_score && (
                <div className="flex items-center gap-1 mt-2">
                  <TrendingUp className="w-3 h-3 text-green-400" />
                  <span className="text-xs text-green-400">
                    +{Math.round(track.trend_score * 100)}%
                  </span>
                </div>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


