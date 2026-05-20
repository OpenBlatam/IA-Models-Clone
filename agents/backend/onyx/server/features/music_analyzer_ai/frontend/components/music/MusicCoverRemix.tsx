'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Music2, Loader2, Play } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicCoverRemixProps {
  trackId: string;
  onTrackSelect?: (track: Track) => void;
}

export function MusicCoverRemix({ trackId, onTrackSelect }: MusicCoverRemixProps) {
  const [type, setType] = useState<'covers' | 'remixes'>('covers');

  const { data: coversRemixes, isLoading } = useQuery({
    queryKey: ['covers-remixes', trackId, type],
    queryFn: async () => {
      if (type === 'covers') {
        return await musicApiService.findCoversRemixes?.(trackId) || { covers: [] };
      } else {
        return await musicApiService.findCoversRemixes?.(trackId) || { remixes: [] };
      }
    },
    enabled: !!trackId,
  });

  const items = type === 'covers' ? coversRemixes?.covers || [] : coversRemixes?.remixes || [];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Music2 className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">
            {type === 'covers' ? 'Covers' : 'Remixes'}
          </h3>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setType('covers')}
            className={`px-3 py-1 rounded-lg text-sm transition-colors ${
              type === 'covers'
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            Covers
          </button>
          <button
            onClick={() => setType('remixes')}
            className={`px-3 py-1 rounded-lg text-sm transition-colors ${
              type === 'remixes'
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            Remixes
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-purple-300 animate-spin" />
        </div>
      ) : items.length === 0 ? (
        <div className="text-center py-12">
          <Music2 className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No se encontraron {type === 'covers' ? 'covers' : 'remixes'}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
          {items.map((item: any, idx: number) => (
            <button
              key={item.id || idx}
              onClick={() => onTrackSelect?.(item)}
              className="p-4 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
            >
              <div className="w-full aspect-square rounded bg-purple-500 flex items-center justify-center mb-3 relative group">
                <Music2 className="w-12 h-12 text-white" />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <Play className="w-8 h-8 text-white" />
                </div>
              </div>
              <p className="text-white font-medium truncate text-sm mb-1">
                {item.name || item.track_name || 'Canción desconocida'}
              </p>
              <p className="text-xs text-gray-300 truncate">
                {Array.isArray(item.artists) ? item.artists.join(', ') : item.artists || 'Artista'}
              </p>
              {item.similarity_score && (
                <p className="text-xs text-purple-300 mt-1">
                  Similitud: {Math.round(item.similarity_score * 100)}%
                </p>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


