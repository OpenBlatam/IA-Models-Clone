'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Library, Grid, List, Music } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicLibraryProps {
  onTrackSelect?: (track: Track) => void;
}

export function MusicLibrary({ onTrackSelect }: MusicLibraryProps) {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list');
  const [filter, setFilter] = useState<'all' | 'favorites' | 'recent'>('all');

  const { data: favorites } = useQuery({
    queryKey: ['favorites', 'user123'],
    queryFn: () => musicApiService.getFavorites('user123') || Promise.resolve({ favorites: [] }),
    enabled: filter === 'favorites',
  });

  const { data: history } = useQuery({
    queryKey: ['history', 'user123'],
    queryFn: () => musicApiService.getHistory('user123', 50) || Promise.resolve({ history: [] }),
    enabled: filter === 'recent',
  });

  const getTracks = () => {
    switch (filter) {
      case 'favorites':
        return favorites?.favorites || [];
      case 'recent':
        return history?.history || [];
      default:
        return [];
    }
  };

  const tracks = getTracks();

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Library className="w-6 h-6 text-purple-300" />
          <h2 className="text-2xl font-semibold text-white">Mi Biblioteca</h2>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex bg-white/10 rounded-lg p-1">
            <button
              onClick={() => setFilter('all')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                filter === 'all'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              Todos
            </button>
            <button
              onClick={() => setFilter('favorites')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                filter === 'favorites'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              Favoritos
            </button>
            <button
              onClick={() => setFilter('recent')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                filter === 'recent'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              Recientes
            </button>
          </div>
          <div className="flex bg-white/10 rounded-lg p-1">
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded transition-colors ${
                viewMode === 'list'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              <List className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded transition-colors ${
                viewMode === 'grid'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              <Grid className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {tracks.length === 0 ? (
        <div className="text-center py-12">
          <Music className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No hay canciones en esta sección</p>
        </div>
      ) : viewMode === 'list' ? (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {tracks.map((track: any, idx: number) => (
            <button
              key={track.id || idx}
              onClick={() => onTrackSelect?.(track)}
              className="w-full flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
            >
              <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                <Music className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">
                  {track.track_name || track.name || 'Canción desconocida'}
                </p>
                <p className="text-sm text-gray-300 truncate">
                  {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists || 'Artista desconocido'}
                </p>
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-h-96 overflow-y-auto">
          {tracks.map((track: any, idx: number) => (
            <button
              key={track.id || idx}
              onClick={() => onTrackSelect?.(track)}
              className="p-4 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
            >
              <div className="w-full aspect-square rounded bg-purple-500 flex items-center justify-center mb-2">
                <Music className="w-8 h-8 text-white" />
              </div>
              <p className="text-white font-medium truncate text-sm">
                {track.track_name || track.name || 'Canción desconocida'}
              </p>
              <p className="text-xs text-gray-300 truncate">
                {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists || 'Artista'}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


