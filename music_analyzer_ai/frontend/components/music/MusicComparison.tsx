'use client';

import { useState } from 'react';
import { GitCompare, Plus, X } from 'lucide-react';
import toast from 'react-hot-toast';
import { type Track } from '@/lib/api/music-api';

interface MusicComparisonProps {
  tracks: Track[];
  onTracksChange: (tracks: Track[]) => void;
  onCompare: () => void;
}

export function MusicComparison({ tracks, onTracksChange, onCompare }: MusicComparisonProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const handleRemove = (trackId: string) => {
    onTracksChange(tracks.filter((t) => t.id !== trackId));
  };

  const handleAdd = () => {
    // En producción, esto abriría un modal de búsqueda
    toast.info('Funcionalidad de agregar próximamente');
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <GitCompare className="w-6 h-6 text-purple-300" />
          <h2 className="text-2xl font-semibold text-white">Comparar Canciones</h2>
        </div>
        {tracks.length >= 2 && (
          <button
            onClick={onCompare}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
          >
            Comparar
          </button>
        )}
      </div>

      <div className="mb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Buscar canción para agregar..."
            className="flex-1 px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
          />
          <button
            onClick={handleAdd}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Agregar
          </button>
        </div>
      </div>

      <div className="space-y-2">
        {tracks.length === 0 ? (
          <div className="text-center py-12">
            <GitCompare className="w-16 h-16 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">Agrega al menos 2 canciones para comparar</p>
          </div>
        ) : (
          tracks.map((track, idx) => (
            <div
              key={track.id || idx}
              className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold">{idx + 1}</span>
              </div>
              {track.images && track.images[0] && (
                <img
                  src={track.images[0].url}
                  alt={track.name}
                  className="w-12 h-12 rounded"
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">{track.name}</p>
                <p className="text-sm text-gray-300 truncate">
                  {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists}
                </p>
              </div>
              <button
                onClick={() => handleRemove(track.id)}
                className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/20 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          ))
        )}
      </div>

      {tracks.length > 0 && tracks.length < 2 && (
        <div className="mt-4 p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-lg">
          <p className="text-yellow-300 text-sm">
            Agrega al menos {2 - tracks.length} canción más para comparar
          </p>
        </div>
      )}
    </div>
  );
}

