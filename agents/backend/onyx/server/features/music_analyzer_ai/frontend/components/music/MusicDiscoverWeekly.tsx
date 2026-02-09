'use client';

import { useState } from 'react';
import { Sparkles, Music, Play, Heart } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicDiscoverWeeklyProps {
  onTrackSelect?: (track: Track) => void;
}

export function MusicDiscoverWeekly({ onTrackSelect }: MusicDiscoverWeeklyProps) {
  const [discoverTracks] = useState<Array<Partial<Track>>>([]);

  // Simular tracks de descubrimiento
  const mockTracks: Array<Partial<Track>> = [
    {
      id: '1',
      name: 'Nueva Canción 1',
      artists: ['Artista Desconocido'],
      album: 'Álbum Nuevo',
    },
    {
      id: '2',
      name: 'Nueva Canción 2',
      artists: ['Otro Artista'],
      album: 'Otro Álbum',
    },
    {
      id: '3',
      name: 'Nueva Canción 3',
      artists: ['Tercer Artista'],
      album: 'Tercer Álbum',
    },
  ];

  const tracks = discoverTracks.length > 0 ? discoverTracks : mockTracks;

  return (
    <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">Descubrimiento Semanal</h3>
        </div>
        <span className="text-xs text-purple-300 bg-purple-500/20 px-2 py-1 rounded">
          Nuevo
        </span>
      </div>

      {tracks.length === 0 ? (
        <div className="text-center py-8">
          <Music className="w-12 h-12 text-gray-500 mx-auto mb-2" />
          <p className="text-gray-400 text-sm">No hay descubrimientos esta semana</p>
        </div>
      ) : (
        <div className="space-y-3">
          {tracks.map((track, idx) => (
            <div
              key={track.id || idx}
              className="flex items-center gap-3 p-3 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-colors group"
            >
              <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <Music className="w-6 h-6 text-purple-300" />
              </div>
              <div className="flex-1">
                <p className="text-white font-medium text-sm">{track.name}</p>
                <p className="text-gray-400 text-xs">
                  {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors">
                  <Heart className="w-4 h-4" />
                </button>
                <button
                  onClick={() => onTrackSelect?.(track as Track)}
                  className="p-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                >
                  <Play className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


