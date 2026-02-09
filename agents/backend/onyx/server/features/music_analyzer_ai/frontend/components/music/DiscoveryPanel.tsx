'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { Compass, Search, Sparkles, Music } from 'lucide-react';
import toast from 'react-hot-toast';

export function DiscoveryPanel() {
  const [discoveryType, setDiscoveryType] = useState<'underground' | 'similar-artists'>('underground');
  const [artistName, setArtistName] = useState('');

  const undergroundMutation = useMutation({
    mutationFn: () => musicApiService.getUndergroundTracks(20),
    onSuccess: () => {
      toast.success('Tracks underground encontrados');
    },
    onError: () => {
      toast.error('Error al buscar tracks underground');
    },
  });

  const similarArtistsMutation = useMutation({
    mutationFn: (name: string) => musicApiService.getSimilarArtists(name, 10),
    onSuccess: () => {
      toast.success('Artistas similares encontrados');
    },
    onError: () => {
      toast.error('Error al buscar artistas similares');
    },
  });

  const handleDiscover = () => {
    if (discoveryType === 'underground') {
      undergroundMutation.mutate();
    } else if (discoveryType === 'similar-artists' && artistName.trim()) {
      similarArtistsMutation.mutate(artistName);
    } else if (discoveryType === 'similar-artists') {
      toast.error('Ingresa un nombre de artista');
    }
  };

  const currentData = discoveryType === 'underground' 
    ? undergroundMutation.data 
    : similarArtistsMutation.data;

  const isLoading = undergroundMutation.isPending || similarArtistsMutation.isPending;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Compass className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Descubrimiento Musical</h2>
      </div>

      {/* Discovery Type Selector */}
      <div className="mb-4">
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setDiscoveryType('underground')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              discoveryType === 'underground'
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            Tracks Underground
          </button>
          <button
            onClick={() => setDiscoveryType('similar-artists')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              discoveryType === 'similar-artists'
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            Artistas Similares
          </button>
        </div>

        {discoveryType === 'similar-artists' && (
          <div className="mb-4">
            <input
              type="text"
              value={artistName}
              onChange={(e) => setArtistName(e.target.value)}
              placeholder="Nombre del artista..."
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
            />
          </div>
        )}

        <button
          onClick={handleDiscover}
          disabled={isLoading}
          className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          <Compass className="w-5 h-5" />
          {isLoading ? 'Descubriendo...' : 'Descubrir'}
        </button>
      </div>

      {/* Results */}
      {currentData && (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {discoveryType === 'underground' && currentData.tracks && (
            <>
              {currentData.tracks.map((track: Track) => (
                <div
                  key={track.id}
                  className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
                >
                  {track.images && track.images[0] ? (
                    <img
                      src={track.images[0].url}
                      alt={track.name}
                      className="w-12 h-12 rounded"
                    />
                  ) : (
                    <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center">
                      <Music className="w-6 h-6 text-white" />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium truncate">{track.name}</p>
                    <p className="text-sm text-gray-300 truncate">
                      {track.artists.join(', ')}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-400">Popularidad</p>
                    <p className="text-sm text-white font-medium">{track.popularity}</p>
                  </div>
                </div>
              ))}
            </>
          )}

          {discoveryType === 'similar-artists' && currentData.artists && (
            <>
              {currentData.artists.map((artist: any) => (
                <div
                  key={artist.id || artist.name}
                  className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
                >
                  <div className="w-12 h-12 rounded-full bg-purple-500 flex items-center justify-center">
                    <Music className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <p className="text-white font-medium">{artist.name}</p>
                    {artist.genres && (
                      <p className="text-sm text-gray-300">
                        {Array.isArray(artist.genres) ? artist.genres.join(', ') : artist.genres}
                      </p>
                    )}
                    {artist.similarity_score && (
                      <p className="text-xs text-purple-300">
                        Similitud: {Math.round(artist.similarity_score * 100)}%
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

