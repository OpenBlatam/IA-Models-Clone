'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Users, Network, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

export function CollaborationNetwork() {
  const [artistNames, setArtistNames] = useState<string>('');

  const networkMutation = useMutation({
    mutationFn: (artists: string[]) => musicApiService.getCollaborationNetwork?.(artists) || Promise.resolve({}),
    onSuccess: () => {
      toast.success('Red de colaboraciones generada');
    },
    onError: () => {
      toast.error('Error al analizar colaboraciones');
    },
  });

  const handleAnalyze = () => {
    const artists = artistNames.split(',').map((a) => a.trim()).filter(Boolean);
    if (artists.length < 2) {
      toast.error('Ingresa al menos 2 artistas separados por comas');
      return;
    }
    networkMutation.mutate(artists);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Network className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Red de Colaboraciones</h2>
      </div>

      <div className="mb-4">
        <input
          type="text"
          value={artistNames}
          onChange={(e) => setArtistNames(e.target.value)}
          placeholder="Artista 1, Artista 2, Artista 3..."
          className="w-full px-4 py-2 mb-3 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />
        <button
          onClick={handleAnalyze}
          disabled={networkMutation.isPending}
          className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {networkMutation.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analizando...
            </>
          ) : (
            <>
              <Users className="w-5 h-5" />
              Analizar Red
            </>
          )}
        </button>
      </div>

      {networkMutation.data && (
        <div className="space-y-4">
          {networkMutation.data.collaborations && (
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-2">Colaboraciones Encontradas</p>
              <div className="space-y-2">
                {networkMutation.data.collaborations.map((collab: any, idx: number) => (
                  <div key={idx} className="flex items-center gap-2 text-sm text-white">
                    <span className="text-purple-300">{collab.artist1}</span>
                    <span className="text-gray-400">↔</span>
                    <span className="text-purple-300">{collab.artist2}</span>
                    {collab.count && (
                      <span className="text-xs text-gray-400 ml-auto">
                        {collab.count} colaboraciones
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {networkMutation.data.network_stats && (
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(networkMutation.data.network_stats).map(([key, value]: [string, any]) => (
                <div key={key} className="bg-white/5 rounded-lg p-3 text-center">
                  <p className="text-xs text-gray-400 mb-1 capitalize">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <p className="text-lg font-semibold text-white">{value}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

