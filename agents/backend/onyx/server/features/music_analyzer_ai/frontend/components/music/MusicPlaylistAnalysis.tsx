'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { ListMusic, BarChart3, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

interface MusicPlaylistAnalysisProps {
  playlistId?: string;
}

export function MusicPlaylistAnalysis({ playlistId }: MusicPlaylistAnalysisProps) {
  const [selectedPlaylistId, setSelectedPlaylistId] = useState(playlistId || '');

  const { data: playlists } = useQuery({
    queryKey: ['playlists', 'user123'],
    queryFn: () => musicApiService.getPlaylists?.('user123') || Promise.resolve({ playlists: [] }),
  });

  const { data: analysis, isLoading } = useQuery({
    queryKey: ['playlist-analysis', selectedPlaylistId],
    queryFn: () => musicApiService.analyzePlaylist?.(selectedPlaylistId) || Promise.resolve({}),
    enabled: !!selectedPlaylistId,
  });

  const playlistList = playlists?.playlists || [];

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 0.8) return <CheckCircle className="w-5 h-5 text-green-400" />;
    return <AlertCircle className="w-5 h-5 text-yellow-400" />;
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <ListMusic className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Análisis de Playlist</h3>
      </div>

      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">Seleccionar Playlist</label>
        <select
          value={selectedPlaylistId}
          onChange={(e) => setSelectedPlaylistId(e.target.value)}
          className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
        >
          <option value="">Seleccionar playlist</option>
          {playlistList.map((playlist: any) => (
            <option key={playlist.id} value={playlist.id}>
              {playlist.name || 'Playlist sin nombre'}
            </option>
          ))}
        </select>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-purple-300 animate-spin" />
        </div>
      ) : analysis && selectedPlaylistId ? (
        <div className="space-y-4">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-white/5 rounded-lg border border-white/10">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-4 h-4 text-purple-300" />
                <span className="text-sm text-gray-400">Diversidad</span>
              </div>
              <div className="flex items-center gap-2">
                {getScoreIcon(analysis.diversity_score || 0)}
                <span className={`text-2xl font-bold ${getScoreColor(analysis.diversity_score || 0)}`}>
                  {Math.round((analysis.diversity_score || 0) * 100)}%
                </span>
              </div>
            </div>

            <div className="p-4 bg-white/5 rounded-lg border border-white/10">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-4 h-4 text-purple-300" />
                <span className="text-sm text-gray-400">Coherencia</span>
              </div>
              <div className="flex items-center gap-2">
                {getScoreIcon(analysis.coherence_score || 0)}
                <span className={`text-2xl font-bold ${getScoreColor(analysis.coherence_score || 0)}`}>
                  {Math.round((analysis.coherence_score || 0) * 100)}%
                </span>
              </div>
            </div>

            <div className="p-4 bg-white/5 rounded-lg border border-white/10">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-4 h-4 text-purple-300" />
                <span className="text-sm text-gray-400">Energía Promedio</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-white">
                  {Math.round((analysis.avg_energy || 0) * 100)}%
                </span>
              </div>
            </div>
          </div>

          {analysis.genre_distribution && (
            <div className="p-4 bg-white/5 rounded-lg border border-white/10">
              <h4 className="text-white font-medium mb-3">Distribución por Género</h4>
              <div className="space-y-2">
                {Object.entries(analysis.genre_distribution).map(([genre, count]: [string, any]) => (
                  <div key={genre} className="flex items-center justify-between">
                    <span className="text-gray-300">{genre}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-purple-500 h-2 rounded-full"
                          style={{ width: `${(count / analysis.total_tracks) * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-400 w-12 text-right">{count}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12">
          <ListMusic className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">Selecciona una playlist para analizar</p>
        </div>
      )}
    </div>
  );
}


