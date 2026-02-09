'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { ListMusic, Loader2, TrendingUp, Sparkles, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export function PlaylistAnalyzer() {
  const [trackIds, setTrackIds] = useState<string>('');

  const analyzeMutation = useMutation({
    mutationFn: (ids: string[]) => musicApiService.analyzePlaylist(ids),
    onSuccess: () => {
      toast.success('Análisis de playlist completado');
    },
    onError: () => {
      toast.error('Error al analizar playlist');
    },
  });

  const handleAnalyze = () => {
    const ids = trackIds.split(',').map((id) => id.trim()).filter(Boolean);
    if (ids.length < 2) {
      toast.error('Ingresa al menos 2 track IDs separados por comas');
      return;
    }
    analyzeMutation.mutate(ids);
  };

  const COLORS = ['#a855f7', '#ec4899', '#8b5cf6', '#d946ef', '#f472b6'];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <ListMusic className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Análisis de Playlist</h2>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">
            Track IDs (separados por comas)
          </label>
          <textarea
            value={trackIds}
            onChange={(e) => setTrackIds(e.target.value)}
            placeholder="track_id_1, track_id_2, track_id_3..."
            rows={4}
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
          />
        </div>

        <button
          onClick={handleAnalyze}
          disabled={analyzeMutation.isPending}
          className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {analyzeMutation.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analizando...
            </>
          ) : (
            <>
              <ListMusic className="w-5 h-5" />
              Analizar Playlist
            </>
          )}
        </button>

        {analyzeMutation.data && (
          <div className="mt-6 space-y-6">
            {/* Metrics */}
            {analyzeMutation.data.metrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(analyzeMutation.data.metrics).map(([key, value]: [string, any]) => (
                  <div key={key} className="bg-white/5 rounded-lg p-4 text-center">
                    <p className="text-xs text-gray-400 mb-1 capitalize">
                      {key.replace(/_/g, ' ')}
                    </p>
                    <p className="text-2xl font-bold text-white">
                      {typeof value === 'number' ? value.toFixed(2) : String(value)}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {/* Diversity Chart */}
            {analyzeMutation.data.diversity && (
              <div className="bg-white/5 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-4">Diversidad Musical</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Diversidad', value: analyzeMutation.data.diversity.score || 0 },
                        { name: 'Resto', value: 1 - (analyzeMutation.data.diversity.score || 0) },
                      ]}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      dataKey="value"
                    >
                      <Cell fill="#a855f7" />
                      <Cell fill="#4a5568" />
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <p className="text-center text-sm text-gray-300 mt-2">
                  Score: {Math.round((analyzeMutation.data.diversity.score || 0) * 100)}%
                </p>
              </div>
            )}

            {/* Genre Distribution */}
            {analyzeMutation.data.genre_distribution && (
              <div className="bg-white/5 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-4">Distribución por Género</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={Object.entries(analyzeMutation.data.genre_distribution).map(([genre, count]) => ({ genre, count }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                    <XAxis
                      dataKey="genre"
                      stroke="#e2e8f0"
                      tick={{ fill: '#e2e8f0', fontSize: 12 }}
                    />
                    <YAxis
                      stroke="#e2e8f0"
                      tick={{ fill: '#e2e8f0', fontSize: 12 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1a1a2e',
                        border: '1px solid #4a5568',
                        borderRadius: '8px',
                        color: '#fff',
                      }}
                    />
                    <Bar dataKey="count" fill="#a855f7" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Suggestions */}
            {analyzeMutation.data.suggestions && analyzeMutation.data.suggestions.length > 0 && (
              <div className="bg-purple-500/20 rounded-lg p-4 border border-purple-400/30">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles className="w-5 h-5 text-purple-300" />
                  <h3 className="text-lg font-semibold text-white">Sugerencias de Mejora</h3>
                </div>
                <ul className="space-y-2">
                  {analyzeMutation.data.suggestions.map((suggestion: string, idx: number) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-gray-300">
                      <AlertCircle className="w-4 h-4 text-purple-300 mt-0.5 flex-shrink-0" />
                      <span>{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Flow Analysis */}
            {analyzeMutation.data.flow && (
              <div className="bg-white/5 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-3">Análisis de Flujo</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Coherencia</span>
                    <span className="text-white font-medium">
                      {Math.round((analyzeMutation.data.flow.coherence || 0) * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-purple-400 h-2 rounded-full"
                      style={{ width: `${(analyzeMutation.data.flow.coherence || 0) * 100}%` }}
                    />
                  </div>
                  {analyzeMutation.data.flow.transitions && (
                    <p className="text-xs text-gray-400 mt-2">
                      Transiciones suaves: {analyzeMutation.data.flow.transitions}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

