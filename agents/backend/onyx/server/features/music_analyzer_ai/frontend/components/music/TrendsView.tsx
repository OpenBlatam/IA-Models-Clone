'use client';

import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { TrendingUp, TrendingDown, BarChart3, Loader2 } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import toast from 'react-hot-toast';

export function TrendsView() {
  const [selectedTrackId, setSelectedTrackId] = useState<string>('');

  const { data: trends, isLoading: trendsLoading } = useQuery({
    queryKey: ['trends'],
    queryFn: () => musicApiService.getTrends(),
    refetchInterval: 60000, // Actualizar cada minuto
  });

  const successPredictionMutation = useMutation({
    mutationFn: (trackId: string) => musicApiService.predictSuccess(trackId),
    onSuccess: () => {
      toast.success('Predicción de éxito generada');
    },
    onError: () => {
      toast.error('Error al predecir éxito');
    },
  });

  const handlePredictSuccess = () => {
    if (!selectedTrackId.trim()) {
      toast.error('Ingresa un track ID');
      return;
    }
    successPredictionMutation.mutate(selectedTrackId);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-6">
        <TrendingUp className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Tendencias Musicales</h2>
      </div>

      {trendsLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-purple-300 animate-spin" />
        </div>
      ) : (
        <div className="space-y-6">
          {/* Trends Chart */}
          {trends && trends.popularity_trends && (
            <div className="bg-white/5 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-4">Tendencias de Popularidad</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trends.popularity_trends}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                  <XAxis
                    dataKey="period"
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
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="popularity"
                    stroke="#a855f7"
                    strokeWidth={2}
                    name="Popularidad"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Genre Trends */}
          {trends && trends.genre_trends && (
            <div className="bg-white/5 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-4">Tendencias por Género</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={trends.genre_trends}>
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
                  <Bar dataKey="count" fill="#a855f7" name="Canciones" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Success Prediction */}
          <div className="bg-white/5 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Predicción de Éxito</h3>
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={selectedTrackId}
                onChange={(e) => setSelectedTrackId(e.target.value)}
                placeholder="Track ID de Spotify..."
                className="flex-1 px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
              <button
                onClick={handlePredictSuccess}
                disabled={successPredictionMutation.isPending}
                className="px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                {successPredictionMutation.isPending ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <TrendingUp className="w-5 h-5" />
                )}
                Predecir
              </button>
            </div>

            {successPredictionMutation.data && (
              <div className="mt-4 p-4 bg-purple-500/20 rounded-lg border border-purple-400/30">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Probabilidad de Éxito</span>
                  <span className="text-2xl font-bold text-white">
                    {Math.round((successPredictionMutation.data.success_probability || 0) * 100)}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                  <div
                    className="bg-purple-400 h-2 rounded-full transition-all"
                    style={{
                      width: `${(successPredictionMutation.data.success_probability || 0) * 100}%`,
                    }}
                  />
                </div>
                {successPredictionMutation.data.reasons && (
                  <div className="mt-4">
                    <p className="text-sm text-gray-400 mb-2">Razones:</p>
                    <ul className="space-y-1">
                      {successPredictionMutation.data.reasons.map((reason: string, idx: number) => (
                        <li key={idx} className="text-sm text-gray-300">• {reason}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

