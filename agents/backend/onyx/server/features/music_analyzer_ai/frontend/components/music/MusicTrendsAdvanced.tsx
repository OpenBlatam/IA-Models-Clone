'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { TrendingUp, TrendingDown, Minus, ArrowUp, ArrowDown } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicTrendsAdvancedProps {
  onTrackSelect?: (track: Track) => void;
}

export function MusicTrendsAdvanced({ onTrackSelect }: MusicTrendsAdvancedProps) {
  const { data: trends } = useQuery({
    queryKey: ['trends-advanced'],
    queryFn: () => musicApiService.getTrends?.(50) || Promise.resolve({ trends: [] }),
  });

  const { data: predictions } = useQuery({
    queryKey: ['trend-predictions'],
    queryFn: () => musicApiService.predictSuccess?.() || Promise.resolve({ predictions: [] }),
  });

  const trendsData = trends?.trends || [];
  const predictionsData = predictions?.predictions || [];

  const getTrendIcon = (change: number) => {
    if (change > 0) return <ArrowUp className="w-4 h-4 text-green-400" />;
    if (change < 0) return <ArrowDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-gray-400" />;
  };

  return (
    <div className="space-y-6">
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">Tendencias Actuales</h3>
        </div>

        {trendsData.length > 0 ? (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {trendsData.map((trend: any, idx: number) => (
              <button
                key={trend.id || idx}
                onClick={() => onTrackSelect?.(trend)}
                className="w-full flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
              >
                <div className="w-10 h-10 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                  <span className="text-white font-bold text-sm">#{idx + 1}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">
                    {trend.name || trend.track_name || 'Canción desconocida'}
                  </p>
                  <p className="text-sm text-gray-300 truncate">
                    {Array.isArray(trend.artists) ? trend.artists.join(', ') : trend.artists || 'Artista'}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {getTrendIcon(trend.trend_score || 0)}
                  <span className="text-sm text-purple-300 font-medium">
                    {trend.trend_score ? `+${Math.round(trend.trend_score * 100)}%` : 'N/A'}
                  </span>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <TrendingUp className="w-16 h-16 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">No hay tendencias disponibles</p>
          </div>
        )}
      </div>

      {predictionsData.length > 0 && (
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <TrendingDown className="w-5 h-5 text-yellow-300" />
            <h3 className="text-lg font-semibold text-white">Predicciones de Éxito</h3>
          </div>

          <div className="space-y-2 max-h-96 overflow-y-auto">
            {predictionsData.map((prediction: any, idx: number) => (
              <div
                key={idx}
                className="p-3 bg-white/5 rounded-lg border border-white/10"
              >
                <div className="flex items-center justify-between mb-2">
                  <p className="text-white font-medium">
                    {prediction.track_name || 'Canción desconocida'}
                  </p>
                  <span className="text-sm text-yellow-300 font-medium">
                    {prediction.success_probability
                      ? `${Math.round(prediction.success_probability * 100)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-yellow-400 h-2 rounded-full"
                    style={{
                      width: `${(prediction.success_probability || 0) * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


