'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Brain, Loader2, BarChart3 } from 'lucide-react';

interface MusicMLAdvancedProps {
  trackId: string;
}

export function MusicMLAdvanced({ trackId }: MusicMLAdvancedProps) {
  const { data: genrePrediction } = useQuery({
    queryKey: ['ml-genre', trackId],
    queryFn: () => musicApiService.predictGenre?.(trackId) || Promise.resolve({}),
    enabled: !!trackId,
  });

  const { data: multiTask } = useQuery({
    queryKey: ['ml-multitask', trackId],
    queryFn: () => musicApiService.predictMultiTask?.(trackId) || Promise.resolve({}),
    enabled: !!trackId,
  });

  return (
    <div className="space-y-6">
      {genrePrediction && (
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-5 h-5 text-purple-300" />
            <h3 className="text-lg font-semibold text-white">Predicción de Género (ML)</h3>
          </div>

          {genrePrediction.predicted_genre && (
            <div className="space-y-3">
              <div className="p-4 bg-purple-500/20 border border-purple-500/30 rounded-lg">
                <p className="text-white font-medium">
                  Género Predicho: <span className="text-purple-300">{genrePrediction.predicted_genre}</span>
                </p>
                {genrePrediction.confidence && (
                  <p className="text-sm text-gray-300 mt-1">
                    Confianza: {Math.round(genrePrediction.confidence * 100)}%
                  </p>
                )}
              </div>

              {genrePrediction.genre_probabilities && (
                <div className="space-y-2">
                  <h4 className="text-white font-medium text-sm">Probabilidades por Género</h4>
                  {Object.entries(genrePrediction.genre_probabilities)
                    .sort(([, a]: any, [, b]: any) => b - a)
                    .slice(0, 5)
                    .map(([genre, prob]: [string, any]) => (
                      <div key={genre} className="flex items-center justify-between">
                        <span className="text-gray-300 text-sm">{genre}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-purple-500 h-2 rounded-full"
                              style={{ width: `${prob * 100}%` }}
                            />
                          </div>
                          <span className="text-sm text-white w-12 text-right">
                            {Math.round(prob * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {multiTask && (
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-purple-300" />
            <h3 className="text-lg font-semibold text-white">Análisis Multi-Tarea (ML)</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {multiTask.predictions && Object.entries(multiTask.predictions).map(([key, value]: [string, any]) => (
              <div key={key} className="p-4 bg-white/5 rounded-lg border border-white/10">
                <p className="text-gray-400 text-sm mb-1 capitalize">{key.replace('_', ' ')}</p>
                <p className="text-white font-medium text-lg">
                  {typeof value === 'number' ? Math.round(value * 100) / 100 : value}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


