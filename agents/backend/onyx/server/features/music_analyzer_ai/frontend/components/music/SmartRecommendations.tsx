'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Sparkles, Loader2, Lightbulb } from 'lucide-react';
import toast from 'react-hot-toast';

interface SmartRecommendationsProps {
  trackId: string;
  currentAnalysis?: any;
}

export function SmartRecommendations({ trackId, currentAnalysis }: SmartRecommendationsProps) {
  const [recommendationType, setRecommendationType] = useState<'similar' | 'complementary' | 'discovery'>('similar');

  const recommendationMutation = useMutation({
    mutationFn: async () => {
      switch (recommendationType) {
        case 'similar':
          return musicApiService.getRecommendations(trackId, 10);
        case 'complementary':
          // Recomendaciones complementarias basadas en análisis
          return musicApiService.getContextualRecommendations(trackId, 'complementary', 10);
        case 'discovery':
          return musicApiService.getUndergroundTracks(10);
        default:
          return musicApiService.getRecommendations(trackId, 10);
      }
    },
    onSuccess: () => {
      toast.success('Recomendaciones generadas');
    },
    onError: () => {
      toast.error('Error al generar recomendaciones');
    },
  });

  const handleGetRecommendations = () => {
    recommendationMutation.mutate();
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Lightbulb className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Recomendaciones Inteligentes</h2>
      </div>

      <div className="space-y-4">
        {/* Type Selector */}
        <div className="flex gap-2">
          {[
            { value: 'similar', label: 'Similares' },
            { value: 'complementary', label: 'Complementarias' },
            { value: 'discovery', label: 'Descubrimiento' },
          ].map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setRecommendationType(value as any)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                recommendationType === value
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Analysis-based insights */}
        {currentAnalysis && (
          <div className="bg-purple-500/20 rounded-lg p-4 border border-purple-400/30">
            <p className="text-sm text-gray-300 mb-2">
              Basado en el análisis, recomendamos canciones con características similares:
            </p>
            <div className="flex flex-wrap gap-2">
              {currentAnalysis.musical_analysis?.key_signature && (
                <span className="px-2 py-1 bg-purple-500/30 rounded text-xs text-white">
                  Tonalidad: {currentAnalysis.musical_analysis.key_signature}
                </span>
              )}
              {currentAnalysis.musical_analysis?.tempo && (
                <span className="px-2 py-1 bg-purple-500/30 rounded text-xs text-white">
                  Tempo: {Math.round(currentAnalysis.musical_analysis.tempo.bpm)} BPM
                </span>
              )}
            </div>
          </div>
        )}

        <button
          onClick={handleGetRecommendations}
          disabled={recommendationMutation.isPending}
          className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {recommendationMutation.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Generando...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              Obtener Recomendaciones
            </>
          )}
        </button>

        {/* Results */}
        {recommendationMutation.data && recommendationMutation.data.recommendations && (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {recommendationMutation.data.recommendations.map((rec: any) => (
              <div
                key={rec.id}
                className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
              >
                {rec.images && rec.images[0] ? (
                  <img
                    src={rec.images[0].url}
                    alt={rec.name}
                    className="w-12 h-12 rounded"
                  />
                ) : (
                  <div className="w-12 h-12 rounded bg-purple-500" />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">{rec.name}</p>
                  <p className="text-sm text-gray-300 truncate">
                    {rec.artists?.join(', ') || 'Artista desconocido'}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-400">Match</p>
                  <p className="text-sm text-white font-medium">
                    {rec.match_score ? `${Math.round(rec.match_score * 100)}%` : 'N/A'}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}


