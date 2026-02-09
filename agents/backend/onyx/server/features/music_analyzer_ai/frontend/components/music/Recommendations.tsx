'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { Sparkles, Clock, Activity, Heart, Music } from 'lucide-react';
import toast from 'react-hot-toast';

interface RecommendationsProps {
  trackId: string;
  track: Track;
}

export function Recommendations({ trackId, track }: RecommendationsProps) {
  const [recommendationType, setRecommendationType] = useState<
    'similar' | 'mood' | 'activity' | 'time'
  >('similar');
  const [mood, setMood] = useState('happy');
  const [activity, setActivity] = useState('workout');
  const [timeOfDay, setTimeOfDay] = useState('morning');

  const recommendationsMutation = useMutation({
    mutationFn: async () => {
      switch (recommendationType) {
        case 'similar':
          return musicApiService.getRecommendations(trackId, 20);
        case 'mood':
          return musicApiService.getRecommendationsByMood(mood, 20);
        case 'activity':
          return musicApiService.getRecommendationsByActivity(activity, 20);
        case 'time':
          return musicApiService.getRecommendationsByTimeOfDay(timeOfDay, 20);
        default:
          return musicApiService.getRecommendations(trackId, 20);
      }
    },
    onSuccess: () => {
      toast.success('Recomendaciones generadas');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error al obtener recomendaciones');
    },
  });

  const handleGetRecommendations = () => {
    recommendationsMutation.mutate();
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
        <Sparkles className="w-6 h-6" />
        Recomendaciones
      </h2>

      {/* Recommendation Type Selector */}
      <div className="mb-4">
        <div className="flex flex-wrap gap-2 mb-4">
          {[
            { value: 'similar', label: 'Similares', icon: Music },
            { value: 'mood', label: 'Por Mood', icon: Heart },
            { value: 'activity', label: 'Por Actividad', icon: Activity },
            { value: 'time', label: 'Por Hora', icon: Clock },
          ].map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => setRecommendationType(value as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                recommendationType === value
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{label}</span>
            </button>
          ))}
        </div>

        {/* Additional Options */}
        {recommendationType === 'mood' && (
          <select
            value={mood}
            onChange={(e) => setMood(e.target.value)}
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
          >
            <option value="happy">Feliz</option>
            <option value="sad">Triste</option>
            <option value="energetic">Energético</option>
            <option value="calm">Calmado</option>
            <option value="romantic">Romántico</option>
          </select>
        )}

        {recommendationType === 'activity' && (
          <select
            value={activity}
            onChange={(e) => setActivity(e.target.value)}
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
          >
            <option value="workout">Ejercicio</option>
            <option value="study">Estudio</option>
            <option value="party">Fiesta</option>
            <option value="relax">Relajación</option>
            <option value="drive">Conducir</option>
          </select>
        )}

        {recommendationType === 'time' && (
          <select
            value={timeOfDay}
            onChange={(e) => setTimeOfDay(e.target.value)}
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
          >
            <option value="morning">Mañana</option>
            <option value="afternoon">Tarde</option>
            <option value="evening">Noche</option>
            <option value="night">Madrugada</option>
          </select>
        )}

        <button
          onClick={handleGetRecommendations}
          disabled={recommendationsMutation.isPending}
          className="mt-4 w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50"
        >
          {recommendationsMutation.isPending ? 'Generando...' : 'Obtener Recomendaciones'}
        </button>
      </div>

      {/* Recommendations List */}
      {recommendationsMutation.data && recommendationsMutation.data.recommendations && (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {recommendationsMutation.data.recommendations.map((rec: Track) => (
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
                  {rec.artists.join(', ')}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-400">Popularidad</p>
                <p className="text-sm text-white font-medium">{rec.popularity}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

