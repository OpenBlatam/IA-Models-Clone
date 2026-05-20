'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Sparkles, Sun, Moon, Coffee, Dumbbell, Car } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicContextualProps {
  trackId: string;
  onTrackSelect?: (track: Track) => void;
}

export function MusicContextual({ trackId, onTrackSelect }: MusicContextualProps) {
  const [contextType, setContextType] = useState<'mood' | 'activity' | 'time'>('mood');
  const [contextValue, setContextValue] = useState('happy');

  const { data: recommendations, isLoading } = useQuery({
    queryKey: ['contextual-recommendations', trackId, contextType, contextValue],
    queryFn: async () => {
      switch (contextType) {
        case 'mood':
          return await musicApiService.getRecommendationsByMood?.(contextValue) || { recommendations: [] };
        case 'activity':
          return await musicApiService.getRecommendationsByActivity?.(contextValue) || { recommendations: [] };
        case 'time':
          return await musicApiService.getRecommendationsByTimeOfDay?.(contextValue) || { recommendations: [] };
        default:
          return { recommendations: [] };
      }
    },
    enabled: !!trackId,
  });

  const tracks = recommendations?.recommendations || [];

  const contextOptions = {
    mood: ['happy', 'sad', 'energetic', 'calm', 'romantic'],
    activity: ['workout', 'study', 'party', 'relax', 'driving'],
    time: ['morning', 'afternoon', 'evening', 'night'],
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Recomendaciones Contextuales</h3>
      </div>

      <div className="space-y-4 mb-4">
        <div className="flex gap-2">
          {(['mood', 'activity', 'time'] as const).map((type) => (
            <button
              key={type}
              onClick={() => {
                setContextType(type);
                setContextValue(contextOptions[type][0]);
              }}
              className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                contextType === type
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              {type === 'mood' && 'Mood'}
              {type === 'activity' && 'Actividad'}
              {type === 'time' && 'Hora'}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          {contextOptions[contextType].map((option) => (
            <button
              key={option}
              onClick={() => setContextValue(option)}
              className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                contextValue === option
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              {option}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-8 h-8 border-2 border-purple-300 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : tracks.length === 0 ? (
        <div className="text-center py-12">
          <Sparkles className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No hay recomendaciones disponibles</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
          {tracks.map((track: any, idx: number) => (
            <button
              key={track.id || idx}
              onClick={() => onTrackSelect?.(track)}
              className="p-4 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-left"
            >
              <div className="w-full aspect-square rounded bg-purple-500 flex items-center justify-center mb-3">
                <Sparkles className="w-12 h-12 text-white" />
              </div>
              <p className="text-white font-medium truncate text-sm mb-1">
                {track.name || track.track_name || 'Canción desconocida'}
              </p>
              <p className="text-xs text-gray-300 truncate">
                {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists || 'Artista'}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


