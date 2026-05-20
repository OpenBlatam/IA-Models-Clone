'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { GraduationCap, Lightbulb, Target, TrendingUp } from 'lucide-react';

interface MusicCoachingProps {
  trackId: string;
  trackName?: string;
}

export function MusicCoaching({ trackId, trackName }: MusicCoachingProps) {
  const { data: coaching, isLoading } = useQuery({
    queryKey: ['coaching', trackId, trackName],
    queryFn: () => musicApiService.getCoaching?.(trackName || trackId) || Promise.resolve({}),
    enabled: !!(trackId || trackName),
  });

  if (isLoading) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 text-center">
        <p className="text-gray-400">Cargando coaching...</p>
      </div>
    );
  }

  if (!coaching || !coaching.suggestions) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 text-center">
        <GraduationCap className="w-16 h-16 text-gray-500 mx-auto mb-4" />
        <p className="text-gray-400">No hay coaching disponible</p>
      </div>
    );
  }

  const suggestions = coaching.suggestions || [];
  const improvements = coaching.improvements || [];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <GraduationCap className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Coaching Musical</h3>
      </div>

      <div className="space-y-6">
        {suggestions.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Lightbulb className="w-4 h-4 text-yellow-400" />
              <h4 className="text-white font-medium">Sugerencias</h4>
            </div>
            <div className="space-y-2">
              {suggestions.map((suggestion: string, idx: number) => (
                <div
                  key={idx}
                  className="p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-lg"
                >
                  <p className="text-sm text-white">{suggestion}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {improvements.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Target className="w-4 h-4 text-green-400" />
              <h4 className="text-white font-medium">Áreas de Mejora</h4>
            </div>
            <div className="space-y-2">
              {improvements.map((improvement: string, idx: number) => (
                <div
                  key={idx}
                  className="p-3 bg-green-500/20 border border-green-500/30 rounded-lg"
                >
                  <p className="text-sm text-white">{improvement}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {coaching.strengths && coaching.strengths.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-4 h-4 text-blue-400" />
              <h4 className="text-white font-medium">Fortalezas</h4>
            </div>
            <div className="flex flex-wrap gap-2">
              {coaching.strengths.map((strength: string, idx: number) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-blue-500/20 border border-blue-500/30 rounded-full text-sm text-white"
                >
                  {strength}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

