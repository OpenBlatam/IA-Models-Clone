'use client';

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { type Track, type MusicAnalysisResponse, musicApiService } from '@/lib/api/music-api';
import { Music, TrendingUp, Zap, Heart, Volume2 } from 'lucide-react';
import { formatBPM, formatPercentage } from '@/lib/utils';
import toast from 'react-hot-toast';

interface TrackAnalysisProps {
  analysis: MusicAnalysisResponse;
  track: Track;
}

export function TrackAnalysis({ analysis, track }: TrackAnalysisProps) {
  const { musical_analysis, technical_analysis, coaching } = analysis;
  const [userId] = useState('user123');
  const queryClient = useQueryClient();

  const addFavoriteMutation = useMutation({
    mutationFn: () =>
      musicApiService.addToFavorites(userId, track.id, track.name, track.artists),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['favorites', userId] });
      toast.success('Agregado a favoritos');
    },
    onError: () => {
      toast.error('Error al agregar a favoritos');
    },
  });

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 space-y-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-white flex items-center gap-2">
          <Music className="w-6 h-6" />
          Análisis Musical
        </h2>
        <button
          onClick={() => addFavoriteMutation.mutate()}
          disabled={addFavoriteMutation.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg transition-colors disabled:opacity-50"
        >
          <Heart className="w-5 h-5" />
          Agregar a Favoritos
        </button>
      </div>

      {/* Musical Analysis */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white/5 rounded-lg p-4">
          <p className="text-sm text-gray-400 mb-1">Tonalidad</p>
          <p className="text-xl font-semibold text-white">
            {musical_analysis.key_signature}
          </p>
        </div>
        <div className="bg-white/5 rounded-lg p-4">
          <p className="text-sm text-gray-400 mb-1">Tempo</p>
          <p className="text-xl font-semibold text-white">
            {formatBPM(musical_analysis.tempo.bpm)}
          </p>
          <p className="text-xs text-gray-400">{musical_analysis.tempo.category}</p>
        </div>
        <div className="bg-white/5 rounded-lg p-4">
          <p className="text-sm text-gray-400 mb-1">Compás</p>
          <p className="text-xl font-semibold text-white">
            {musical_analysis.time_signature}
          </p>
        </div>
        <div className="bg-white/5 rounded-lg p-4">
          <p className="text-sm text-gray-400 mb-1">Modo</p>
          <p className="text-xl font-semibold text-white">
            {musical_analysis.mode}
          </p>
        </div>
      </div>

      {/* Scale */}
      {musical_analysis.scale && (
        <div className="bg-white/5 rounded-lg p-4">
          <p className="text-sm text-gray-400 mb-2">Escala</p>
          <p className="text-lg font-semibold text-white mb-2">
            {musical_analysis.scale.name}
          </p>
          <div className="flex flex-wrap gap-2">
            {musical_analysis.scale.notes.map((note) => (
              <span
                key={note}
                className="px-3 py-1 bg-purple-500/30 rounded-full text-sm text-white"
              >
                {note}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Technical Analysis */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Análisis Técnico
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white/5 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-yellow-400" />
              <p className="text-sm text-gray-400">Energía</p>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-gray-700 rounded-full h-2">
                <div
                  className="bg-yellow-400 h-2 rounded-full"
                  style={{ width: `${technical_analysis.energy.value * 100}%` }}
                />
              </div>
              <span className="text-sm text-white font-medium">
                {formatPercentage(technical_analysis.energy.value)}
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {technical_analysis.energy.description}
            </p>
          </div>

          <div className="bg-white/5 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Music className="w-4 h-4 text-pink-400" />
              <p className="text-sm text-gray-400">Bailabilidad</p>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-gray-700 rounded-full h-2">
                <div
                  className="bg-pink-400 h-2 rounded-full"
                  style={{ width: `${technical_analysis.danceability.value * 100}%` }}
                />
              </div>
              <span className="text-sm text-white font-medium">
                {formatPercentage(technical_analysis.danceability.value)}
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {technical_analysis.danceability.description}
            </p>
          </div>

          {technical_analysis.valence && (
            <div className="bg-white/5 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Heart className="w-4 h-4 text-red-400" />
                <p className="text-sm text-gray-400">Valencia</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-red-400 h-2 rounded-full"
                    style={{ width: `${technical_analysis.valence.value * 100}%` }}
                  />
                </div>
                <span className="text-sm text-white font-medium">
                  {formatPercentage(technical_analysis.valence.value)}
                </span>
              </div>
            </div>
          )}

          {technical_analysis.loudness && (
            <div className="bg-white/5 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Volume2 className="w-4 h-4 text-blue-400" />
                <p className="text-sm text-gray-400">Volumen</p>
              </div>
              <p className="text-lg font-semibold text-white">
                {technical_analysis.loudness.value.toFixed(1)} dB
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Coaching */}
      {coaching && (
        <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-lg p-4 border border-purple-400/30">
          <h3 className="text-lg font-semibold text-white mb-2">Coaching Musical</h3>
          {coaching.overview && (
            <div className="mb-3">
              <p className="text-sm text-gray-300 mb-1">
                Nivel de Dificultad: <span className="font-semibold text-white">{coaching.overview.difficulty_level}</span>
              </p>
              {coaching.overview.suitable_for && (
                <p className="text-sm text-gray-300">
                  Adecuado para: {coaching.overview.suitable_for.join(', ')}
                </p>
              )}
            </div>
          )}
          {coaching.learning_path && coaching.learning_path.length > 0 && (
            <div className="mt-4">
              <p className="text-sm font-semibold text-white mb-2">Ruta de Aprendizaje:</p>
              <ol className="list-decimal list-inside space-y-1 text-sm text-gray-300">
                {coaching.learning_path.slice(0, 3).map((step: any, idx: number) => (
                  <li key={idx}>{step.title || step.description}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

