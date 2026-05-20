'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService, type Track, type ComparisonResponse } from '@/lib/api/music-api';
import { GitCompare, Loader2, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import toast from 'react-hot-toast';
import { formatBPM } from '@/lib/utils';

interface TrackComparisonProps {
  tracks: Track[];
}

export function TrackComparison({ tracks }: TrackComparisonProps) {
  const [selectedTracks, setSelectedTracks] = useState<Track[]>([]);

  const comparisonMutation = useMutation({
    mutationFn: (trackIds: string[]) => musicApiService.compareTracks(trackIds),
    onSuccess: (data) => {
      toast.success('Comparación completada');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error al comparar canciones');
    },
  });

  const handleToggleTrack = (track: Track) => {
    if (selectedTracks.find((t) => t.id === track.id)) {
      setSelectedTracks(selectedTracks.filter((t) => t.id !== track.id));
    } else if (selectedTracks.length < 5) {
      setSelectedTracks([...selectedTracks, track]);
    } else {
      toast.error('Máximo 5 canciones para comparar');
    }
  };

  const handleCompare = () => {
    if (selectedTracks.length < 2) {
      toast.error('Selecciona al menos 2 canciones para comparar');
      return;
    }
    comparisonMutation.mutate(selectedTracks.map((t) => t.id));
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
        <GitCompare className="w-6 h-6" />
        Comparar Canciones
      </h2>

      {/* Selected Tracks */}
      {selectedTracks.length > 0 && (
        <div className="mb-4">
          <p className="text-sm text-gray-300 mb-2">
            Seleccionadas: {selectedTracks.length}/5
          </p>
          <div className="flex flex-wrap gap-2">
            {selectedTracks.map((track) => (
              <div
                key={track.id}
                className="flex items-center gap-2 bg-purple-500/30 rounded-lg px-3 py-2"
              >
                <span className="text-white text-sm">{track.name}</span>
                <button
                  onClick={() => handleToggleTrack(track)}
                  className="text-white hover:text-red-300"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <button
            onClick={handleCompare}
            disabled={comparisonMutation.isPending || selectedTracks.length < 2}
            className="mt-4 px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {comparisonMutation.isPending ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <GitCompare className="w-5 h-5" />
            )}
            Comparar
          </button>
        </div>
      )}

      {/* Track List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {tracks.map((track) => {
          const isSelected = selectedTracks.find((t) => t.id === track.id);
          return (
            <button
              key={track.id}
              onClick={() => handleToggleTrack(track)}
              className={`w-full flex items-center gap-3 p-3 rounded-lg transition-colors text-left ${
                isSelected
                  ? 'bg-purple-500/30 border-2 border-purple-400'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              {track.images && track.images[0] ? (
                <img
                  src={track.images[0].url}
                  alt={track.name}
                  className="w-12 h-12 rounded"
                />
              ) : (
                <div className="w-12 h-12 rounded bg-purple-500" />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">{track.name}</p>
                <p className="text-sm text-gray-300 truncate">
                  {track.artists.join(', ')}
                </p>
              </div>
              {isSelected && (
                <div className="w-6 h-6 rounded-full bg-purple-500 flex items-center justify-center">
                  <span className="text-white text-xs">✓</span>
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Comparison Results */}
      {comparisonMutation.data && (
        <div className="mt-6 p-4 bg-white/5 rounded-lg border border-white/10">
          <h3 className="text-lg font-semibold text-white mb-4">Resultados de Comparación</h3>
          <div className="space-y-4">
            {/* Key Signatures */}
            <div>
              <p className="text-sm text-gray-400 mb-2">Tonalidades</p>
              <div className="flex items-center gap-2">
                {comparisonMutation.data.comparison.key_signatures.keys.map((key, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-purple-500/30 rounded-full text-sm text-white"
                  >
                    {key}
                  </span>
                ))}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                {comparisonMutation.data.comparison.key_signatures.all_same
                  ? 'Todas las canciones tienen la misma tonalidad'
                  : 'Las canciones tienen diferentes tonalidades'}
              </p>
            </div>

            {/* Tempos */}
            <div>
              <p className="text-sm text-gray-400 mb-2">Tempos</p>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-400">Promedio: </span>
                  <span className="text-white font-medium">
                    {formatBPM(comparisonMutation.data.comparison.tempos.average)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Rango: </span>
                  <span className="text-white font-medium">
                    {formatBPM(comparisonMutation.data.comparison.tempos.min)} -{' '}
                    {formatBPM(comparisonMutation.data.comparison.tempos.max)}
                  </span>
                </div>
              </div>
            </div>

            {/* Similarities */}
            {comparisonMutation.data.similarities &&
              comparisonMutation.data.similarities.length > 0 && (
                <div>
                  <p className="text-sm text-gray-400 mb-2 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" />
                    Similitudes
                  </p>
                  <ul className="space-y-1">
                    {comparisonMutation.data.similarities.slice(0, 3).map((sim: any, idx: number) => (
                      <li key={idx} className="text-sm text-green-300">
                        • {sim}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

            {/* Differences */}
            {comparisonMutation.data.differences &&
              comparisonMutation.data.differences.length > 0 && (
                <div>
                  <p className="text-sm text-gray-400 mb-2 flex items-center gap-2">
                    <TrendingDown className="w-4 h-4" />
                    Diferencias
                  </p>
                  <ul className="space-y-1">
                    {comparisonMutation.data.differences.slice(0, 3).map((diff: any, idx: number) => (
                      <li key={idx} className="text-sm text-red-300">
                        • {diff}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
          </div>
        </div>
      )}
    </div>
  );
}

