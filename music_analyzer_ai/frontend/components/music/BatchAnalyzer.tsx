'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { Loader2, FileText, CheckCircle } from 'lucide-react';
import toast from 'react-hot-toast';

export function BatchAnalyzer() {
  const [trackIds, setTrackIds] = useState<string>('');
  const [results, setResults] = useState<any[]>([]);

  const batchMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      const analyses = await Promise.all(
        ids.map((id) => musicApiService.analyzeTrack(id, undefined, false))
      );
      return analyses;
    },
    onSuccess: (data) => {
      setResults(data);
      toast.success(`Análisis completado para ${data.length} canciones`);
    },
    onError: () => {
      toast.error('Error en análisis en lote');
    },
  });

  const handleAnalyze = () => {
    const ids = trackIds.split(',').map((id) => id.trim()).filter(Boolean);
    if (ids.length === 0) {
      toast.error('Ingresa al menos un track ID');
      return;
    }
    if (ids.length > 10) {
      toast.error('Máximo 10 canciones por análisis en lote');
      return;
    }
    batchMutation.mutate(ids);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <FileText className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Análisis en Lote</h2>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">
            Track IDs (separados por comas, máximo 10)
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
          disabled={batchMutation.isPending}
          className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {batchMutation.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analizando...
            </>
          ) : (
            <>
              <FileText className="w-5 h-5" />
              Analizar en Lote
            </>
          )}
        </button>

        {results.length > 0 && (
          <div className="mt-6 space-y-3">
            <h3 className="text-lg font-semibold text-white mb-3">
              Resultados ({results.length})
            </h3>
            {results.map((result, idx) => (
              <div
                key={idx}
                className="p-4 bg-white/5 rounded-lg border border-white/10"
              >
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <h4 className="text-white font-medium">
                    {result.track_basic_info?.name || `Track ${idx + 1}`}
                  </h4>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-400">Tonalidad: </span>
                    <span className="text-white">
                      {result.musical_analysis?.key_signature || 'N/A'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Tempo: </span>
                    <span className="text-white">
                      {result.musical_analysis?.tempo?.bpm
                        ? `${Math.round(result.musical_analysis.tempo.bpm)} BPM`
                        : 'N/A'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Energía: </span>
                    <span className="text-white">
                      {result.technical_analysis?.energy?.value
                        ? `${Math.round(result.technical_analysis.energy.value * 100)}%`
                        : 'N/A'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Bailabilidad: </span>
                    <span className="text-white">
                      {result.technical_analysis?.danceability?.value
                        ? `${Math.round(result.technical_analysis.danceability.value * 100)}%`
                        : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}


