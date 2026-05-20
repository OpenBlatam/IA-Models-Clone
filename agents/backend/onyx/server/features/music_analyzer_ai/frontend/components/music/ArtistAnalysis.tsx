'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { User, Loader2, TrendingUp, GitCompare } from 'lucide-react';
import toast from 'react-hot-toast';

export function ArtistAnalysis() {
  const [artistName, setArtistName] = useState('');
  const [analysisType, setAnalysisType] = useState<'evolution' | 'compare'>('evolution');
  const [compareArtists, setCompareArtists] = useState<string>('');

  const evolutionMutation = useMutation({
    mutationFn: (name: string) => musicApiService.getArtistEvolution?.(name) || Promise.resolve({}),
    onSuccess: () => {
      toast.success('Análisis de evolución completado');
    },
    onError: () => {
      toast.error('Error al analizar evolución');
    },
  });

  const compareMutation = useMutation({
    mutationFn: (artists: string[]) => musicApiService.compareArtists?.(artists) || Promise.resolve({}),
    onSuccess: () => {
      toast.success('Comparación de artistas completada');
    },
    onError: () => {
      toast.error('Error al comparar artistas');
    },
  });

  const handleAnalyze = () => {
    if (analysisType === 'evolution') {
      if (!artistName.trim()) {
        toast.error('Ingresa un nombre de artista');
        return;
      }
      evolutionMutation.mutate(artistName);
    } else {
      const artists = compareArtists.split(',').map((a) => a.trim()).filter(Boolean);
      if (artists.length < 2) {
        toast.error('Ingresa al menos 2 artistas separados por comas');
        return;
      }
      compareMutation.mutate(artists);
    }
  };

  const currentData = analysisType === 'evolution' ? evolutionMutation.data : compareMutation.data;
  const isLoading = evolutionMutation.isPending || compareMutation.isPending;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <User className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Análisis de Artistas</h2>
      </div>

      <div className="space-y-4">
        {/* Type Selector */}
        <div className="flex gap-2">
          <button
            onClick={() => setAnalysisType('evolution')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              analysisType === 'evolution'
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            Evolución
          </button>
          <button
            onClick={() => setAnalysisType('compare')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              analysisType === 'compare'
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            Comparar
          </button>
        </div>

        {/* Input */}
        {analysisType === 'evolution' ? (
          <div className="flex gap-2">
            <input
              type="text"
              value={artistName}
              onChange={(e) => setArtistName(e.target.value)}
              placeholder="Nombre del artista..."
              className="flex-1 px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
            />
            <button
              onClick={handleAnalyze}
              disabled={isLoading}
              className="px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <TrendingUp className="w-5 h-5" />
              )}
              Analizar
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            <input
              type="text"
              value={compareArtists}
              onChange={(e) => setCompareArtists(e.target.value)}
              placeholder="Artista 1, Artista 2, Artista 3..."
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
            />
            <button
              onClick={handleAnalyze}
              disabled={isLoading}
              className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <GitCompare className="w-5 h-5" />
              )}
              Comparar Artistas
            </button>
          </div>
        )}

        {/* Results */}
        {currentData && (
          <div className="mt-6 space-y-4">
            {analysisType === 'evolution' && currentData.evolution && (
              <div className="bg-white/5 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-3">Evolución del Artista</h3>
                <div className="space-y-3">
                  {currentData.evolution.periods?.map((period: any, idx: number) => (
                    <div key={idx} className="p-3 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-white font-medium">{period.period || `Período ${idx + 1}`}</span>
                        {period.genres && (
                          <span className="text-sm text-purple-300">
                            {Array.isArray(period.genres) ? period.genres.join(', ') : period.genres}
                          </span>
                        )}
                      </div>
                      {period.changes && (
                        <ul className="space-y-1 text-sm text-gray-300">
                          {period.changes.map((change: string, cIdx: number) => (
                            <li key={cIdx}>• {change}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {analysisType === 'compare' && currentData.comparison && (
              <div className="bg-white/5 rounded-lg p-4 space-y-4">
                <h3 className="text-lg font-semibold text-white mb-3">Comparación de Artistas</h3>
                {Object.entries(currentData.comparison).map(([key, value]: [string, any]) => (
                  <div key={key}>
                    <p className="text-sm text-gray-400 capitalize mb-2">
                      {key.replace(/_/g, ' ')}
                    </p>
                    {typeof value === 'object' ? (
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(value).map(([subKey, subValue]: [string, any]) => (
                          <div key={subKey} className="bg-white/5 rounded p-2">
                            <p className="text-xs text-gray-400">{subKey}</p>
                            <p className="text-white">{String(subValue)}</p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-white">{String(value)}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

