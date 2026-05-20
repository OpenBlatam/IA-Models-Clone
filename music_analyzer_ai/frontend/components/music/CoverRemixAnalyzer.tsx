'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Music2, Loader2, Search, GitCompare } from 'lucide-react';
import toast from 'react-hot-toast';

export function CoverRemixAnalyzer() {
  const [trackId, setTrackId] = useState('');
  const [analysisType, setAnalysisType] = useState<'cover' | 'remix' | 'find'>('find');

  const findMutation = useMutation({
    mutationFn: (id: string) => musicApiService.findCoversRemixes?.(id) || Promise.resolve({}),
    onSuccess: () => {
      toast.success('Covers y remixes encontrados');
    },
    onError: () => {
      toast.error('Error al buscar covers/remixes');
    },
  });

  const coverMutation = useMutation({
    mutationFn: (id: string) => musicApiService.analyzeCover?.(id) || Promise.resolve({}),
    onSuccess: () => {
      toast.success('Análisis de cover completado');
    },
    onError: () => {
      toast.error('Error al analizar cover');
    },
  });

  const remixMutation = useMutation({
    mutationFn: (id: string) => musicApiService.analyzeRemix?.(id) || Promise.resolve({}),
    onSuccess: () => {
      toast.success('Análisis de remix completado');
    },
    onError: () => {
      toast.error('Error al analizar remix');
    },
  });

  const handleAnalyze = () => {
    if (!trackId.trim()) {
      toast.error('Ingresa un track ID');
      return;
    }

    switch (analysisType) {
      case 'find':
        findMutation.mutate(trackId);
        break;
      case 'cover':
        coverMutation.mutate(trackId);
        break;
      case 'remix':
        remixMutation.mutate(trackId);
        break;
    }
  };

  const currentData =
    analysisType === 'find'
      ? findMutation.data
      : analysisType === 'cover'
      ? coverMutation.data
      : remixMutation.data;

  const isLoading = findMutation.isPending || coverMutation.isPending || remixMutation.isPending;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Music2 className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Análisis de Covers y Remixes</h2>
      </div>

      <div className="space-y-4">
        {/* Type Selector */}
        <div className="flex gap-2">
          {[
            { value: 'find', label: 'Buscar Versiones' },
            { value: 'cover', label: 'Analizar Cover' },
            { value: 'remix', label: 'Analizar Remix' },
          ].map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setAnalysisType(value as any)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                analysisType === value
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={trackId}
            onChange={(e) => setTrackId(e.target.value)}
            placeholder="Track ID de Spotify..."
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
              <Search className="w-5 h-5" />
            )}
            {analysisType === 'find' ? 'Buscar' : 'Analizar'}
          </button>
        </div>

        {/* Results */}
        {currentData && (
          <div className="mt-6 space-y-4">
            {analysisType === 'find' && currentData.covers && (
              <div className="bg-white/5 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-3">Covers Encontrados</h3>
                <div className="space-y-2">
                  {currentData.covers.map((cover: any, idx: number) => (
                    <div key={idx} className="flex items-center gap-3 p-3 bg-white/5 rounded-lg">
                      <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center">
                        <Music2 className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <p className="text-white font-medium">{cover.name}</p>
                        <p className="text-sm text-gray-300">{cover.artists?.join(', ')}</p>
                        {cover.fidelity_score && (
                          <p className="text-xs text-purple-300">
                            Fidelidad: {Math.round(cover.fidelity_score * 100)}%
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {analysisType === 'find' && currentData.remixes && (
              <div className="bg-white/5 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-3">Remixes Encontrados</h3>
                <div className="space-y-2">
                  {currentData.remixes.map((remix: any, idx: number) => (
                    <div key={idx} className="flex items-center gap-3 p-3 bg-white/5 rounded-lg">
                      <div className="w-12 h-12 rounded bg-pink-500 flex items-center justify-center">
                        <Music2 className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <p className="text-white font-medium">{remix.name}</p>
                        <p className="text-sm text-gray-300">{remix.artists?.join(', ')}</p>
                        {remix.transformation_level && (
                          <p className="text-xs text-pink-300">
                            Transformación: {remix.transformation_level}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {(analysisType === 'cover' || analysisType === 'remix') && currentData.analysis && (
              <div className="bg-white/5 rounded-lg p-4 space-y-3">
                <div className="flex items-center gap-2 mb-3">
                  <GitCompare className="w-5 h-5 text-purple-300" />
                  <h3 className="text-lg font-semibold text-white">Análisis Detallado</h3>
                </div>
                {Object.entries(currentData.analysis).map(([key, value]: [string, any]) => (
                  <div key={key}>
                    <p className="text-sm text-gray-400 capitalize mb-1">
                      {key.replace(/_/g, ' ')}
                    </p>
                    <p className="text-white">
                      {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                    </p>
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

