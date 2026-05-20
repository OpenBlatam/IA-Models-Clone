'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Award, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

interface QualityAnalysisProps {
  trackId: string;
}

export function QualityAnalysis({ trackId }: QualityAnalysisProps) {
  const qualityMutation = useMutation({
    mutationFn: () => musicApiService.analyzeQuality(trackId),
    onSuccess: () => {
      toast.success('Análisis de calidad completado');
    },
    onError: () => {
      toast.error('Error al analizar calidad');
    },
  });

  const handleAnalyze = () => {
    qualityMutation.mutate();
  };

  const getQualityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getQualityIcon = (score: number) => {
    if (score >= 0.8) return CheckCircle;
    if (score >= 0.6) return AlertCircle;
    return XCircle;
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Award className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Análisis de Calidad</h2>
      </div>

      <button
        onClick={handleAnalyze}
        disabled={qualityMutation.isPending}
        className="w-full mb-6 px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
      >
        {qualityMutation.isPending ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analizando...
          </>
        ) : (
          <>
            <Award className="w-5 h-5" />
            Analizar Calidad
          </>
        )}
      </button>

      {qualityMutation.data && (
        <div className="space-y-4">
          {/* Overall Quality Score */}
          {qualityMutation.data.overall_quality && (
            <div className="bg-white/5 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Calidad General</span>
                <div className="flex items-center gap-2">
                  {(() => {
                    const Icon = getQualityIcon(qualityMutation.data.overall_quality.score);
                    return <Icon className={`w-5 h-5 ${getQualityColor(qualityMutation.data.overall_quality.score)}`} />;
                  })()}
                  <span className={`text-2xl font-bold ${getQualityColor(qualityMutation.data.overall_quality.score)}`}>
                    {Math.round(qualityMutation.data.overall_quality.score * 100)}%
                  </span>
                </div>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3 mt-2">
                <div
                  className={`h-3 rounded-full transition-all ${
                    qualityMutation.data.overall_quality.score >= 0.8
                      ? 'bg-green-400'
                      : qualityMutation.data.overall_quality.score >= 0.6
                      ? 'bg-yellow-400'
                      : 'bg-red-400'
                  }`}
                  style={{ width: `${qualityMutation.data.overall_quality.score * 100}%` }}
                />
              </div>
              {qualityMutation.data.overall_quality.description && (
                <p className="text-sm text-gray-300 mt-2">
                  {qualityMutation.data.overall_quality.description}
                </p>
              )}
            </div>
          )}

          {/* Quality Aspects */}
          {qualityMutation.data.aspects && (
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(qualityMutation.data.aspects).map(([key, value]: [string, any]) => (
                <div key={key} className="bg-white/5 rounded-lg p-3">
                  <p className="text-xs text-gray-400 mb-1 capitalize">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-purple-400 h-2 rounded-full"
                        style={{ width: `${(value.score || value) * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-white font-medium">
                      {Math.round((value.score || value) * 100)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Recommendations */}
          {qualityMutation.data.recommendations && qualityMutation.data.recommendations.length > 0 && (
            <div className="bg-purple-500/20 rounded-lg p-4 border border-purple-400/30">
              <p className="text-sm font-semibold text-white mb-2">Recomendaciones de Mejora</p>
              <ul className="space-y-1">
                {qualityMutation.data.recommendations.map((rec: string, idx: number) => (
                  <li key={idx} className="text-sm text-gray-300">• {rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

