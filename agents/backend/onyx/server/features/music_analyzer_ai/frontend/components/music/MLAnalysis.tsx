'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Brain, Loader2, Sparkles, TrendingUp } from 'lucide-react';
import toast from 'react-hot-toast';

interface MLAnalysisProps {
  trackId: string;
}

export function MLAnalysis({ trackId }: MLAnalysisProps) {
  const [analysisType, setAnalysisType] = useState<'comprehensive' | 'genre' | 'multitask'>(
    'comprehensive'
  );

  const comprehensiveMutation = useMutation({
    mutationFn: () => musicApiService.analyzeComprehensive(trackId),
    onSuccess: () => {
      toast.success('Análisis ML completado');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error en análisis ML');
    },
  });

  const genreMutation = useMutation({
    mutationFn: () => musicApiService.predictGenre(trackId),
    onSuccess: () => {
      toast.success('Predicción de género completada');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error en predicción');
    },
  });

  const multitaskMutation = useMutation({
    mutationFn: () => musicApiService.predictMultiTask(trackId),
    onSuccess: () => {
      toast.success('Predicción multi-tarea completada');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error en predicción');
    },
  });

  const handleAnalyze = () => {
    switch (analysisType) {
      case 'comprehensive':
        comprehensiveMutation.mutate();
        break;
      case 'genre':
        genreMutation.mutate();
        break;
      case 'multitask':
        multitaskMutation.mutate();
        break;
    }
  };

  const currentData =
    analysisType === 'comprehensive'
      ? comprehensiveMutation.data
      : analysisType === 'genre'
      ? genreMutation.data
      : multitaskMutation.data;

  const isLoading =
    comprehensiveMutation.isPending ||
    genreMutation.isPending ||
    multitaskMutation.isPending;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
        <Brain className="w-6 h-6" />
        Análisis con Machine Learning
      </h2>

      {/* Analysis Type Selector */}
      <div className="mb-4">
        <div className="flex gap-2 mb-4">
          {[
            { value: 'comprehensive', label: 'Comprehensivo' },
            { value: 'genre', label: 'Género' },
            { value: 'multitask', label: 'Multi-Tarea' },
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

        <button
          onClick={handleAnalyze}
          disabled={isLoading}
          className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analizando...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              Ejecutar Análisis ML
            </>
          )}
        </button>
      </div>

      {/* Results */}
      {currentData && (
        <div className="mt-6 space-y-4">
          {/* Genre Prediction */}
          {currentData.genre && (
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-2">Género Predicho</p>
              <div className="flex items-center gap-2">
                <span className="text-xl font-semibold text-white">{currentData.genre}</span>
                {currentData.confidence && (
                  <span className="text-sm text-gray-400">
                    ({Math.round(currentData.confidence * 100)}% confianza)
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Emotion Prediction */}
          {currentData.emotion && (
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-2">Emoción Predicha</p>
              <div className="flex items-center gap-2">
                <span className="text-xl font-semibold text-white capitalize">
                  {currentData.emotion}
                </span>
              </div>
            </div>
          )}

          {/* Multi-task Results */}
          {currentData.predictions && (
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-3">Predicciones Multi-Tarea</p>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(currentData.predictions).map(([key, value]: [string, any]) => (
                  <div key={key}>
                    <p className="text-xs text-gray-400 capitalize mb-1">
                      {key.replace(/_/g, ' ')}
                    </p>
                    <p className="text-white font-medium">
                      {typeof value === 'number' ? value.toFixed(2) : String(value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Comprehensive Analysis */}
          {currentData.comprehensive_analysis && (
            <div className="bg-white/5 rounded-lg p-4 space-y-3">
              <p className="text-sm text-gray-400 mb-2 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Análisis Comprehensivo
              </p>
              {Object.entries(currentData.comprehensive_analysis).map(([key, value]: [string, any]) => (
                <div key={key}>
                  <p className="text-xs text-gray-400 capitalize mb-1">
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
  );
}

