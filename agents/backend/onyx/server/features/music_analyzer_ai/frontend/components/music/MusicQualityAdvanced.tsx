'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Award, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

interface MusicQualityAdvancedProps {
  trackId: string;
}

export function MusicQualityAdvanced({ trackId }: MusicQualityAdvancedProps) {
  const { data: quality, isLoading } = useQuery({
    queryKey: ['quality-analysis', trackId],
    queryFn: () => musicApiService.analyzeQuality?.(trackId) || Promise.resolve({}),
    enabled: !!trackId,
  });

  if (isLoading) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 text-center">
        <p className="text-gray-400">Analizando calidad...</p>
      </div>
    );
  }

  if (!quality) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 text-center">
        <Award className="w-16 h-16 text-gray-500 mx-auto mb-4" />
        <p className="text-gray-400">No hay datos de calidad disponibles</p>
      </div>
    );
  }

  const getQualityIcon = (score: number) => {
    if (score >= 0.8) return <CheckCircle className="w-5 h-5 text-green-400" />;
    if (score >= 0.5) return <AlertCircle className="w-5 h-5 text-yellow-400" />;
    return <XCircle className="w-5 h-5 text-red-400" />;
  };

  const qualityMetrics = [
    { label: 'Calidad General', value: quality.overall_quality || 0 },
    { label: 'Producción', value: quality.production_quality || 0 },
    { label: 'Mezcla', value: quality.mixing_quality || 0 },
    { label: 'Masterización', value: quality.mastering_quality || 0 },
  ];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Award className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Análisis de Calidad Avanzado</h3>
      </div>

      <div className="space-y-4">
        {qualityMetrics.map((metric, idx) => (
          <div key={idx}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {getQualityIcon(metric.value)}
                <span className="text-white font-medium">{metric.label}</span>
              </div>
              <span className="text-sm text-gray-400">
                {Math.round(metric.value * 100)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className={`h-3 rounded-full ${
                  metric.value >= 0.8
                    ? 'bg-green-500'
                    : metric.value >= 0.5
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
                }`}
                style={{ width: `${metric.value * 100}%` }}
              />
            </div>
          </div>
        ))}

        {quality.recommendations && quality.recommendations.length > 0 && (
          <div className="mt-6 p-4 bg-white/5 rounded-lg border border-white/10">
            <h4 className="text-white font-medium mb-2">Recomendaciones</h4>
            <ul className="space-y-1">
              {quality.recommendations.map((rec: string, idx: number) => (
                <li key={idx} className="text-sm text-gray-300 flex items-start gap-2">
                  <span className="text-purple-300">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}


