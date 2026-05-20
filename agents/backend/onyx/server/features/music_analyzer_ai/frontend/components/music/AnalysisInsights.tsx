'use client';

import { Lightbulb, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';

interface AnalysisInsightsProps {
  analysis: any;
}

export function AnalysisInsights({ analysis }: AnalysisInsightsProps) {
  const insights = [];

  // Generar insights basados en el análisis
  if (analysis.musical_analysis) {
    if (analysis.musical_analysis.tempo?.bpm) {
      const bpm = analysis.musical_analysis.tempo.bpm;
      if (bpm > 120) {
        insights.push({
          type: 'info',
          icon: TrendingUp,
          message: 'Tempo rápido, ideal para ejercicio o actividades energéticas',
        });
      } else if (bpm < 80) {
        insights.push({
          type: 'info',
          icon: AlertCircle,
          message: 'Tempo lento, perfecto para relajación o estudio',
        });
      }
    }

    if (analysis.technical_analysis?.energy?.value) {
      const energy = analysis.technical_analysis.energy.value;
      if (energy > 0.7) {
        insights.push({
          type: 'success',
          icon: CheckCircle,
          message: 'Alta energía, excelente para motivación',
        });
      }
    }
  }

  if (insights.length === 0) return null;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Lightbulb className="w-5 h-5 text-yellow-400" />
        <h3 className="text-lg font-semibold text-white">Insights Inteligentes</h3>
      </div>

      <div className="space-y-3">
        {insights.map((insight, idx) => {
          const Icon = insight.icon;
          const colorClass =
            insight.type === 'success'
              ? 'text-green-400 bg-green-500/20 border-green-500/30'
              : insight.type === 'error'
              ? 'text-red-400 bg-red-500/20 border-red-500/30'
              : 'text-blue-400 bg-blue-500/20 border-blue-500/30';

          return (
            <div
              key={idx}
              className={`flex items-start gap-3 p-3 rounded-lg border ${colorClass}`}
            >
              <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <p className="text-sm">{insight.message}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}


