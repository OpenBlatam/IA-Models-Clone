'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Lightbulb, TrendingUp, Award, Zap } from 'lucide-react';

export function MusicInsights() {
  const { data: analytics } = useQuery({
    queryKey: ['music-analytics'],
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 30000,
  });

  const insights = [];

  if (analytics?.stats) {
    const stats = analytics.stats;

    if (stats.total_tracks > 100) {
      insights.push({
        icon: Award,
        color: 'text-yellow-400',
        title: 'Colección Grande',
        description: `Tienes ${stats.total_tracks} canciones en tu colección`,
      });
    }

    if (stats.success_rate > 0.95) {
      insights.push({
        icon: Zap,
        color: 'text-green-400',
        title: 'Alto Rendimiento',
        description: `${(stats.success_rate * 100).toFixed(1)}% de éxito en análisis`,
      });
    }

    if (stats.avg_response_time < 0.5) {
      insights.push({
        icon: TrendingUp,
        color: 'text-blue-400',
        title: 'Respuesta Rápida',
        description: `Tiempo promedio de ${(stats.avg_response_time * 1000).toFixed(0)}ms`,
      });
    }

    insights.push({
      icon: Lightbulb,
      color: 'text-purple-400',
      title: 'Sugerencia',
      description: 'Explora nuevos géneros para descubrir más música',
    });
  }

  if (insights.length === 0) {
    return null;
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Lightbulb className="w-5 h-5 text-yellow-400" />
        <h3 className="text-lg font-semibold text-white">Insights</h3>
      </div>

      <div className="space-y-3">
        {insights.map((insight, idx) => {
          const Icon = insight.icon;
          return (
            <div
              key={idx}
              className="flex items-start gap-3 p-3 bg-white/5 rounded-lg border border-white/10"
            >
              <Icon className={`w-5 h-5 ${insight.color} flex-shrink-0 mt-0.5`} />
              <div>
                <p className="text-white font-medium text-sm">{insight.title}</p>
                <p className="text-gray-400 text-xs mt-1">{insight.description}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


